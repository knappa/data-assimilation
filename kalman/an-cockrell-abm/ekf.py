#!/usr/bin/env python3
import argparse
import sys
from copy import deepcopy
from typing import Dict, Final, List, Tuple

import an_cockrell
import h5py
import matplotlib.pyplot as plt
import numpy as np
from an_cockrell import AnCockrellModel
from scipy.stats import multivariate_normal
from sklearn.covariance import LedoitWolf
from tqdm.auto import tqdm

from consts import (
    UNIFIED_STATE_SPACE_DIMENSION,
    default_params,
    init_only_params,
    state_var_indices,
    state_vars,
    variational_params,
)
from transform import transform_intrinsic_to_kf, transform_kf_to_intrinsic
from util import fix_title, gale_shapely_matching, model_macro_data, slogdet

################################################################################

if hasattr(sys, "ps1"):
    # interactive mode
    args = object()
else:
    parser = argparse.ArgumentParser()

    parser.add_argument("--prefix", type=str, default="", help="output file prefix")

    parser.add_argument(
        "--measurements",
        type=str,
        choices=[
            "P_DAMPS",
            "T1IFN",
            "TNF",
            "IFNg",
            "IL1",
            "IL6",
            "IL8",
            "IL10",
            "IL12",
            "IL18",
            "extracellular_virus",
        ],
        nargs="+",
        required=True,
        help="which things to measure (required)",
    )

    parser.add_argument("--graphs", help="make pdf graphs", action="store_true")

    parser.add_argument(
        "--update-algorithm",
        type=str,
        choices=["simple", "spatial", "full-spatial"],
        required=True,
    )

    # parameters for the measurement uncertainty (coeffs in the Kalman filter's R matrix)
    parser.add_argument(
        "--uncertainty-P_DAMPS", type=float, default=0.001, required=False
    )
    parser.add_argument(
        "--uncertainty-T1IFN", type=float, default=0.001, required=False
    )
    parser.add_argument("--uncertainty-TNF", type=float, default=0.001, required=False)
    parser.add_argument("--uncertainty-IFNg", type=float, default=0.001, required=False)
    parser.add_argument("--uncertainty-IL6", type=float, default=0.001, required=False)
    parser.add_argument("--uncertainty-IL1", type=float, default=0.001, required=False)
    parser.add_argument("--uncertainty-IL8", type=float, default=0.001, required=False)
    parser.add_argument("--uncertainty-IL10", type=float, default=0.001, required=False)
    parser.add_argument("--uncertainty-IL12", type=float, default=0.001, required=False)
    parser.add_argument("--uncertainty-IL18", type=float, default=0.001, required=False)
    parser.add_argument(
        "--uncertainty-extracellular_virus", type=float, default=0.001, required=False
    )

    parser.add_argument(
        "--param_stoch_level",
        help="stochasticity parameter for parameter random walk",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--verbose", help="print extra diagnostic messages", action="store_true"
    )

    parser.add_argument(
        "--predict",
        type=str,
        choices=["to-kf-update", "to-next-kf-update", "to-end"],
        help="how far to extend predictions",
        required=True,
    )

    parser.add_argument("--grid_width", help="width of simulation grid", type=int)
    parser.add_argument("--grid_height", help="height of simulation grid", type=int)

    parser.add_argument(
        "--time_span",
        help="number of abm iterations in a full run",
        type=int,
        default=2016,
    )
    parser.add_argument(
        "--sample_interval",
        help="iterations in the interval between samples",
        type=int,
        default=48,
    )

    args = parser.parse_args()

################################################################################
# command line option interpretation / backfilling those options when you,
# for example, paste the code into ipython

if hasattr(args, "grid_width"):
    default_params["GRID_WIDTH"] = args.grid_width

if hasattr(args, "grid_height"):
    default_params["GRID_HEIGHT"] = args.grid_height

VERBOSE: Final[bool] = False if not hasattr(args, "verbose") else args.verbose

modification_algorithm: Final[str] = (
    "spatial" if not hasattr(args, "update_algorithm") else args.update_algorithm
)

if modification_algorithm == "full-spatial":
    from modify_full_spatial import modify_model
elif modification_algorithm == "spatial":
    from modify_epi_spatial import modify_model
else:
    from modify_simple import modify_model

# rs encodes the uncertainty in the various observations. Defaults are 1.0 if
# not set via the command line. (See above for defaults if you are. Right now,
# they are also 1.0.)
rs: Final[Dict[str, float]] = {
    "total_"
    + name: (
        1.0
        if not hasattr(args, "uncertainty_" + name)
        else getattr(args, "uncertainty_" + name)
    )
    for name in [
        "P_DAMPS",
        "T1IFN",
        "TNF",
        "IFNg",
        "IL1",
        "IL6",
        "IL8",
        "IL10",
        "IL12",
        "IL18",
        "extracellular_virus",
    ]
}

PARAM_STOCH_LEVEL: Final[float] = (
    0.001
    if not hasattr(args, "param_stoch_level")
    or args.param_stoch_level is None
    or args.param_stoch_level == -1
    else args.param_stoch_level
)

ENSEMBLE_SIZE: Final[int] = (
    (UNIFIED_STATE_SPACE_DIMENSION + 1) * UNIFIED_STATE_SPACE_DIMENSION // 2
)  # max(50, (UNIFIED_STATE_SPACE_DIMENSION + 1))
OBSERVABLES: Final[List[str]] = (
    ["extracellular_virus"] if not hasattr(args, "measurements") else args.measurements
)
OBSERVABLE_VAR_NAMES: Final[List[str]] = ["total_" + name for name in OBSERVABLES]

FILE_PREFIX: Final[str] = (
    "" if not hasattr(args, "prefix") or len(args.prefix) == 0 else args.prefix + "-"
)

GRAPHS: Final[bool] = True if not hasattr(args, "graphs") else bool(args.graphs)

PREDICT: Final[str] = "to-kf-update" if not hasattr(args, "predict") else args.predict

TIME_SPAN: Final[int] = (
    2016 if not hasattr(args, "time_span") or args.time_span <= 0 else args.time_span
)
SAMPLE_INTERVAL: Final[int] = (
    48
    if not hasattr(args, "sample_interval") or args.sample_interval <= 0
    else args.sample_interval
)

################################################################################
# constants

# have the models' parameters do a random walk over time (should help
# with covariance starvation)
PARAMETER_RANDOM_WALK: Final[bool] = True

NUM_CYCLES: Final[int] = TIME_SPAN // SAMPLE_INTERVAL

################################################################################
# graph layout computations

# layout for graphing state variables.
# Attempts to be mostly square, with possibly more rows than columns
state_var_graphs_cols: Final[int] = int(np.floor(np.sqrt(len(state_vars))))
state_var_graphs_rows: Final[int] = int(
    np.ceil(len(state_vars) / state_var_graphs_cols)
)
state_var_graphs_figsize: Final[Tuple[float, float]] = (
    1.8 * state_var_graphs_rows,
    1.8 * state_var_graphs_cols,
)

# layout for graphing parameters.
# Attempts to be mostly square, with possibly more rows than columns
variational_params_graphs_cols: Final[int] = int(
    np.floor(np.sqrt(len(variational_params)))
)
variational_params_graphs_rows: Final[int] = int(
    np.ceil(len(variational_params) / variational_params_graphs_cols)
)
variational_params_graphs_figsize: Final[Tuple[float, float]] = (
    1.8 * variational_params_graphs_rows,
    1.8 * variational_params_graphs_cols,
)

################################################################################
# statistical parameters

init_mean_vec = np.array(
    [default_params[param] for param in (init_only_params + variational_params)]
)

init_cov_matrix = np.diag(
    np.array(
        [
            0.75 * np.sqrt(default_params[param])
            for param in (init_only_params + variational_params)
        ]
    )
)

################################################################################
# sample a virtual patient

# sampled virtual patient parameters
vp_init_params = default_params.copy()
vp_init_param_sample = np.abs(
    multivariate_normal(mean=init_mean_vec, cov=init_cov_matrix).rvs()
)
for sample_component, param_name in zip(
    vp_init_param_sample,
    (init_only_params + variational_params),
):
    vp_init_params[param_name] = (
        round(sample_component)
        if isinstance(default_params[param_name], int)
        else sample_component
    )

# create model for virtual patient
virtual_patient_model = an_cockrell.AnCockrellModel(**vp_init_params)

# evaluate the virtual patient's trajectory
vp_trajectory = np.zeros(
    (TIME_SPAN + 1, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64
)

vp_trajectory[0, :] = model_macro_data(virtual_patient_model)
# noinspection PyTypeChecker
for t in tqdm(range(1, TIME_SPAN + 1), desc="create virtual patient"):
    virtual_patient_model.time_step()
    vp_trajectory[t, :] = model_macro_data(virtual_patient_model)

################################################################################
# plot virtual patient

if GRAPHS:
    fig, axs = plt.subplots(
        nrows=state_var_graphs_rows,
        ncols=state_var_graphs_cols,
        figsize=state_var_graphs_figsize,
        sharex=True,
        sharey=False,
    )
    for idx, state_var_name in enumerate(state_vars):
        row, col = divmod(idx, state_var_graphs_cols)
        axs[row, col].plot(vp_trajectory[:, idx])

        axs[row, col].set_title(
            fix_title(state_var_name),
            loc="center",
            wrap=True,
        )
        axs[row, col].set_ylim(bottom=max(0.0, axs[row, col].get_ylim()[0]))
    for idx in range(len(state_vars), state_var_graphs_rows * state_var_graphs_cols):
        row, col = divmod(idx, state_var_graphs_cols)
        axs[row, col].set_axis_off()
    fig.tight_layout()
    fig.savefig(FILE_PREFIX + "virtual-patient-trajectory.pdf")
    plt.close(fig)


################################################################################


def model_ensemble_from(means, covariances):
    """
    Create an ensemble of models from a distribution. Uses init-only
    and variational parameters

    :param means:
    :param covariances:
    :return:
    """
    mdl_ensemble = []
    distribution = multivariate_normal(mean=means, cov=covariances, allow_singular=True)
    for _ in range(ENSEMBLE_SIZE):
        model_param_dict = default_params.copy()
        sampled_params = np.abs(distribution.rvs())
        # noinspection PyShadowingNames
        for sample_component, parameter_name in zip(
            sampled_params,
            (init_only_params + variational_params),
        ):
            model_param_dict[parameter_name] = (
                round(sample_component)
                if isinstance(default_params[parameter_name], int)
                else sample_component
            )
        # create model for virtual patient
        model = AnCockrellModel(**model_param_dict)
        mdl_ensemble.append(model)

    return mdl_ensemble


################################################################################
# Kalman filter simulation
################################################################################

# create ensemble of models for kalman filter
model_ensemble = model_ensemble_from(init_mean_vec, init_cov_matrix)

# mean and covariances through time
mean_vec = np.zeros(
    (NUM_CYCLES + 1, TIME_SPAN + 1, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64
)
cov_matrix = np.zeros(
    (
        NUM_CYCLES + 1,
        TIME_SPAN + 1,
        UNIFIED_STATE_SPACE_DIMENSION,
        UNIFIED_STATE_SPACE_DIMENSION,
    ),
    dtype=np.float64,
)

lw = LedoitWolf(assume_centered=False)

# collect initial statistics
time = 0
macro_data = np.array(
    [transform_intrinsic_to_kf(model_macro_data(model)) for model in model_ensemble]
)
# mean_vec[:, time, :] = np.mean(macro_data, axis=0)
# cov_matrix[:, time, :, :] = np.cov(macro_data, rowvar=False)
lw.fit(macro_data)
mean_vec[:, time, :] = lw.location_
cov_matrix[:, time, :, :] = lw.covariance_

for cycle in tqdm(range(NUM_CYCLES), desc="cycle"):
    # advance ensemble of models
    for _ in tqdm(range(SAMPLE_INTERVAL), desc="time steps to prediction"):
        for model in model_ensemble:
            model.time_step()
            if PARAMETER_RANDOM_WALK:
                macrostate = transform_intrinsic_to_kf(model_macro_data(model))
                random_walk_macrostate = np.abs(
                    macrostate
                    + multivariate_normal(
                        mean=np.zeros_like(macrostate),
                        cov=np.diag(PARAM_STOCH_LEVEL * np.ones_like(macrostate)),
                    ).rvs()
                )
                modify_model(
                    model,
                    transform_kf_to_intrinsic(random_walk_macrostate),
                    ignore_state_vars=True,
                    verbose=VERBOSE,
                    state_var_indices=state_var_indices,
                    state_vars=state_vars,
                    variational_params=variational_params,
                )
        time += 1
        macro_data = np.array(
            [
                transform_intrinsic_to_kf(model_macro_data(model))
                for model in model_ensemble
            ]
        )
        # mean_vec[cycle:, time, :] = np.mean(macro_data, axis=0)
        # cov_matrix[cycle:, time, :, :] = np.cov(macro_data, rowvar=False)
        lw.fit(macro_data)
        mean_vec[cycle:, time, :] = lw.location_
        cov_matrix[cycle:, time, :, :] = lw.covariance_

    if PREDICT != "to-kf-update":
        # make copy of the models and advance them to the end of the simulation time
        model_ensemble_copy = deepcopy(model_ensemble)
        if PREDICT == "to-next-kf-update":
            final_time = time + SAMPLE_INTERVAL + 1
        else:
            # PREDICT == "to-end"
            final_time = TIME_SPAN + 1
        for future_time in tqdm(
            range(time, min(TIME_SPAN + 1, final_time)),
            desc="time steps past kf-update",
        ):
            for model in model_ensemble_copy:
                model.time_step()
                if PARAMETER_RANDOM_WALK:
                    macrostate = transform_intrinsic_to_kf(model_macro_data(model))
                    random_walk_macrostate = np.abs(
                        macrostate
                        + multivariate_normal(
                            mean=np.zeros_like(macrostate),
                            cov=np.diag(0.01 * np.ones_like(macrostate)),
                        ).rvs()
                    )
                    modify_model(
                        model,
                        transform_kf_to_intrinsic(random_walk_macrostate),
                        ignore_state_vars=True,
                        verbose=VERBOSE,
                        state_var_indices=state_var_indices,
                        state_vars=state_vars,
                        variational_params=variational_params,
                    )
            macro_data = np.array(
                [
                    transform_intrinsic_to_kf(model_macro_data(model))
                    for model in model_ensemble_copy
                ]
            )
            # mean_vec[cycle:, future_time, :] = np.mean(macro_data, axis=0)
            # cov_matrix[cycle:, future_time, :, :] = np.cov(macro_data, rowvar=False)
            lw.fit(macro_data)
            mean_vec[cycle:, future_time, :] = lw.location_
            cov_matrix[cycle:, future_time, :, :] = lw.covariance_

    ################################################################################
    # plot projection of state variables

    if GRAPHS:
        fig, axs = plt.subplots(
            nrows=state_var_graphs_rows,
            ncols=state_var_graphs_cols,
            figsize=state_var_graphs_figsize,
            sharex=True,
            sharey=False,
            layout="constrained",
        )
        for idx, state_var_name in enumerate(state_vars):
            row, col = divmod(idx, state_var_graphs_cols)
            (true_value,) = axs[row, col].plot(
                range(TIME_SPAN + 1),
                vp_trajectory[:, idx],
                label="true value",
                linestyle=":",
                color="gray",
            )
            (past_estimate_center_line,) = axs[row, col].plot(
                range((cycle + 1) * SAMPLE_INTERVAL),
                transform_kf_to_intrinsic(
                    mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx], index=idx
                ),
                label="estimate of past",
                color="black",
            )
            past_estimate_range = axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL),
                np.maximum(
                    0.0,
                    transform_kf_to_intrinsic(
                        mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx]
                        - np.sqrt(
                            cov_matrix[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx, idx]
                        ),
                        index=idx,
                    ),
                ),
                # np.minimum(
                #     10 * max_scales[state_var_name],
                transform_kf_to_intrinsic(
                    mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx]
                    + np.sqrt(
                        cov_matrix[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx, idx]
                    ),
                    index=idx,
                ),
                # ),
                color="gray",
                alpha=0.35,
            )

            mu = mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx]
            sigma = np.sqrt(
                cov_matrix[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]
            )

            (prediction_center_line,) = axs[row, col].plot(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                transform_kf_to_intrinsic(mu, index=idx),
                label="prediction",
                color="blue",
            )
            prediction_range = axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                np.maximum(0.0, transform_kf_to_intrinsic(mu - sigma, index=idx)),
                transform_kf_to_intrinsic(
                    mu + sigma,
                    index=idx,
                ),
                color="blue",
                alpha=0.35,
            )
            axs[row, col].set_title(fix_title(state_var_name), loc="left", wrap=True)
            axs[row, col].set_ylim(bottom=max(0.0, axs[row, col].get_ylim()[0]))
        # remove axes on unused graphs
        for idx in range(
            len(state_vars),
            state_var_graphs_rows * state_var_graphs_cols,
        ):
            row, col = divmod(idx, state_var_graphs_cols)
            axs[row, col].set_axis_off()

        # place legend
        if len(state_vars) < state_var_graphs_rows * state_var_graphs_cols:
            legend_placement = axs[state_var_graphs_rows - 1, state_var_graphs_cols - 1]
            legend_loc = "upper left"
        else:
            legend_placement = fig
            legend_loc = "outside lower center"

        # noinspection PyUnboundLocalVariable
        legend_placement.legend(
            [
                true_value,
                (past_estimate_center_line, past_estimate_range),
                (prediction_center_line, prediction_range),
            ],
            [
                true_value.get_label(),
                past_estimate_center_line.get_label(),
                prediction_center_line.get_label(),
            ],
            loc=legend_loc,
        )
        fig.suptitle("State Prediction")
        fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-state.pdf")
        plt.close(fig)

    ################################################################################
    # plot projection of parameters

    len_state_vars = len(state_vars)
    if GRAPHS:
        fig, axs = plt.subplots(
            nrows=variational_params_graphs_rows,
            ncols=variational_params_graphs_cols,
            figsize=variational_params_graphs_figsize,
            sharex=True,
            sharey=False,
            layout="constrained",
        )
        for idx, param_name in enumerate(variational_params):
            row, col = divmod(idx, variational_params_graphs_cols)

            if param_name in vp_init_params:
                (true_value,) = axs[row, col].plot(
                    [0, TIME_SPAN + 1],
                    [vp_init_params[param_name]] * 2,
                    label="true value",
                    color="gray",
                    linestyle=":",
                )

            (past_estimate_center_line,) = axs[row, col].plot(
                range((cycle + 1) * SAMPLE_INTERVAL),
                transform_kf_to_intrinsic(
                    mean_vec[
                        cycle, : (cycle + 1) * SAMPLE_INTERVAL, len_state_vars + idx
                    ],
                    index=len_state_vars + idx,
                ),
                color="black",
                label="estimate of past",
            )

            past_estimate_range = axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL),
                np.maximum(
                    0.0,
                    transform_kf_to_intrinsic(
                        mean_vec[
                            cycle, : (cycle + 1) * SAMPLE_INTERVAL, len_state_vars + idx
                        ]
                        - np.sqrt(
                            cov_matrix[
                                cycle,
                                : (cycle + 1) * SAMPLE_INTERVAL,
                                len_state_vars + idx,
                                len_state_vars + idx,
                            ]
                        ),
                        index=len_state_vars + idx,
                    ),
                ),
                transform_kf_to_intrinsic(
                    mean_vec[
                        cycle, : (cycle + 1) * SAMPLE_INTERVAL, len_state_vars + idx
                    ]
                    + np.sqrt(
                        cov_matrix[
                            cycle,
                            : (cycle + 1) * SAMPLE_INTERVAL,
                            len_state_vars + idx,
                            len_state_vars + idx,
                        ]
                    ),
                    index=len_state_vars + idx,
                ),
                color="gray",
                alpha=0.35,
                label="past cone of uncertainty",
            )

            (prediction_center_line,) = axs[row, col].plot(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                transform_kf_to_intrinsic(
                    mean_vec[
                        cycle, (cycle + 1) * SAMPLE_INTERVAL :, len_state_vars + idx
                    ],
                    index=len_state_vars + idx,
                ),
                label="predictive estimate",
                color="blue",
            )

            prediction_range = axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                np.maximum(
                    0.0,
                    transform_kf_to_intrinsic(
                        mean_vec[
                            cycle, (cycle + 1) * SAMPLE_INTERVAL :, len_state_vars + idx
                        ]
                        - np.sqrt(
                            cov_matrix[
                                cycle,
                                (cycle + 1) * SAMPLE_INTERVAL :,
                                len_state_vars + idx,
                                len_state_vars + idx,
                            ]
                        ),
                        index=len_state_vars + idx,
                    ),
                ),
                transform_kf_to_intrinsic(
                    mean_vec[
                        cycle, (cycle + 1) * SAMPLE_INTERVAL :, len_state_vars + idx
                    ]
                    + np.sqrt(
                        cov_matrix[
                            cycle,
                            (cycle + 1) * SAMPLE_INTERVAL :,
                            len_state_vars + idx,
                            len_state_vars + idx,
                        ]
                    ),
                    index=len_state_vars + idx,
                ),
                color="blue",
                alpha=0.35,
            )
            axs[row, col].set_title(fix_title(param_name), loc="center", wrap=True)
            axs[row, col].set_ylim(bottom=max(0.0, axs[row, col].get_ylim()[0]))

        # remove axes on unused graphs
        for idx in range(
            len(variational_params),
            variational_params_graphs_rows * variational_params_graphs_cols,
        ):
            row, col = divmod(idx, variational_params_graphs_cols)
            axs[row, col].set_axis_off()

        # place legend
        if (
            len(state_vars)
            < variational_params_graphs_rows * variational_params_graphs_cols
        ):
            legend_placement = axs[
                variational_params_graphs_rows - 1, variational_params_graphs_cols - 1
            ]
            legend_loc = "upper left"
        else:
            legend_placement = fig
            legend_loc = "outside lower center"

        legend_placement.legend(
            [
                true_value,
                (past_estimate_center_line, past_estimate_range),
                (prediction_center_line, prediction_range),
            ],
            [
                true_value.get_label(),
                past_estimate_center_line.get_label(),
                prediction_center_line.get_label(),
            ],
            loc=legend_loc,
        )
        fig.suptitle("Parameter Projection")
        fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-params.pdf")
        plt.close(fig)

    ################################################################################
    # Kalman filter

    num_observables = len(OBSERVABLE_VAR_NAMES)

    R = np.diag([rs[obs_name] for obs_name in OBSERVABLE_VAR_NAMES])

    H = np.zeros((num_observables, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64)
    for h_idx, obs_name in enumerate(OBSERVABLE_VAR_NAMES):
        H[h_idx, state_var_indices[obs_name]] = 1.0

    observation = np.array(
        [
            transform_intrinsic_to_kf(
                vp_trajectory[time, state_var_indices[obs_name]],
                index=state_var_indices[obs_name],
            )
            for obs_name in OBSERVABLE_VAR_NAMES
        ],
        dtype=np.float64,
    )

    v = observation - (H @ mean_vec[cycle, time, :])
    S = H @ cov_matrix[cycle, time, :, :] @ H.T + R
    K = cov_matrix[cycle, time, :, :] @ H.T @ np.linalg.pinv(S)

    mean_vec[cycle + 1, time, :] += K @ v
    # cov_matrix[cycle + 1, time, :, :] -= K @ S @ K.T
    # Joseph form update (See e.g. https://www.anuncommonlab.com/articles/how-kalman-filters-work/part2.html)
    A = np.eye(cov_matrix.shape[-1]) - K @ H
    cov_matrix[cycle + 1, time, :, :] = np.nan_to_num(
        A @ cov_matrix[cycle + 1, time, :, :] @ A.T + K @ R @ K.T
    )
    min_diag = np.min(np.diag(cov_matrix[cycle + 1, time, :, :]))
    if min_diag <= 0.0:
        cov_matrix[cycle + 1, time, :, :] += (1e-6 - min_diag) * np.eye(
            cov_matrix.shape[-1]
        )

    ################################################################################
    # recreate ensemble with new distribution

    new_sample = multivariate_normal(
        mean=mean_vec[cycle + 1, time, :],
        cov=cov_matrix[cycle + 1, time, :, :],
        allow_singular=True,
    ).rvs(size=ENSEMBLE_SIZE)

    # Gale-Shapely matching algorithm to try and pair up the models and these new samples
    model_to_sample_pairing = gale_shapely_matching(
        new_sample=new_sample, macro_data=macro_data
    )

    # now do the model modifications
    for model_idx in tqdm(range(ENSEMBLE_SIZE), desc="model modifications"):
        modify_model(
            model_ensemble[model_idx],
            transform_kf_to_intrinsic(
                new_sample[model_to_sample_pairing[model_idx], :]
            ),
            verbose=VERBOSE,
            state_var_indices=state_var_indices,
            state_vars=state_vars,
            variational_params=variational_params,
        )

################################################################################
# plot kalman update of state variables

if GRAPHS:
    for cycle in range(NUM_CYCLES - 1):
        fig, axs = plt.subplots(
            nrows=state_var_graphs_rows,
            ncols=state_var_graphs_cols,
            figsize=state_var_graphs_figsize,
            sharex=True,
            sharey=False,
            layout="constrained",
        )
        for idx, state_var_name in enumerate(state_vars):
            row, col = divmod(idx, state_var_graphs_cols)

            (true_value,) = axs[row, col].plot(
                range(TIME_SPAN + 1),
                vp_trajectory[:, idx],
                label="true value",
                color="black",
            )

            (past_est_center_line,) = axs[row, col].plot(
                range((cycle + 1) * SAMPLE_INTERVAL),
                transform_kf_to_intrinsic(
                    mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx], index=idx
                ),
                color="green",
                label="estimate of past",
            )

            past_est_range = axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL),
                np.maximum(
                    0.0,
                    transform_kf_to_intrinsic(
                        mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx]
                        - np.sqrt(
                            cov_matrix[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx, idx]
                        ),
                        index=idx,
                    ),
                ),
                transform_kf_to_intrinsic(
                    mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx]
                    + np.sqrt(
                        cov_matrix[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx, idx]
                    ),
                    index=idx,
                ),
                color="green",
                alpha=0.35,
            )

            (future_est_before_update_center_line,) = axs[row, col].plot(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                transform_kf_to_intrinsic(
                    mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx], index=idx
                ),
                color="#d5b60a",
                label="previous future estimate",
            )
            future_est_before_update_range = axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                np.maximum(
                    0.0,
                    transform_kf_to_intrinsic(
                        mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx]
                        - np.sqrt(
                            cov_matrix[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]
                        ),
                        index=idx,
                    ),
                ),
                transform_kf_to_intrinsic(
                    mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx]
                    + np.sqrt(
                        cov_matrix[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]
                    ),
                    index=idx,
                ),
                color="#d5b60a",
                alpha=0.35,
            )

            (future_est_after_update_center_line,) = axs[row, col].plot(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                transform_kf_to_intrinsic(
                    mean_vec[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, idx], index=idx
                ),
                label="updated future estimate",
                color="blue",
            )

            future_est_after_update_range = axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                np.maximum(
                    0.0,
                    transform_kf_to_intrinsic(
                        mean_vec[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, idx]
                        - np.sqrt(
                            cov_matrix[
                                cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx
                            ]
                        ),
                        index=idx,
                    ),
                ),
                transform_kf_to_intrinsic(
                    mean_vec[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, idx]
                    + np.sqrt(
                        cov_matrix[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]
                    ),
                    index=idx,
                ),
                color="blue",
                alpha=0.35,
            )
            axs[row, col].set_title(fix_title(state_var_name), loc="left", wrap=True)
            axs[row, col].set_ylim(bottom=max(0.0, axs[row, col].get_ylim()[0]))
        # remove axes on unused graphs
        for idx in range(
            len(state_vars),
            state_var_graphs_rows * state_var_graphs_cols,
        ):
            row, col = divmod(idx, state_var_graphs_cols)
            axs[row, col].set_axis_off()

        # place legend
        if len(state_vars) < state_var_graphs_rows * state_var_graphs_cols:
            legend_placement = axs[state_var_graphs_rows - 1, state_var_graphs_cols - 1]
            legend_loc = "upper left"
        else:
            legend_placement = fig
            legend_loc = "outside lower center"

        # noinspection PyUnboundLocalVariable
        legend_placement.legend(
            [
                true_value,
                (past_est_center_line, past_est_range),
                (future_est_before_update_center_line, future_est_before_update_range),
                (future_est_after_update_center_line, future_est_after_update_range),
            ],
            [
                true_value.get_label(),
                past_est_center_line.get_label(),
                future_est_before_update_center_line.get_label(),
                future_est_after_update_center_line.get_label(),
            ],
            loc=legend_loc,
        )
        fig.suptitle("State Projection", ha="left")
        fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-state-kfupd.pdf")
        plt.close(fig)

################################################################################
# plot kalman update of parameters

if GRAPHS:
    len_state_vars = len(state_vars)
    for cycle in range(NUM_CYCLES - 1):
        fig, axs = plt.subplots(
            nrows=variational_params_graphs_rows,
            ncols=variational_params_graphs_cols,
            figsize=variational_params_graphs_figsize,
            sharex=True,
            sharey=False,
            layout="constrained",
        )
        for idx, param_name in enumerate(variational_params):
            row, col = divmod(idx, variational_params_graphs_cols)

            if param_name in vp_init_params:
                (true_value,) = axs[row, col].plot(
                    [0, TIME_SPAN + 1],
                    [vp_init_params[param_name]] * 2,
                    label="true value",
                    color="black",
                )

            (past_est_center_line,) = axs[row, col].plot(
                range((cycle + 1) * SAMPLE_INTERVAL),
                transform_kf_to_intrinsic(
                    mean_vec[
                        cycle, : (cycle + 1) * SAMPLE_INTERVAL, len_state_vars + idx
                    ],
                    index=len_state_vars + idx,
                ),
                label="estimate of past",
                color="green",
            )

            past_est_range = axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL),
                np.maximum(
                    0.0,
                    transform_kf_to_intrinsic(
                        mean_vec[
                            cycle, : (cycle + 1) * SAMPLE_INTERVAL, len_state_vars + idx
                        ]
                        - np.sqrt(
                            cov_matrix[
                                cycle,
                                : (cycle + 1) * SAMPLE_INTERVAL,
                                len_state_vars + idx,
                                len_state_vars + idx,
                            ]
                        ),
                        index=len_state_vars + idx,
                    ),
                ),
                transform_kf_to_intrinsic(
                    mean_vec[
                        cycle, : (cycle + 1) * SAMPLE_INTERVAL, len_state_vars + idx
                    ]
                    + np.sqrt(
                        cov_matrix[
                            cycle,
                            : (cycle + 1) * SAMPLE_INTERVAL,
                            len_state_vars + idx,
                            len_state_vars + idx,
                        ]
                    ),
                    index=len_state_vars + idx,
                ),
                color="green",
                alpha=0.35,
            )

            (future_est_before_update_center_line,) = axs[row, col].plot(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                transform_kf_to_intrinsic(
                    mean_vec[
                        cycle, (cycle + 1) * SAMPLE_INTERVAL :, len_state_vars + idx
                    ],
                    index=len_state_vars + idx,
                ),
                label="old estimate",
                color="#d5b60a",
            )

            future_est_before_update_range = axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                np.maximum(
                    0.0,
                    transform_kf_to_intrinsic(
                        mean_vec[
                            cycle, (cycle + 1) * SAMPLE_INTERVAL :, len_state_vars + idx
                        ]
                        - np.sqrt(
                            cov_matrix[
                                cycle,
                                (cycle + 1) * SAMPLE_INTERVAL :,
                                len_state_vars + idx,
                                len_state_vars + idx,
                            ]
                        ),
                        index=len_state_vars + idx,
                    ),
                ),
                transform_kf_to_intrinsic(
                    mean_vec[
                        cycle, (cycle + 1) * SAMPLE_INTERVAL :, len_state_vars + idx
                    ]
                    + np.sqrt(
                        cov_matrix[
                            cycle,
                            (cycle + 1) * SAMPLE_INTERVAL :,
                            len_state_vars + idx,
                            len_state_vars + idx,
                        ]
                    ),
                    index=len_state_vars + idx,
                ),
                color="#d5b60a",
                alpha=0.35,
            )

            (future_est_after_update_center_line,) = axs[row, col].plot(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                transform_kf_to_intrinsic(
                    mean_vec[
                        cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, len_state_vars + idx
                    ],
                    index=len_state_vars + idx,
                ),
                label="updated estimate",
                color="blue",
            )

            future_est_after_update_range = axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                np.maximum(
                    0.0,
                    transform_kf_to_intrinsic(
                        mean_vec[
                            cycle + 1,
                            (cycle + 1) * SAMPLE_INTERVAL :,
                            len_state_vars + idx,
                        ]
                        - np.sqrt(
                            cov_matrix[
                                cycle + 1,
                                (cycle + 1) * SAMPLE_INTERVAL :,
                                len_state_vars + idx,
                                len_state_vars + idx,
                            ]
                        ),
                        index=len_state_vars + idx,
                    ),
                ),
                transform_kf_to_intrinsic(
                    mean_vec[
                        cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, len_state_vars + idx
                    ]
                    + np.sqrt(
                        cov_matrix[
                            cycle + 1,
                            (cycle + 1) * SAMPLE_INTERVAL :,
                            len_state_vars + idx,
                            len_state_vars + idx,
                        ]
                    ),
                    index=len_state_vars + idx,
                ),
                color="blue",
                alpha=0.35,
                label="new future cone of uncertainty",
            )
            axs[row, col].set_title(fix_title(param_name), loc="center", wrap=True)
            axs[row, col].set_ylim(bottom=max(0.0, axs[row, col].get_ylim()[0]))

        # remove axes on unused graphs
        for idx in range(
            len(variational_params),
            variational_params_graphs_rows * variational_params_graphs_cols,
        ):
            row, col = divmod(idx, variational_params_graphs_cols)
            axs[row, col].set_axis_off()

        # place the legend
        if (
            len(variational_params)
            < variational_params_graphs_rows * variational_params_graphs_cols
        ):
            legend_placement = axs[
                variational_params_graphs_rows - 1, variational_params_graphs_cols - 1
            ]
            legend_loc = "upper left"
        else:
            legend_placement = fig
            legend_loc = "outside lower center"

        legend_placement.legend(
            [
                true_value,
                (past_est_center_line, past_est_range),
                (future_est_before_update_center_line, future_est_before_update_range),
                (future_est_after_update_center_line, future_est_after_update_range),
            ],
            [
                true_value.get_label(),
                past_est_center_line.get_label(),
                future_est_before_update_center_line.get_label(),
                future_est_after_update_center_line.get_label(),
            ],
            loc=legend_loc,
        )
        fig.suptitle("Parameter Projection", x=0, ha="left")
        fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-params-kfupd.pdf")
        plt.close(fig)

################################################################################
# calculate surprisal information

#####
# full surprisal: all state vars and params
delta_full = mean_vec - transform_intrinsic_to_kf(vp_trajectory)
_, logdet = slogdet(cov_matrix)
sigma_inv_delta = np.array(
    [
        [
            np.linalg.lstsq(
                cov_matrix[cycle, t_idx, :, :], delta_full[cycle, t_idx, :], rcond=None
            )[0]
            for t_idx in range(cov_matrix.shape[1])
        ]
        for cycle in range(NUM_CYCLES + 1)
    ]
)
surprisal_quadratic_part = np.einsum("cij,cij->ci", delta_full, sigma_inv_delta)
surprisal_full = (
    surprisal_quadratic_part
    + logdet
    + UNIFIED_STATE_SPACE_DIMENSION * np.log(2 * np.pi)
) / 2.0

#####
# state surprisal: restrict to just the state vars
delta_state = (
    mean_vec[:, :, : len(state_vars)]
    - transform_intrinsic_to_kf(vp_trajectory)[:, : len(state_vars)]
)
_, logdet = slogdet(cov_matrix[:, :, : len(state_vars), : len(state_vars)])
sigma_inv_delta = np.array(
    [
        [
            np.linalg.lstsq(
                cov_matrix[cycle, t_idx, : len(state_vars), : len(state_vars)],
                delta_state[cycle, t_idx, :],
                rcond=None,
            )[0]
            for t_idx in range(cov_matrix.shape[1])
        ]
        for cycle in range(NUM_CYCLES + 1)
    ]
)
surprisal_quadratic_part = np.einsum("cij,cij->ci", delta_state, sigma_inv_delta)
surprisal_state = (
    surprisal_quadratic_part + logdet + len(state_vars) * np.log(2 * np.pi)
) / 2.0

#####
# param surprisal: restrict to just the params
vp_param_trajectory = transform_intrinsic_to_kf(vp_trajectory)[:, len(state_vars) :]

delta_param = mean_vec[:, :, len(state_vars) :] - vp_param_trajectory
_, logdet = slogdet(cov_matrix[:, :, len(state_vars) :, len(state_vars) :])
sigma_inv_delta = np.array(
    [
        [
            np.linalg.lstsq(
                cov_matrix[cycle, t_idx, len(state_vars) :, len(state_vars) :],
                delta_param[cycle, t_idx, :],
                rcond=None,
            )[0]
            for t_idx in range(cov_matrix.shape[1])
        ]
        for cycle in range(NUM_CYCLES + 1)
    ]
)
surprisal_quadratic_part = np.einsum("cij,cij->ci", delta_param, sigma_inv_delta)
surprisal_param = (
    surprisal_quadratic_part + logdet + len(variational_params) * np.log(2 * np.pi)
) / 2.0

################################################################################

# see the dimension label information here:
# https://docs.h5py.org/en/latest/high/dims.html

with h5py.File(FILE_PREFIX + "data.hdf5", "w") as f:
    f["virtual_patient_trajectory"] = vp_trajectory
    f["virtual_patient_trajectory"].dims[0].label = "time"
    f["virtual_patient_trajectory"].dims[1].label = "state component"

    f["means"] = mean_vec
    f["means"].dims[0].label = "kalman update number"
    f["means"].dims[1].label = "time"
    f["means"].dims[2].label = "state component"

    f["covs"] = cov_matrix
    f["covs"].dims[0].label = "kalman update number"
    f["covs"].dims[1].label = "time"
    f["covs"].dims[2].label = "state component"
    f["covs"].dims[3].label = "state component"

    f["surprisal_full"] = surprisal_full
    f["surprisal_full"].dims[0].label = "kalman update number"
    f["surprisal_full"].dims[1].label = "time"

    f["surprisal_state"] = surprisal_state
    f["surprisal_state"].dims[0].label = "kalman update number"
    f["surprisal_state"].dims[1].label = "time"

    f["surprisal_param"] = surprisal_param
    f["surprisal_param"].dims[0].label = "kalman update number"
    f["surprisal_param"].dims[1].label = "time"
