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
from tqdm.auto import tqdm

from consts import (
    UNIFIED_STATE_SPACE_DIMENSION,
    default_params,
    init_only_params,
    state_var_indices,
    state_vars,
    variational_params,
)
from util import cov_cleanup, fix_title, gale_shapely_matching, model_macro_data

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
        "--uncertainty-P_DAMPS", type=float, default=1.0, required=False
    )
    parser.add_argument("--uncertainty-T1IFN", type=float, default=1.0, required=False)
    parser.add_argument("--uncertainty-TNF", type=float, default=1.0, required=False)
    parser.add_argument("--uncertainty-IFNg", type=float, default=1.0, required=False)
    parser.add_argument("--uncertainty-IL6", type=float, default=1.0, required=False)
    parser.add_argument("--uncertainty-IL1", type=float, default=1.0, required=False)
    parser.add_argument("--uncertainty-IL8", type=float, default=1.0, required=False)
    parser.add_argument("--uncertainty-IL10", type=float, default=1.0, required=False)
    parser.add_argument("--uncertainty-IL12", type=float, default=1.0, required=False)
    parser.add_argument("--uncertainty-IL18", type=float, default=1.0, required=False)
    parser.add_argument(
        "--uncertainty-extracellular_virus", type=float, default=1.0, required=False
    )

    parser.add_argument(
        "--verbose", help="print extra diagnostic messages", action="store_true"
    )

    parser.add_argument("--grid_width", help="width of simulation grid", type=int)
    parser.add_argument("--grid_height", help="height of simulation grid", type=int)

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
        "IL8",
        "extracellular_virus",
    ]
}


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

################################################################################
# constants

# have the models' parameters do a random walk over time (should help
# with covariance starvation)
PARAMETER_RANDOM_WALK: Final[bool] = True

TIME_SPAN: Final[int] = 2016
SAMPLE_INTERVAL: Final[int] = 48  # how often to make measurements
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
mean_vec = np.full(
    (NUM_CYCLES + 1, TIME_SPAN + 1, UNIFIED_STATE_SPACE_DIMENSION), -1, dtype=np.float64
)
cov_matrix = np.full(
    (
        NUM_CYCLES + 1,
        TIME_SPAN + 1,
        UNIFIED_STATE_SPACE_DIMENSION,
        UNIFIED_STATE_SPACE_DIMENSION,
    ),
    -1,
    dtype=np.float64,
)

# collect initial statistics
time = 0
macro_data = np.array([model_macro_data(model) for model in model_ensemble])
mean_vec[time, :] = np.mean(macro_data, axis=0)
cov_matrix[time, :, :] = np.cov(macro_data, rowvar=False)

for cycle in tqdm(range(NUM_CYCLES), desc="cycle #"):
    # advance ensemble of models
    for _ in tqdm(range(SAMPLE_INTERVAL), desc="time steps to prediction"):
        for model in model_ensemble:
            model.time_step()
            if PARAMETER_RANDOM_WALK:
                macrostate = model_macro_data(model)
                random_walk_macrostate = np.abs(
                    macrostate
                    + multivariate_normal(
                        mean=np.zeros_like(macrostate),
                        cov=np.diag(0.01 * np.ones_like(macrostate)),
                    ).rvs()
                )
                modify_model(
                    model,
                    random_walk_macrostate,
                    ignore_state_vars=True,
                    verbose=VERBOSE,
                    state_var_indices=state_var_indices,
                    state_vars=state_vars,
                    variational_params=variational_params,
                )
        time += 1
        macro_data = np.array([model_macro_data(model) for model in model_ensemble])
        mean_vec[cycle:, time, :] = np.mean(macro_data, axis=0)
        cov_matrix[cycle:, time, :, :] = np.cov(macro_data, rowvar=False)

    # make copy of the models and advance them to the end of the simulation time
    model_ensemble_copy = deepcopy(model_ensemble)
    for future_time in tqdm(range(time, TIME_SPAN + 1), desc="time steps to end"):
        for model in model_ensemble_copy:
            model.time_step()
            if PARAMETER_RANDOM_WALK:
                macrostate = model_macro_data(model)
                random_walk_macrostate = np.abs(
                    macrostate
                    + multivariate_normal(
                        mean=np.zeros_like(macrostate),
                        cov=np.diag(0.01 * np.ones_like(macrostate)),
                    ).rvs()
                )
                modify_model(
                    model,
                    random_walk_macrostate,
                    ignore_state_vars=True,
                    verbose=VERBOSE,
                    state_var_indices=state_var_indices,
                    state_vars=state_vars,
                    variational_params=variational_params,
                )
        macro_data = np.array(
            [model_macro_data(model) for model in model_ensemble_copy]
        )
        mean_vec[cycle:, future_time, :] = np.mean(macro_data, axis=0)
        cov_matrix[cycle:, future_time, :, :] = np.cov(macro_data, rowvar=False)

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
            axs[row, col].plot(
                range(TIME_SPAN + 1),
                vp_trajectory[:, idx],
                label="true value",
                color="black",
            )
            axs[row, col].plot(
                range(TIME_SPAN + 1),
                mean_vec[cycle, :, idx],
                label="estimate",
            )
            axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL + 1),
                np.maximum(
                    0.0,
                    mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL + 1, idx]
                    - np.sqrt(
                        cov_matrix[cycle, : (cycle + 1) * SAMPLE_INTERVAL + 1, idx, idx]
                    ),
                ),
                mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL + 1, idx]
                + np.sqrt(
                    cov_matrix[cycle, : (cycle + 1) * SAMPLE_INTERVAL + 1, idx, idx]
                ),
                color="gray",
                alpha=0.35,
                label="past cone of uncertainty",
            )
            axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                np.maximum(
                    0.0,
                    mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx]
                    - np.sqrt(
                        cov_matrix[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]
                    ),
                ),
                mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx]
                + np.sqrt(cov_matrix[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]),
                color="red",  # TODO: pick better color
                alpha=0.35,
                label="future cone of uncertainty",
            )
            axs[row, col].set_title(fix_title(state_var_name), loc="center", wrap=True)
            # fix y-range for the case where the variance is overwhelming
            ymax = min(
                axs[row, col].get_ylim()[1],
                max(
                    1.1
                    * np.max(vp_trajectory[: (cycle + 1) * SAMPLE_INTERVAL + 1, idx]),
                    1.1
                    * np.max(mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL + 1, idx]),
                ),
            )
            if ymax == 0:
                ymax = 1.0
            axs[row, col].set_ylim([0, ymax])
        # remove axes on unused graphs
        for idx in range(
            len(state_vars),
            state_var_graphs_rows * state_var_graphs_cols,
        ):
            row, col = divmod(idx, state_var_graphs_cols)
            axs[row, col].set_axis_off()
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower right")
        fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-state.pdf")
        plt.close(fig)

    ################################################################################
    # plot projection of parameters

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
                axs[row, col].plot(
                    [0, TIME_SPAN + 1],
                    [vp_init_params[param_name]] * 2,
                    label="true value",
                    color="black",
                )
            axs[row, col].plot(
                range(TIME_SPAN + 1),
                mean_vec[cycle, :, len(state_vars) + idx],
                label="estimate",
            )
            axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL),
                np.maximum(
                    0.0,
                    mean_vec[
                        cycle, : (cycle + 1) * SAMPLE_INTERVAL, len(state_vars) + idx
                    ]
                    - np.sqrt(
                        cov_matrix[
                            cycle,
                            : (cycle + 1) * SAMPLE_INTERVAL,
                            len(state_vars) + idx,
                            len(state_vars) + idx,
                        ]
                    ),
                ),
                np.minimum(
                    (
                        10 * vp_init_params[param_name]
                        if param_name in vp_init_params
                        else float("inf")
                    ),
                    mean_vec[
                        cycle, : (cycle + 1) * SAMPLE_INTERVAL, len(state_vars) + idx
                    ]
                    + np.sqrt(
                        cov_matrix[
                            cycle,
                            : (cycle + 1) * SAMPLE_INTERVAL,
                            len(state_vars) + idx,
                            len(state_vars) + idx,
                        ]
                    ),
                ),
                color="gray",
                alpha=0.35,
                label="past cone of uncertainty",
            )
            axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                np.maximum(
                    0.0,
                    mean_vec[
                        cycle, (cycle + 1) * SAMPLE_INTERVAL :, len(state_vars) + idx
                    ]
                    - np.sqrt(
                        cov_matrix[
                            cycle,
                            (cycle + 1) * SAMPLE_INTERVAL :,
                            len(state_vars) + idx,
                            len(state_vars) + idx,
                        ]
                    ),
                ),
                np.minimum(
                    (
                        10 * vp_init_params[param_name]
                        if param_name in vp_init_params
                        else float("inf")
                    ),
                    mean_vec[
                        cycle, (cycle + 1) * SAMPLE_INTERVAL :, len(state_vars) + idx
                    ]
                    + np.sqrt(
                        cov_matrix[
                            cycle,
                            (cycle + 1) * SAMPLE_INTERVAL :,
                            len(state_vars) + idx,
                            len(state_vars) + idx,
                        ]
                    ),
                ),
                color="red",  # TODO: pick better color
                alpha=0.35,
                label="future cone of uncertainty",
            )
            axs[row, col].set_title(fix_title(param_name), loc="center", wrap=True)
            ymax = 1.1 * max(
                np.max(
                    vp_trajectory[
                        : (cycle + 1) * SAMPLE_INTERVAL + 1, len(state_vars) + idx
                    ]
                ),
                np.max(mean_vec[: cycle * SAMPLE_INTERVAL + 1, len(state_vars) + idx]),
            )
            if ymax == 0:
                ymax = 1.0
            axs[row, col].set_ylim([0, ymax])
        # remove axes on unused graphs
        for idx in range(
            len(variational_params),
            variational_params_graphs_rows * variational_params_graphs_cols,
        ):
            row, col = divmod(idx, variational_params_graphs_cols)
            axs[row, col].set_axis_off()

        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower right")
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
            vp_trajectory[time, state_var_indices[obs_name]]
            for obs_name in OBSERVABLE_VAR_NAMES
        ],
        dtype=np.float64,
    )

    v = observation - (H @ mean_vec[cycle, time, :])
    S = H @ cov_matrix[cycle, time, :, :] @ H.T + R
    K = cov_matrix[cycle, time, :, :] @ H.T @ np.linalg.pinv(S)

    mean_vec[cycle + 1, time, :] += K @ v
    cov_matrix[cycle + 1, time, :, :] -= K @ S @ K.T

    cov_matrix[cycle + 1, time, :, :] = cov_cleanup(cov_matrix[cycle + 1, time, :, :])

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
            new_sample[model_to_sample_pairing[model_idx], :],
            verbose=VERBOSE,
            state_var_indices=state_var_indices,
            state_vars=state_vars,
            variational_params=variational_params,
        )

################################################################################

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
            axs[row, col].plot(
                range(TIME_SPAN + 1),
                vp_trajectory[:, idx],
                label="true value",
                color="black",
            )

            axs[row, col].plot(
                range(TIME_SPAN + 1),
                mean_vec[cycle, :, idx],
                label="old estimate",
            )
            axs[row, col].plot(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                mean_vec[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, idx],
                label="updated estimate",
                color="red",
            )

            axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL + 1),
                np.maximum(
                    0.0,
                    mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL + 1, idx]
                    - np.sqrt(
                        cov_matrix[cycle, : (cycle + 1) * SAMPLE_INTERVAL + 1, idx, idx]
                    ),
                ),
                mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL + 1, idx]
                + np.sqrt(
                    cov_matrix[cycle, : (cycle + 1) * SAMPLE_INTERVAL + 1, idx, idx]
                ),
                color="gray",
                alpha=0.35,
                label="past cone of uncertainty",
            )
            axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                np.maximum(
                    0.0,
                    mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx]
                    - np.sqrt(
                        cov_matrix[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]
                    ),
                ),
                mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx]
                + np.sqrt(cov_matrix[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]),
                color="red",  # TODO: pick better color
                alpha=0.35,
                label="old future cone of uncertainty",
            )
            axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                np.maximum(
                    0.0,
                    mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx]
                    - np.sqrt(
                        cov_matrix[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]
                    ),
                ),
                mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx]
                + np.sqrt(cov_matrix[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]),
                color="green",  # TODO: pick better color
                alpha=0.35,
                label="new future cone of uncertainty",
            )

            axs[row, col].set_title(fix_title(state_var_name), loc="center", wrap=True)
            # fix y-range for the case where the variance is overwhelming
            ymax = min(
                axs[row, col].get_ylim()[1],
                max(
                    1.1
                    * np.max(vp_trajectory[: (cycle + 1) * SAMPLE_INTERVAL + 1, idx]),
                    1.1
                    * np.max(mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL + 1, idx]),
                ),
            )
            if ymax == 0:
                ymax = 1.0
            axs[row, col].set_ylim([0, ymax])
        # remove axes on unused graphs
        for idx in range(
            len(state_vars),
            state_var_graphs_rows * state_var_graphs_cols,
        ):
            row, col = divmod(idx, state_var_graphs_cols)
            axs[row, col].set_axis_off()
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower right")
        fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-state-kfupd.pdf")
        plt.close(fig)

################################################################################
# plot kalman update of parameters

if GRAPHS:
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
                axs[row, col].plot(
                    [0, TIME_SPAN + 1],
                    [vp_init_params[param_name]] * 2,
                    label="true value",
                    color="black",
                )

            axs[row, col].plot(
                range(TIME_SPAN + 1),
                mean_vec[cycle, :, len(state_vars) + idx],
                label="old estimate",
            )
            axs[row, col].plot(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                mean_vec[
                    cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, len(state_vars) + idx
                ],
                label="updated estimate",
                color="red",
            )

            axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL),
                np.maximum(
                    0.0,
                    mean_vec[
                        cycle, : (cycle + 1) * SAMPLE_INTERVAL, len(state_vars) + idx
                    ]
                    - np.sqrt(
                        cov_matrix[
                            cycle,
                            : (cycle + 1) * SAMPLE_INTERVAL,
                            len(state_vars) + idx,
                            len(state_vars) + idx,
                        ]
                    ),
                ),
                np.minimum(
                    (
                        10 * vp_init_params[param_name]
                        if param_name in vp_init_params
                        else float("inf")
                    ),
                    mean_vec[
                        cycle, : (cycle + 1) * SAMPLE_INTERVAL, len(state_vars) + idx
                    ]
                    + np.sqrt(
                        cov_matrix[
                            cycle,
                            : (cycle + 1) * SAMPLE_INTERVAL,
                            len(state_vars) + idx,
                            len(state_vars) + idx,
                        ]
                    ),
                ),
                color="gray",
                alpha=0.35,
                label="past cone of uncertainty",
            )
            axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                np.maximum(
                    0.0,
                    mean_vec[
                        cycle, (cycle + 1) * SAMPLE_INTERVAL :, len(state_vars) + idx
                    ]
                    - np.sqrt(
                        cov_matrix[
                            cycle,
                            (cycle + 1) * SAMPLE_INTERVAL :,
                            len(state_vars) + idx,
                            len(state_vars) + idx,
                        ]
                    ),
                ),
                np.minimum(
                    (
                        10 * vp_init_params[param_name]
                        if param_name in vp_init_params
                        else float("inf")
                    ),
                    mean_vec[
                        cycle, (cycle + 1) * SAMPLE_INTERVAL :, len(state_vars) + idx
                    ]
                    + np.sqrt(
                        cov_matrix[
                            cycle,
                            (cycle + 1) * SAMPLE_INTERVAL :,
                            len(state_vars) + idx,
                            len(state_vars) + idx,
                        ]
                    ),
                ),
                color="red",  # TODO: pick better color
                alpha=0.35,
                label="old future cone of uncertainty",
            )
            axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                np.maximum(
                    0.0,
                    mean_vec[
                        cycle + 1,
                        (cycle + 1) * SAMPLE_INTERVAL :,
                        len(state_vars) + idx,
                    ]
                    - np.sqrt(
                        cov_matrix[
                            cycle + 1,
                            (cycle + 1) * SAMPLE_INTERVAL :,
                            len(state_vars) + idx,
                            len(state_vars) + idx,
                        ]
                    ),
                ),
                np.minimum(
                    (
                        10 * vp_init_params[param_name]
                        if param_name in vp_init_params
                        else float("inf")
                    ),
                    mean_vec[
                        cycle + 1,
                        (cycle + 1) * SAMPLE_INTERVAL :,
                        len(state_vars) + idx,
                    ]
                    + np.sqrt(
                        cov_matrix[
                            cycle + 1,
                            (cycle + 1) * SAMPLE_INTERVAL :,
                            len(state_vars) + idx,
                            len(state_vars) + idx,
                        ]
                    ),
                ),
                color="green",  # TODO: pick better color
                alpha=0.35,
                label="new future cone of uncertainty",
            )

            axs[row, col].set_title(fix_title(param_name), loc="center", wrap=True)
            ymax = 1.1 * max(
                np.max(
                    vp_trajectory[
                        : (cycle + 1) * SAMPLE_INTERVAL + 1, len(state_vars) + idx
                    ]
                ),
                np.max(mean_vec[: cycle * SAMPLE_INTERVAL + 1, len(state_vars) + idx]),
            )
            if ymax == 0:
                ymax = 1.0
            axs[row, col].set_ylim([0, ymax])
        # remove axes on unused graphs
        for idx in range(
            len(variational_params),
            variational_params_graphs_rows * variational_params_graphs_cols,
        ):
            row, col = divmod(idx, variational_params_graphs_cols)
            axs[row, col].set_axis_off()

        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower right")
        fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-params-kfupd.pdf")
        plt.close(fig)

################################################################################
# calculate surprisal information

#####
# full surprisal: all state vars and params
delta_full = mean_vec - vp_trajectory
_, logdet = np.linalg.slogdet(cov_matrix)
surprisal_quadratic_part = np.einsum(
    "cij,cij->ci", delta_full, np.linalg.solve(cov_matrix, delta_full)
)
surprisal_full = (
    surprisal_quadratic_part
    + logdet
    + UNIFIED_STATE_SPACE_DIMENSION * np.log(2 * np.pi)
) / 2.0

# average of surprisal over all time
# note: dt = 1 so integral is just mean
surprisal_average_full = np.mean(surprisal_full, axis=1)

# average of surprisal over all _future_ times
future_surprisal_average_full = np.array(
    [
        np.mean(surprisal_full[cycle, (cycle + 1) * SAMPLE_INTERVAL :])
        for cycle in range(NUM_CYCLES - 1)
    ]
)

#####
# state surprisal: restrict to just the state vars
vp_state_trajectory = vp_trajectory[:, : len(state_vars)]

delta_state = mean_vec[:, :, :3] - vp_state_trajectory
_, logdet = np.linalg.slogdet(cov_matrix[:, :, :3, :3])
surprisal_quadratic_part = np.einsum(
    "cij,cij->ci", delta_state, np.linalg.solve(cov_matrix[:, :, :3, :3], delta_state)
)
surprisal_state = (
    surprisal_quadratic_part + logdet + len(state_vars) * np.log(2 * np.pi)
) / 2.0

# average of state surprisal over all time
# note: dt = 1 so integral is just mean
surprisal_average_state = np.mean(surprisal_state, axis=1)

# average of state surprisal over all _future_ times
future_surprisal_average_state = np.array(
    [
        np.mean(surprisal_state[cycle, (cycle + 1) * SAMPLE_INTERVAL :])
        for cycle in range(NUM_CYCLES - 1)
    ]
)

#####
# param surprisal: restrict to just the params
vp_param_trajectory = vp_trajectory[:, len(state_vars) :]

delta_param = mean_vec[:, :, len(state_vars) :] - vp_param_trajectory
_, logdet = np.linalg.slogdet(cov_matrix[:, :, len(state_vars) :, len(state_vars) :])
surprisal_quadratic_part = np.einsum(
    "cij,cij->ci",
    delta_param,
    np.linalg.solve(
        cov_matrix[:, :, len(state_vars) :, len(state_vars) :], delta_param
    ),
)
surprisal_param = (
    surprisal_quadratic_part + logdet + len(variational_params) * np.log(2 * np.pi)
) / 2.0

# average of state surprisal over all time
# note: dt = 1 so integral is just mean
surprisal_average_param = np.mean(surprisal_param, axis=1)

# average of state surprisal over all _future_ times
future_surprisal_average_param = np.array(
    [
        np.mean(surprisal_param[cycle, (cycle + 1) * SAMPLE_INTERVAL :])
        for cycle in range(NUM_CYCLES - 1)
    ]
)

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
