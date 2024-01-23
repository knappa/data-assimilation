import argparse
import sys
from typing import Dict

import an_cockrell
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
from util import cov_cleanup, model_macro_data

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
            "T1IFN",
            "TNF",
            "IFNg",
            "IL6",
            "IL1",
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

    parser.add_argument(
        "--matchmaker",
        help="try to match resampled macrostates with microstate "
        "models to minimize change magnitude",
        type=str,
        choices=["yes", "no"],
        required=True,
    )

    parser.add_argument("--graphs", help="make pdf graphs", action="store_true")

    parser.add_argument(
        "--update-algorithm", type=str, choices=["simple", "spatial"], required=True
    )

    args = parser.parse_args()

VERBOSE = False

modification_algorithm = (
    "spatial" if not hasattr(args, "update-algorithm") else args.update_algorithm
)

if modification_algorithm == "spatial":
    from modify_epi_spatial import modify_model
else:
    from modify_simple import modify_model

################################################################################
# constants


# layout for graphing state variables.
# Attempts to be mostly square, with possibly more rows than columns
state_var_graphs_cols: int = int(np.floor(np.sqrt(len(state_vars))))
state_var_graphs_rows: int = int(np.ceil(len(state_vars) / state_var_graphs_cols))
state_var_graphs_figsize = (1.8 * state_var_graphs_rows, 1.8 * state_var_graphs_cols)

# layout for graphing parameters.
# Attempts to be mostly square, with possibly more rows than columns
variational_params_graphs_cols: int = int(np.floor(np.sqrt(len(variational_params))))
variational_params_graphs_rows: int = int(
    np.ceil(len(variational_params) / variational_params_graphs_cols)
)
variational_params_graphs_figsize = (
    1.8 * variational_params_graphs_rows,
    1.8 * variational_params_graphs_cols,
)

assert all(param in default_params for param in variational_params)

TIME_SPAN = 2016
SAMPLE_INTERVAL = 48  # how often to make measurements

ENSEMBLE_SIZE = (
    (UNIFIED_STATE_SPACE_DIMENSION + 1) * UNIFIED_STATE_SPACE_DIMENSION // 2
)  # max(50, (UNIFIED_STATE_SPACE_DIMENSION + 1))
OBSERVABLES = (
    ["extracellular_virus"] if not hasattr(args, "measurements") else args.measurements
)
OBSERVABLE_VAR_NAMES = ["total_" + name for name in OBSERVABLES]

RESAMPLE_MODELS = False

# if we are altering the models (as opposed to resampling) try to match the
# models to minimize the changes necessary.
MODEL_MATCHMAKER = (
    True if not hasattr(args, "matchmaker") else (args.matchmaker == "yes")
)

# have the models' parameters do a random walk over time (should help
# with covariance starvation)
PARAMETER_RANDOM_WALK = True

FILE_PREFIX = "" if not hasattr(args, "prefix") else args.prefix + "-"

GRAPHS = True if not hasattr(args, "graphs") else bool(args.graphs)

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
        row, col = idx // state_var_graphs_cols, idx % state_var_graphs_cols
        axs[row, col].plot(vp_trajectory[:, idx])
        axs[row, col].set_title(state_var_name)
    fig.tight_layout()
    fig.savefig(FILE_PREFIX + "virtual-patient.pdf")
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
        for sample_component, param_name in zip(
            sampled_params,
            (init_only_params + variational_params),
        ):
            model_param_dict[param_name] = (
                round(sample_component)
                if isinstance(default_params[param_name], int)
                else sample_component
            )
        # create model for virtual patient
        model = AnCockrellModel(**model_param_dict)
        mdl_ensemble.append(model)

    return mdl_ensemble


################################################################################


################################################################################
# Kalman filter simulation
################################################################################

# create ensemble of models for kalman filter
model_ensemble = model_ensemble_from(init_mean_vec, init_cov_matrix)

# mean and covariances through time
mean_vec = np.full((TIME_SPAN + 1, UNIFIED_STATE_SPACE_DIMENSION), -1, dtype=np.float64)
cov_matrix = np.full(
    (TIME_SPAN + 1, UNIFIED_STATE_SPACE_DIMENSION, UNIFIED_STATE_SPACE_DIMENSION),
    -1,
    dtype=np.float64,
)

# collect initial statistics
time = 0
initial_macro_data = np.array([model_macro_data(model) for model in model_ensemble])
mean_vec[time, :] = np.mean(initial_macro_data, axis=0)
cov_matrix[time, :, :] = np.cov(initial_macro_data, rowvar=False)
# with warnings.catch_warnings(action="ignore", category=UserWarning):
#     cov_obj = covariance.LedoitWolf().fit(initial_macro_data)
#     # noinspection PyUnresolvedReferences
#     mean_vec[time, :] = cov_obj.location_
#     # noinspection PyUnresolvedReferences
#     cov_matrix[time, :, :], _ = covariance.graphical_lasso(cov_obj.covariance_, 1.0)
#     # # noinspection PyUnresolvedReferences
#     # cov_matrix[time,:,:] = cov_obj.covariance_

cycle = 0
while time < TIME_SPAN:
    cycle += 1
    print(f" *** {cycle=} *** ")
    # advance ensemble of models
    for _ in tqdm(range(SAMPLE_INTERVAL), desc="time step"):
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
        mean_vec[time, :] = np.mean(macro_data, axis=0)
        cov_matrix[time, :, :] = np.cov(macro_data, rowvar=False)
        # with warnings.catch_warnings(action="ignore", category=UserWarning):
        #     cov_obj = covariance.LedoitWolf().fit(macro_data)
        #     # noinspection PyUnresolvedReferences
        #     mean_vec[time, :] = cov_obj.location_
        #     # noinspection PyUnresolvedReferences
        #     cov_matrix[time, :, :], _ = covariance.graphical_lasso(cov_obj.covariance_, 1.0)
        #     # # noinspection PyUnresolvedReferences
        #     # cov_matrix[time,:,:] = cov_obj.covariance_

    ################################################################################
    # plot state variables

    if GRAPHS:
        fig, axs = plt.subplots(
            nrows=state_var_graphs_rows,
            ncols=state_var_graphs_cols,
            figsize=state_var_graphs_figsize,
            sharex=True,
            sharey=False,
        )
        for idx, state_var_name in enumerate(state_vars):
            row, col = idx // state_var_graphs_cols, idx % state_var_graphs_cols
            axs[row, col].plot(
                vp_trajectory[: (cycle + 1) * SAMPLE_INTERVAL + 1, idx],
                label="true value",
                color="black",
            )
            axs[row, col].plot(
                range(cycle * SAMPLE_INTERVAL + 1),
                mean_vec[: cycle * SAMPLE_INTERVAL + 1, idx],
                label="estimate",
            )
            axs[row, col].fill_between(
                range(cycle * SAMPLE_INTERVAL + 1),
                np.maximum(
                    0.0,
                    mean_vec[: cycle * SAMPLE_INTERVAL + 1, idx]
                    - np.sqrt(cov_matrix[: cycle * SAMPLE_INTERVAL + 1, idx, idx]),
                ),
                mean_vec[: cycle * SAMPLE_INTERVAL + 1, idx]
                + np.sqrt(cov_matrix[: cycle * SAMPLE_INTERVAL + 1, idx, idx]),
                color="gray",
                alpha=0.35,
            )
            axs[row, col].set_title(state_var_name)
            ymax = max(
                1.1 * np.max(vp_trajectory[: (cycle + 1) * SAMPLE_INTERVAL + 1, idx]),
                1.1 * np.max(mean_vec[: cycle * SAMPLE_INTERVAL + 1, idx]),
            )
            if ymax == 0:
                ymax = 1.0
            axs[row, col].set_ylim([0, ymax])
            # axs[row, col].legend()
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center")
        fig.tight_layout()
        fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-match.pdf")
        plt.close(fig)

    ################################################################################
    # plot parameters

    if GRAPHS:
        fig, axs = plt.subplots(
            nrows=variational_params_graphs_rows,
            ncols=variational_params_graphs_cols,
            figsize=variational_params_graphs_figsize,
            sharex=True,
            sharey=False,
        )
        for idx, param_name in enumerate(variational_params):
            row, col = (
                idx // variational_params_graphs_cols,
                idx % variational_params_graphs_cols,
            )

            if param_name in vp_init_params:
                axs[row, col].plot(
                    [0, (cycle + 1) * SAMPLE_INTERVAL + 1],
                    [vp_init_params[param_name]] * 2,
                    label="true value",
                    color="black",
                )
            axs[row, col].plot(
                range(cycle * SAMPLE_INTERVAL + 1),
                mean_vec[: cycle * SAMPLE_INTERVAL + 1, len(state_vars) + idx],
                label="estimate",
            )
            axs[row, col].fill_between(
                range(cycle * SAMPLE_INTERVAL + 1),
                np.maximum(
                    0.0,
                    mean_vec[: cycle * SAMPLE_INTERVAL + 1, len(state_vars) + idx]
                    - np.sqrt(
                        cov_matrix[
                            : cycle * SAMPLE_INTERVAL + 1,
                            len(state_vars) + idx,
                            len(state_vars) + idx,
                        ]
                    ),
                ),
                np.minimum(
                    10 * vp_init_params[param_name]
                    if param_name in vp_init_params
                    else float("inf"),
                    mean_vec[: cycle * SAMPLE_INTERVAL + 1, len(state_vars) + idx]
                    + np.sqrt(
                        cov_matrix[
                            : cycle * SAMPLE_INTERVAL + 1,
                            len(state_vars) + idx,
                            len(state_vars) + idx,
                        ]
                    ),
                ),
                color="gray",
                alpha=0.35,
            )
            axs[row, col].set_title(param_name)
            ymax = 1.1 * max(
                np.max(
                    vp_trajectory[
                        : (cycle + 1) * SAMPLE_INTERVAL + 1, len(state_vars) + idx
                    ]
                ),
                np.max(mean_vec[: cycle * SAMPLE_INTERVAL + 1, len(state_vars) + idx]),
            )
            # most_of_variation = np.percentile((mean_vec[: cycle * SAMPLE_INTERVAL + 1, len(state_vars) + idx]
            # + np.sqrt(
            #     cov_matrix[
            #     : cycle * SAMPLE_INTERVAL + 1,
            #     len(state_vars) + idx,
            #     len(state_vars) + idx,
            #     ]
            # )),0.9)
            if ymax == 0:
                ymax = 1.0
            axs[row, col].set_ylim([0, ymax])
        # remove axes on unused graphs
        for idx in range(
            len(variational_params),
            variational_params_graphs_rows * variational_params_graphs_cols,
        ):
            row, col = (
                idx // variational_params_graphs_cols,
                idx % variational_params_graphs_cols,
            )
            axs[row, col].axis("off")

        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center")
        fig.tight_layout()
        fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-match-params.pdf")
        plt.close(fig)

    ################################################################################
    # Kalman filter

    num_observables = len(OBSERVABLE_VAR_NAMES)

    # rs encodes the uncertainty in the various observations
    rs: Dict[str, float] = {
        # "total_T1IFN": 1.0,
        # "total_TNF": 1.0,
        # "total_IFNg": 1.0,
        # "total_IL6": 1.0,
        # "total_IL1": 1.0,
        # "total_IL8": 1.0,
        # "total_IL10": 1.0,
        # "total_IL12": 1.0,
        # "total_IL18": 1.0,
        "total_extracellular_virus": 1.0,
    }
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

    v = observation - (H @ mean_vec[time, :])
    S = H @ cov_matrix[time, :, :] @ H.T + R
    K = cov_matrix[time, :, :] @ H.T @ np.linalg.pinv(S)

    mean_vec[time, :] += K @ v
    cov_matrix[time, :, :] -= K @ S @ K.T

    cov_matrix[time, :, :] = cov_cleanup(cov_matrix[time, :, :])

    # recreate ensemble
    if RESAMPLE_MODELS:
        # create an entirely new set of model instances sampled from KF-learned distribution
        model_ensemble = model_ensemble_from(mean_vec[time, :], cov_matrix[time, :, :])
    else:
        dist = multivariate_normal(
            mean=mean_vec[time, :], cov=cov_matrix[time, :, :], allow_singular=True
        )
        if MODEL_MATCHMAKER:
            new_sample = dist.rvs(size=ENSEMBLE_SIZE)
            # Gale-Shapely matching algorithm to try and pair up the models and these new samples

            # fill out preference lists for the models
            prefs = np.zeros((ENSEMBLE_SIZE, ENSEMBLE_SIZE), dtype=np.int64)
            for idx in range(ENSEMBLE_SIZE):
                # noinspection PyUnboundLocalVariable
                dists = np.linalg.norm(new_sample - macro_data[idx], axis=1)
                prefs[idx, :] = np.argsort(dists)

            # arrays to record pairings
            model_to_sample_pairing = np.full(ENSEMBLE_SIZE, -1, dtype=np.int64)
            sample_to_model_pairing = np.full(ENSEMBLE_SIZE, -1, dtype=np.int64)

            all_paired = False
            while not all_paired:
                all_paired = True
                for model_idx in range(ENSEMBLE_SIZE):
                    if model_to_sample_pairing[model_idx] != -1:
                        # skip already paired models
                        continue
                    # found an unpaired model, find the first thing not yet
                    # checked on its preference list
                    min_pref_idx = np.argmax(prefs[model_idx, :] >= 0)
                    for pref_idx in range(min_pref_idx, ENSEMBLE_SIZE):
                        possible_sample_pair = prefs[model_idx, pref_idx]
                        competitor_model_idx = sample_to_model_pairing[
                            possible_sample_pair
                        ]
                        if competitor_model_idx == -1:
                            # if the sample is unpaired, pair the two
                            sample_to_model_pairing[possible_sample_pair] = model_idx
                            model_to_sample_pairing[model_idx] = possible_sample_pair
                            # erase this possibility for future pairings
                            prefs[model_idx, pref_idx] = -1
                            break  # stop looking now
                        else:
                            # compare preferences
                            established_pair_dist = np.linalg.norm(
                                macro_data[competitor_model_idx, :]
                                - new_sample[possible_sample_pair, :]
                            )
                            proposed_pair_dist = np.linalg.norm(
                                macro_data[model_idx, :]
                                - new_sample[possible_sample_pair, :]
                            )
                            if proposed_pair_dist < established_pair_dist:
                                model_to_sample_pairing[
                                    competitor_model_idx
                                ] = -1  # free the competitor
                                all_paired = False
                                # make new pair
                                sample_to_model_pairing[
                                    possible_sample_pair
                                ] = model_idx
                                model_to_sample_pairing[
                                    model_idx
                                ] = possible_sample_pair
                                # erase this possibility for future pairings
                                prefs[model_idx, pref_idx] = -1
                                break  # stop looking now
                            else:
                                prefs[model_idx, pref_idx] = -1  # this one didn't work
                                continue

            # now do the model modifications
            for model_idx in range(ENSEMBLE_SIZE):
                modify_model(
                    model_ensemble[model_idx],
                    new_sample[model_to_sample_pairing[model_idx], :],
                    verbose=VERBOSE,
                    state_var_indices=state_var_indices,
                    state_vars=state_vars,
                    variational_params=variational_params,
                )
        else:
            # sample from KF-learned dist and modify existing models to fit
            for model in model_ensemble:
                state = dist.rvs()
                modify_model(
                    model,
                    state,
                    verbose=VERBOSE,
                    state_var_indices=state_var_indices,
                    state_vars=state_vars,
                    variational_params=variational_params,
                )
#
# ################################################################################
#
# vp_full_trajectory = np.array(
#     (
#         vp_wolf_counts,
#         vp_sheep_counts,
#         vp_grass_counts,
#         [vp_wolf_gain_from_food] * (TIME_SPAN + 1),
#         [vp_sheep_gain_from_food] * (TIME_SPAN + 1),
#         [vp_wolf_reproduce] * (TIME_SPAN + 1),
#         [vp_sheep_reproduce] * (TIME_SPAN + 1),
#         [vp_grass_regrowth_time] * (TIME_SPAN + 1),
#     )
# ).T
#
# delta_full = mean_vec - vp_full_trajectory
# surprisal_full = np.einsum("ij,ij->i", delta_full, np.linalg.solve(cov_matrix, delta_full))
# mean_surprisal_full = np.mean(surprisal_full)
#
# vp_state_trajectory = np.array(
#     (
#         vp_wolf_counts,
#         vp_sheep_counts,
#         vp_grass_counts,
#     )
# ).T
# delta_state = mean_vec[:, :3] - vp_state_trajectory
# surprisal_state = np.einsum(
#     "ij,ij->i", delta_state, np.linalg.solve(cov_matrix[:, :3, :3], delta_state)
# )
# mean_surprisal_state = np.mean(surprisal_state)
#
# vp_param_trajectory = np.array(
#     (
#         [vp_wolf_gain_from_food] * (TIME_SPAN + 1),
#         [vp_sheep_gain_from_food] * (TIME_SPAN + 1),
#         [vp_wolf_reproduce] * (TIME_SPAN + 1),
#         [vp_sheep_reproduce] * (TIME_SPAN + 1),
#         [vp_grass_regrowth_time] * (TIME_SPAN + 1),
#     )
# ).T
# delta_param = mean_vec[:, 3:] - vp_param_trajectory
# surprisal_param = np.einsum(
#     "ij,ij->i", delta_param, np.linalg.solve(cov_matrix[:, 3:, 3:], delta_param)
# )
# mean_surprisal_param = np.mean(surprisal_param)
#
# if GRAPHS:
#     plt.plot(surprisal_full, label="full surprisal")
#     plt.plot(surprisal_state, label="state surprisal")
#     plt.plot(surprisal_param, label="param surprisal")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(FILE_PREFIX + f"surprisal.pdf")
#     plt.close()
#
# if GRAPHS:
#     fig, axs = plt.subplots(4, figsize=(6, 8))
#     plural = {"wolf": "wolves", "sheep": "sheep", "grass": "grass"}
#     vp_data = {
#         "wolf": vp_wolf_counts,
#         "sheep": vp_sheep_counts,
#         "grass": vp_grass_counts,
#     }
#     max_scales = {
#         "wolf": 10 * mean_init_wolves,
#         "sheep": 10 * mean_init_sheep,
#         "grass": 10 * mean_init_grass_proportion * GRID_HEIGHT * GRID_WIDTH,
#     }
#     for idx, state_var_name in enumerate(["wolf", "sheep", "grass"]):
#         axs[idx].plot(
#             vp_data[state_var_name],
#             label="true value",
#             color="black",
#         )
#         axs[idx].plot(
#             range(TIME_SPAN + 1),
#             mean_vec[:, idx],
#             label="estimate",
#         )
#         axs[idx].fill_between(
#             range(TIME_SPAN + 1),
#             np.maximum(
#                 0.0,
#                 mean_vec[:, idx] - np.sqrt(cov_matrix[:, idx, idx]),
#             ),
#             np.minimum(
#                 max_scales[state_var_name],
#                 mean_vec[:, idx] + np.sqrt(cov_matrix[:, idx, idx]),
#             ),
#             color="gray",
#             alpha=0.35,
#         )
#         axs[idx].set_title(state_var_name)
#         axs[idx].legend()
#     axs[3].set_title("surprisal")
#     axs[3].plot(surprisal_state, label="state surprisal")
#     axs[3].plot(
#         [0, TIME_SPAN + 1], [mean_surprisal_state, mean_surprisal_state], ":", color="black"
#     )
#     fig.tight_layout()
#     fig.savefig(FILE_PREFIX + f"match.pdf")
#     plt.close(fig)
#
# np.savez_compressed(
#     FILE_PREFIX + f"data.npz",
#     vp_full_trajectory=vp_full_trajectory,
#     mean_vec=mean_vec,
#     cov_matrix=cov_matrix,
# )
# with open(FILE_PREFIX + "meansurprisal.csv", "w") as file:
#     csvwriter = csv.writer(file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
#     csvwriter.writerow(["full", "state", "param"])
#     csvwriter.writerow([mean_surprisal_full, mean_surprisal_state, mean_surprisal_param])
