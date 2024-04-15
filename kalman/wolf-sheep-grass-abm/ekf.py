#!/usr/bin/env python3
import argparse
import sys
from copy import deepcopy
from typing import Final

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm
from wolf_sheep_grass import WolfSheepGrassModel

from transform import transform_intrinsic_to_kf, transform_kf_to_intrinsic
from util import random_walk_covariance, slogdet

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
            "wolves",
            "wolf",
            "sheep",
            "grass",
            "wolves+grass",
            "wolf+grass",
            "sheep+grass",
            "wolves+sheep",
            "wolf+sheep",
            "wolves+sheep+grass",
            "wolf+sheep+grass",
        ],
        default="grass",
        help="which things to measure, either wolf/wolves produce same result",
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

    parser.add_argument("--verbose", help="print extra messages", action="store_true")

    parser.add_argument("--grid_width", help="width of simulation grid", type=int, default=51)
    parser.add_argument("--grid_height", help="height of simulation grid", type=int, default=51)
    parser.add_argument(
        "--init_wolf_density",
        help="init number of wolves = density*gridsize",
        type=float,
        default=-1,
    )
    parser.add_argument(
        "--init_sheep_density",
        help="init number of sheep = density*gridsize",
        type=float,
        default=-1,
    )
    parser.add_argument(
        "--init_grass_density", help="approx init density of grass", type=float, default=-1
    )

    parser.add_argument("--time_span", help="total simulation time", type=int, default=1000)
    parser.add_argument(
        "--sample_interval", help="interval between measurements", type=int, default=50
    )

    parser.add_argument(
        "--ensemble_size",
        help="number of members of ensemble, defaults to dim*(dim-1)",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--wolf_r",
        help="R matrix term for wolves",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--sheep_r",
        help="R matrix term for sheep",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--grass_r",
        help="R matrix term for grass",
        type=float,
        default=-1.0,
    )

    parser.add_argument(
        "--param_stoch_level",
        help="stochasticity parameter for parameter random walk",
        type=float,
        default=0.01,
    )

    args = parser.parse_args()

################################################################################
# constants

GRID_WIDTH: Final[int] = (
    51 if not hasattr(args, "grid_width") or args.grid_width is None else args.grid_width
)
GRID_HEIGHT: Final[int] = (
    51 if not hasattr(args, "grid_height") or args.grid_height is None else args.grid_height
)
TIME_SPAN: Final[int] = (
    1000 if not hasattr(args, "time_span") or args.time_span is None else args.time_span
)
SAMPLE_INTERVAL: Final[int] = (
    50
    if not hasattr(args, "sample_interval") or args.sample_interval is None
    else args.sample_interval
)
NUM_CYCLES: Final[int] = TIME_SPAN // SAMPLE_INTERVAL
UNIFIED_STATE_SPACE_DIMENSION: Final[int] = 8  # 3 macrostates and 5 parameters
ENSEMBLE_SIZE: Final[int] = (
    2 * (UNIFIED_STATE_SPACE_DIMENSION * (UNIFIED_STATE_SPACE_DIMENSION - 1) // 2)
    if not hasattr(args, "ensemble_size") or args.ensemble_size is None or args.ensemble_size == -1
    else args.ensemble_size
)
OBSERVABLE_temp: Final[str] = (
    "wolves+sheep+grass" if not hasattr(args, "measurements") else args.measurements
)
OBSERVABLE: Final[str] = OBSERVABLE_temp.replace("wolf", "wolves")  # canonicalize the name

WOLF_R: Final[float] = (
    1.0 if not hasattr(args, "wolf_r") or args.wolf_r is None or args.wolf_r == -1 else args.wolf_r
)
SHEEP_R: Final[float] = (
    1.0
    if not hasattr(args, "sheep_r") or args.sheep_r is None or args.sheep_r == -1
    else args.sheep_r
)
GRASS_R: Final[float] = (
    1.0
    if not hasattr(args, "grass_r") or args.grass_r is None or args.grass_r == -1
    else args.grass_r
)

PARAM_STOCH_LEVEL: Final[float] = (
    0.01
    if not hasattr(args, "param_stoch_level")
    or args.param_stoch_level is None
    or args.param_stoch_level == -1
    else args.param_stoch_level
)

# if we are altering the models (as opposed to resampling) try to match the
# models to minimize the changes necessary.
MODEL_MATCHMAKER: Final[bool] = (
    True if not hasattr(args, "matchmaker") else (args.matchmaker == "yes")
)

# have the models' parameters do a random walk over time (should help
# with covariance starvation)
PARAMETER_RANDOM_WALK: Final[bool] = True

FILE_PREFIX: Final[str] = (
    ""
    if not hasattr(args, "prefix") or args.prefix is None or len(args.prefix) == 0
    else args.prefix + "-"
)

GRAPHS: Final[bool] = (
    True if not hasattr(args, "graphs") or args.graphs is None else bool(args.graphs)
)
VERBOSE: Final[bool] = (
    True if not hasattr(args, "verbose") or args.verbose is None else bool(args.verbose)
)


################################################################################
# Macrostate extraction and transformations


def model_macro_data(model: WolfSheepGrassModel):
    """
    Collect macroscale data from a model
    :param model:
    :return:
    """
    macroscale_data = np.zeros(UNIFIED_STATE_SPACE_DIMENSION, dtype=np.float64)
    macroscale_data[0] = model.num_wolves
    macroscale_data[1] = model.num_sheep
    macroscale_data[2] = np.sum(model.grass)
    macroscale_data[3] = model.WOLF_GAIN_FROM_FOOD
    macroscale_data[4] = model.SHEEP_GAIN_FROM_FOOD
    macroscale_data[5] = model.WOLF_REPRODUCE
    macroscale_data[6] = model.SHEEP_REPRODUCE
    macroscale_data[7] = model.GRASS_REGROWTH_TIME
    return macroscale_data


################################################################################
# initial statistical parameters
# note that these are slightly different from the coordinates for the KF
# e.g. parameters are untransformed

# TODO: command line options of stdevs
mean_init_wolves = (
    50 * GRID_HEIGHT * GRID_WIDTH / (51**2)
    if not hasattr(args, "init_wolf_density") or args.init_wolf_density == -1
    else args.init_wolf_density * GRID_WIDTH * GRID_HEIGHT
)  # state variable (int valued)
std_init_wolves = 5 * GRID_HEIGHT * GRID_WIDTH / (51**2)  # expected to be <= sqrt(50) ~= 7
mean_init_sheep = (
    100 * GRID_HEIGHT * GRID_WIDTH / (51**2)
    if not hasattr(args, "init_sheep_density") or args.init_sheep_density == -1
    else args.init_sheep_density * GRID_WIDTH * GRID_HEIGHT
)  # state variable (int valued)
std_init_sheep = 5 * GRID_HEIGHT * GRID_WIDTH / (51**2)  # expected to be <= sqrt(100) = 10
mean_init_grass_proportion = (
    0.5
    if not hasattr(args, "init_grass_density") or args.init_grass_density == -1
    else args.init_grass_density
)  # state variable
std_init_grass_proportion = 0.02  # expected to be <= sqrt(0.5*51^2)/(51^2) ~= 0.04
mean_wolf_gain_from_food = 20.0  # parameter
std_wolf_gain_from_food = 2.0  # arbitrary
mean_sheep_gain_from_food = 4.0  # parameter
std_sheep_gain_from_food = 2.0  # arbitrary
mean_wolf_reproduce = 5.0  # parameter
std_wolf_reproduce = 2.0  # arbitrary
mean_sheep_reproduce = 4.0  # parameter
std_sheep_reproduce = 2.0  # arbitrary
mean_grass_regrowth_time = 30.0  # parameter
std_grass_regrowth_time = 2.0  # arbitrary

mean_init_vec = np.array(
    [
        mean_init_wolves,
        mean_init_sheep,
        mean_init_grass_proportion * GRID_HEIGHT * GRID_WIDTH,
        mean_wolf_gain_from_food,
        mean_sheep_gain_from_food,
        mean_wolf_reproduce,
        mean_sheep_reproduce,
        mean_grass_regrowth_time,
    ]
)

cov_matrix_init = np.diag(
    np.array(
        [
            std_init_wolves,
            std_init_sheep,
            std_init_grass_proportion * GRID_WIDTH * GRID_HEIGHT,
            std_wolf_gain_from_food,
            std_sheep_gain_from_food,
            std_wolf_reproduce,
            std_sheep_reproduce,
            std_grass_regrowth_time,
        ]
    )
    ** 2
)

################################################################################
# sample a virtual patient

# sampled virtual patient parameters
(
    vp_init_wolves,
    vp_init_sheep,
    vp_init_grass,
    vp_wolf_gain_from_food,
    vp_sheep_gain_from_food,
    vp_wolf_reproduce,
    vp_sheep_reproduce,
    vp_grass_regrowth_time,
) = np.abs(multivariate_normal(mean=mean_init_vec, cov=cov_matrix_init).rvs())

# create model for virtual patient
virtual_patient_model = WolfSheepGrassModel(
    GRID_WIDTH=GRID_WIDTH,
    GRID_HEIGHT=GRID_HEIGHT,
    INIT_WOLVES=int(vp_init_wolves),
    INIT_SHEEP=int(vp_init_sheep),
    INIT_GRASS_PROPORTION=vp_init_grass / (GRID_WIDTH * GRID_HEIGHT),
    WOLF_GAIN_FROM_FOOD=vp_wolf_gain_from_food,
    SHEEP_GAIN_FROM_FOOD=vp_sheep_gain_from_food,
    WOLF_REPRODUCE=vp_wolf_reproduce,
    SHEEP_REPRODUCE=vp_sheep_reproduce,
    GRASS_REGROWTH_TIME=vp_grass_regrowth_time,
    SOFTCAP_AGENT_NUM_AT_GRID_SIZE=True,
)

# evaluate the virtual patient's trajectory
vp_wolf_counts = np.zeros(TIME_SPAN + 1, dtype=int)
vp_sheep_counts = np.zeros(TIME_SPAN + 1, dtype=int)
vp_grass_counts = np.zeros(TIME_SPAN + 1, dtype=int)

vp_wolf_counts[0] = virtual_patient_model.num_wolves
vp_sheep_counts[0] = virtual_patient_model.num_sheep
vp_grass_counts[0] = np.sum(virtual_patient_model.grass)
for t in range(1, TIME_SPAN + 1):
    virtual_patient_model.time_step()
    vp_wolf_counts[t] = virtual_patient_model.num_wolves
    vp_sheep_counts[t] = virtual_patient_model.num_sheep
    vp_grass_counts[t] = np.sum(virtual_patient_model.grass)

################################################################################
# plot virtual patient

if GRAPHS:
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(vp_wolf_counts, label="wolves")
    ax.plot(vp_sheep_counts, label="sheep")
    ax.plot(vp_grass_counts, label="grass")
    ax.legend()
    fig.savefig(FILE_PREFIX + "virtual-patient.pdf")
    plt.close(fig)


################################################################################


def model_ensemble_from(means, covariances, *, transformed_sample: bool = False):
    """
    Create an ensemble of models from a distribution

    :param means:
    :param covariances:
    :param transformed_sample: use True if the means and covariances are in KF coordinates
    :return:
    """
    mdl_ensemble = []
    distribution = multivariate_normal(mean=means, cov=covariances, allow_singular=True)
    for _ in range(ENSEMBLE_SIZE):
        rand_sample = distribution.rvs()

        if transformed_sample:
            (
                en_init_wolves,
                en_init_sheep,
                en_init_grass,
                en_wolf_gain_from_food,
                en_sheep_gain_from_food,
                en_wolf_reproduce,
                en_sheep_reproduce,
                en_grass_regrowth_time,
            ) = transform_kf_to_intrinsic(rand_sample)
        else:
            (
                en_init_wolves,
                en_init_sheep,
                en_init_grass,
                en_wolf_gain_from_food,
                en_sheep_gain_from_food,
                en_wolf_reproduce,
                en_sheep_reproduce,
                en_grass_regrowth_time,
            ) = np.abs(rand_sample)
        en_model = WolfSheepGrassModel(
            GRID_WIDTH=GRID_WIDTH,
            GRID_HEIGHT=GRID_HEIGHT,
            INIT_WOLVES=int(en_init_wolves),
            INIT_SHEEP=int(en_init_sheep),
            INIT_GRASS_PROPORTION=en_init_grass / (GRID_WIDTH * GRID_HEIGHT),
            WOLF_GAIN_FROM_FOOD=en_wolf_gain_from_food,
            SHEEP_GAIN_FROM_FOOD=en_sheep_gain_from_food,
            WOLF_REPRODUCE=en_wolf_reproduce,
            SHEEP_REPRODUCE=en_sheep_reproduce,
            GRASS_REGROWTH_TIME=en_grass_regrowth_time,
        )
        mdl_ensemble.append(en_model)
    return mdl_ensemble


################################################################################


def modify_model(
    model: WolfSheepGrassModel,
    desired_state: np.ndarray,
    *,
    ignore_state_vars: bool = False,
    fix_grass_clocks: bool = True,
):
    """
    Modify a model's microstate to fit a given macrostate

    :param model: model instance (encodes microstate)
    :param desired_state: desired macrostate for the model
    :param ignore_state_vars: if True, only alter parameters, not state variables
    :param fix_grass_clocks: if True, make grass regrowth clocks consistent with new regrowth time
    :return: None
    """
    (
        num_wolves,
        num_sheep,
        num_grass,
        wolf_gain_from_food,
        sheep_gain_from_food,
        wolf_reproduce,
        sheep_reproduce,
        grass_regrowth_time,
    ) = np.maximum(0, desired_state)
    model.WOLF_GAIN_FROM_FOOD = wolf_gain_from_food
    model.SHEEP_GAIN_FROM_FOOD = sheep_gain_from_food
    model.WOLF_REPRODUCE = wolf_reproduce
    model.SHEEP_REPRODUCE = sheep_reproduce
    model.GRASS_REGROWTH_TIME = grass_regrowth_time

    if fix_grass_clocks:
        np.minimum(model.grass_clock, model.GRASS_REGROWTH_TIME, out=model.grass_clock)

    if ignore_state_vars:
        return

    # Fix the number of wolves/sheep/grass by random spawning/killing.

    num_wolves = int(num_wolves)
    if num_wolves > model.num_wolves:
        if VERBOSE:
            print(f"creating {num_wolves - model.num_wolves} new wolves")
        for _ in range(num_wolves - model.num_wolves):
            model.create_wolf()
    elif num_wolves < model.num_wolves:
        if VERBOSE:
            print(f"killing {model.num_wolves - num_wolves} wolves")
        try:
            for _ in range(model.num_wolves - num_wolves):
                model.kill_random_wolf()
        except RuntimeError as e:
            print(e)

    num_sheep = int(num_sheep)
    if num_sheep > model.num_sheep:
        if VERBOSE:
            print(f"creating {num_sheep - model.num_sheep} new sheep")
        for _ in range(num_sheep - model.num_sheep):
            model.create_sheep()
    elif num_sheep < model.num_sheep:
        if VERBOSE:
            print(f"killing {model.num_sheep - num_sheep} sheep")
        try:
            for _ in range(model.num_sheep - num_sheep):
                model.kill_random_sheep()
        except RuntimeError as e:
            if VERBOSE:
                print(e)

    num_grass = int(num_grass)
    grass_present = np.sum(model.grass)
    if num_grass > grass_present:
        if VERBOSE:
            print(f"creating {num_grass - grass_present} new grass")
        try:
            for _ in range(num_grass - grass_present):
                model.spawn_grass()
        except RuntimeError as e:
            if VERBOSE:
                print(e)
    elif num_grass < grass_present:
        if VERBOSE:
            print(f"killing {grass_present - num_grass} grass")
        try:
            for _ in range(grass_present - num_grass):
                model.kill_random_grass()
        except RuntimeError as e:
            if VERBOSE:
                print(e)


################################################################################
# Kalman filter simulation
################################################################################

# create initial ensemble of models for kalman filter
model_ensemble = model_ensemble_from(mean_init_vec, cov_matrix_init)

# mean and covariances through time
mean_vec = np.zeros(
    (NUM_CYCLES + 1, TIME_SPAN + 1, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64
)
cov_matrix = np.zeros(
    (NUM_CYCLES + 1, TIME_SPAN + 1, UNIFIED_STATE_SPACE_DIMENSION, UNIFIED_STATE_SPACE_DIMENSION),
    dtype=np.float64,
)

kalman_v = []
kalman_S = []
kalman_K = []

# collect initial statistics
time = 0
initial_macro_data = np.array(
    [transform_intrinsic_to_kf(model_macro_data(model)) for model in model_ensemble]
)
mean_vec[:, time, :] = np.mean(initial_macro_data, axis=0)
cov_matrix[:, time, :, :] = np.cov(initial_macro_data, rowvar=False)

for cycle in tqdm(range(NUM_CYCLES), desc="cycle"):
    # advance ensemble of models
    for time in range(cycle * SAMPLE_INTERVAL + 1, (cycle + 1) * SAMPLE_INTERVAL + 1):
        for model in model_ensemble:
            model.time_step()
            if PARAMETER_RANDOM_WALK:
                macrostate = transform_intrinsic_to_kf(model_macro_data(model))
                random_walk_macrostate = (
                    macrostate
                    + multivariate_normal(
                        mean=np.zeros_like(macrostate),
                        cov=random_walk_covariance(macrostate, param_stoch_level=PARAM_STOCH_LEVEL),
                        allow_singular=True,
                    ).rvs()
                )
                modify_model(
                    model, transform_kf_to_intrinsic(random_walk_macrostate), ignore_state_vars=True
                )
        macro_data = np.array(
            [transform_intrinsic_to_kf(model_macro_data(model)) for model in model_ensemble]
        )
        mean_vec[cycle:, time, :] = np.mean(macro_data, axis=0)
        cov_matrix[cycle:, time, :, :] = np.cov(macro_data, rowvar=False)

    # make copy of the models and advance them to the end of the simulation time
    model_ensemble_copy = deepcopy(model_ensemble)
    for future_time in range((cycle + 1) * SAMPLE_INTERVAL + 1, TIME_SPAN + 1):
        for model in model_ensemble_copy:
            model.time_step()
            if PARAMETER_RANDOM_WALK:
                macrostate = transform_intrinsic_to_kf(model_macro_data(model))
                random_walk_macrostate = (
                    macrostate
                    + multivariate_normal(
                        mean=np.zeros_like(macrostate),
                        cov=random_walk_covariance(macrostate, param_stoch_level=PARAM_STOCH_LEVEL),
                        allow_singular=True,
                    ).rvs()
                )
                modify_model(
                    model, transform_kf_to_intrinsic(random_walk_macrostate), ignore_state_vars=True
                )
        macro_data = np.array(
            [transform_intrinsic_to_kf(model_macro_data(model)) for model in model_ensemble_copy]
        )
        mean_vec[cycle:, future_time, :] = np.mean(macro_data, axis=0)
        cov_matrix[cycle:, future_time, :, :] = np.cov(macro_data, rowvar=False)

    ################################################################################
    # plot projection of state variables

    if GRAPHS:
        fig, axs = plt.subplots(3, figsize=(6, 6), sharex=True, sharey=False, layout="constrained")
        plural = {"wolf": "wolves", "sheep": "sheep", "grass": "grass"}
        vp_data = {
            "wolf": vp_wolf_counts,
            "sheep": vp_sheep_counts,
            "grass": vp_grass_counts,
        }
        max_scales = {
            "wolf": mean_init_wolves,
            "sheep": mean_init_sheep,
            "grass": mean_init_grass_proportion * GRID_HEIGHT * GRID_WIDTH,
        }
        for idx, state_var_name in enumerate(["wolf", "sheep", "grass"]):
            (true_value,) = axs[idx].plot(
                range(TIME_SPAN + 1),
                vp_data[state_var_name],
                label="true value",
                linestyle=":",
                color="gray",
            )
            (past_estimate_center_line,) = axs[idx].plot(
                range((cycle + 1) * SAMPLE_INTERVAL),
                transform_kf_to_intrinsic(
                    mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx], index=idx
                ),
                label="estimate of past",
                color="black",
            )
            past_estimate_range = axs[idx].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL),
                transform_kf_to_intrinsic(
                    mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx]
                    - np.sqrt(cov_matrix[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx, idx]),
                    index=idx,
                ),
                np.minimum(
                    10 * max_scales[state_var_name],
                    transform_kf_to_intrinsic(
                        mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx]
                        + np.sqrt(cov_matrix[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx, idx]),
                        index=idx,
                    ),
                ),
                color="gray",
                alpha=0.35,
            )

            mu = mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx]
            sigma = np.sqrt(cov_matrix[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx])

            (prediction_center_line,) = axs[idx].plot(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                transform_kf_to_intrinsic(mu, index=idx),
                label="prediction",
                color="blue",
            )
            prediction_range = axs[idx].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                transform_kf_to_intrinsic(mu - sigma, index=idx),
                np.minimum(
                    10 * max_scales[state_var_name],
                    transform_kf_to_intrinsic(
                        mu + sigma,
                        index=idx,
                    ),
                ),
                color="blue",
                alpha=0.35,
            )
            axs[idx].set_title(state_var_name, loc="left")
        # noinspection PyUnboundLocalVariable
        fig.legend(
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
            # loc="outside upper right",
        )
        fig.suptitle("State Prediction")
        fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-state.pdf")
        plt.close(fig)

    ################################################################################
    # plot projection of parameters

    if GRAPHS:
        params = [
            "wolf gain from food",
            "sheep gain from food",
            "wolf reproduce",
            "sheep reproduce",
            "grass regrowth time",
        ]
        vp_param_values = dict(
            zip(
                params,
                [
                    vp_wolf_gain_from_food,
                    vp_sheep_gain_from_food,
                    vp_wolf_reproduce,
                    vp_sheep_reproduce,
                    vp_grass_regrowth_time,
                ],
            )
        )

        fig, axs = plt.subplots(
            3, 2, figsize=(8, 8), sharex=True, sharey=False, layout="constrained"
        )
        for idx, param_name in enumerate(params):
            row, col = divmod(idx, 3)
            (true_value,) = axs[row, col].plot(
                [0, TIME_SPAN + 1],
                [vp_param_values[param_name]] * 2,
                label="true value",
                color="gray",
                linestyle=":",
            )

            (past_estimate_center_line,) = axs[row, col].plot(
                range((cycle + 1) * SAMPLE_INTERVAL),
                transform_kf_to_intrinsic(
                    mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, 3 + idx], index=3 + idx
                ),
                color="black",
                label="estimate of past",
            )
            past_estimate_range = axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL),
                transform_kf_to_intrinsic(
                    mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, 3 + idx]
                    - np.sqrt(cov_matrix[cycle, : (cycle + 1) * SAMPLE_INTERVAL, 3 + idx, 3 + idx]),
                    index=3 + idx,
                ),
                np.minimum(
                    np.maximum(
                        10 * vp_param_values[param_name],
                        np.max(
                            1.1
                            * transform_kf_to_intrinsic(mean_vec[cycle, :, 3 + idx], index=3 + idx)
                        ),
                    ),
                    transform_kf_to_intrinsic(
                        mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, 3 + idx]
                        + np.sqrt(
                            cov_matrix[cycle, : (cycle + 1) * SAMPLE_INTERVAL, 3 + idx, 3 + idx]
                        ),
                        index=3 + idx,
                    ),
                ),
                color="gray",
                alpha=0.35,
                label="past cone of uncertainty",
            )

            (prediction_center_line,) = axs[row, col].plot(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                transform_kf_to_intrinsic(
                    mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, 3 + idx], index=3 + idx
                ),
                label="predictive estimate",
                color="blue",
            )
            prediction_range = axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                transform_kf_to_intrinsic(
                    mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, 3 + idx]
                    - np.sqrt(cov_matrix[cycle, (cycle + 1) * SAMPLE_INTERVAL :, 3 + idx, 3 + idx]),
                    index=3 + idx,
                ),
                np.minimum(
                    10 * vp_param_values[param_name],
                    transform_kf_to_intrinsic(
                        mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, 3 + idx]
                        + np.sqrt(
                            cov_matrix[cycle, (cycle + 1) * SAMPLE_INTERVAL :, 3 + idx, 3 + idx]
                        ),
                        index=3 + idx,
                    ),
                ),
                color="blue",
                alpha=0.35,
            )
            axs[row, col].set_title(param_name)
        axs[2, 1].axis("off")
        # noinspection PyUnboundLocalVariable
        fig.legend(
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
            loc="outside lower right",
        )
        fig.suptitle("Parameter Projection")
        fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-params.pdf")
        plt.close(fig)

    ################################################################################
    # Kalman filter

    match OBSERVABLE:
        case "wolves":
            R = WOLF_R
            H = np.zeros((1, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64)
            H[0, 0] = 1.0  # observe wolves
            observation = transform_intrinsic_to_kf(vp_wolf_counts[time], index=0)
        case "sheep":
            R = SHEEP_R
            H = np.zeros((1, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64)
            H[0, 1] = 1.0  # observe sheep
            observation = transform_intrinsic_to_kf(vp_sheep_counts[time], index=1)
        case "grass":
            R = GRASS_R
            H = np.zeros((1, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64)
            H[0, 2] = 1.0  # observe grass
            observation = transform_intrinsic_to_kf(vp_grass_counts[time], index=2)
        case "wolves+grass":
            R = np.diag([WOLF_R, GRASS_R])
            H = np.zeros((2, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64)
            H[0, 0] = 1.0  # observe wolves
            H[1, 2] = 1.0  # observe grass
            observation = np.array(
                [
                    transform_intrinsic_to_kf(vp_wolf_counts[time], index=0),
                    transform_intrinsic_to_kf(vp_grass_counts[time], index=2),
                ],
                dtype=np.float64,
            )
        case "sheep+grass":
            R = np.diag([SHEEP_R, GRASS_R])
            H = np.zeros((2, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64)
            H[0, 1] = 1.0  # observe sheep
            H[1, 2] = 1.0  # observe grass
            observation = np.array(
                [
                    transform_intrinsic_to_kf(vp_sheep_counts[time], index=1),
                    transform_intrinsic_to_kf(vp_grass_counts[time], index=2),
                ],
                dtype=np.float64,
            )
        case "wolves+sheep":
            R = np.diag([WOLF_R, SHEEP_R])
            H = np.zeros((2, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64)
            H[0, 0] = 1.0  # observe wolves
            H[1, 1] = 1.0  # observe sheep
            observation = np.array(
                [
                    transform_intrinsic_to_kf(vp_wolf_counts[time], index=0),
                    transform_intrinsic_to_kf(vp_sheep_counts[time], index=1),
                ],
                dtype=np.float64,
            )
        case "wolves+sheep+grass":
            R = np.diag([WOLF_R, SHEEP_R, GRASS_R])
            H = np.zeros((3, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64)
            H[0, 0] = 1.0  # observe wolves
            H[1, 1] = 1.0  # observe sheep
            H[2, 2] = 1.0  # observe sheep
            observation = np.array(
                [
                    transform_intrinsic_to_kf(vp_wolf_counts[time], index=0),
                    transform_intrinsic_to_kf(vp_sheep_counts[time], index=1),
                    transform_intrinsic_to_kf(vp_grass_counts[time], index=2),
                ],
                dtype=np.float64,
            )
        case _:
            raise RuntimeError("unknown observable?")

    v = observation - (H @ mean_vec[cycle, time, :])
    S = H @ cov_matrix[cycle, time, :, :] @ H.T + R
    K = cov_matrix[cycle, time, :, :] @ H.T @ np.linalg.pinv(S)

    kalman_v.append(v)
    kalman_S.append(S)
    kalman_K.append(K)

    mean_vec[cycle + 1, time, :] += K @ v
    # cov_matrix[cycle + 1, time, :, :] -= K @ S @ K.T
    # Joseph form update (See e.g. https://www.anuncommonlab.com/articles/how-kalman-filters-work/part2.html)
    A = np.eye(cov_matrix.shape[-1]) - K @ H
    cov_matrix[cycle + 1, time, :, :] = np.nan_to_num(
        A @ cov_matrix[cycle + 1, time, :, :] @ A.T + K @ R @ K.T
    )
    min_diag = np.min(np.diag(cov_matrix[cycle + 1, time, :, :]))
    if min_diag <= 0.0:
        cov_matrix[cycle + 1, time, :, :] += (1e-6 - min_diag) * np.eye(cov_matrix.shape[-1])

    # recreate ensemble
    dist = multivariate_normal(
        mean=mean_vec[cycle + 1, time, :],
        cov=cov_matrix[cycle + 1, time, :, :],
        allow_singular=True,
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
                    competitor_model_idx = sample_to_model_pairing[possible_sample_pair]
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
                            macro_data[model_idx, :] - new_sample[possible_sample_pair, :]
                        )
                        if proposed_pair_dist < established_pair_dist:
                            model_to_sample_pairing[competitor_model_idx] = (
                                -1
                            )  # free the competitor
                            all_paired = False
                            # make new pair
                            sample_to_model_pairing[possible_sample_pair] = model_idx
                            model_to_sample_pairing[model_idx] = possible_sample_pair
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
                transform_kf_to_intrinsic(new_sample[model_to_sample_pairing[model_idx], :]),
            )
    else:
        # sample from KF-learned dist and modify existing models to fit
        for model in model_ensemble:
            state = dist.rvs()
            modify_model(model, transform_kf_to_intrinsic(state))

################################################################################

################################################################################
# plot kalman update of state variables

if GRAPHS:
    plural = {"wolf": "wolves", "sheep": "sheep", "grass": "grass"}
    vp_data = {
        "wolf": vp_wolf_counts,
        "sheep": vp_sheep_counts,
        "grass": vp_grass_counts,
    }
    max_scales = {
        "wolf": mean_init_wolves,
        "sheep": mean_init_sheep,
        "grass": mean_init_grass_proportion * GRID_HEIGHT * GRID_WIDTH,
    }
    for cycle in range(NUM_CYCLES - 1):
        fig, axs = plt.subplots(3, figsize=(6, 6), sharex=True, layout="constrained")
        for idx, state_var_name in enumerate(["wolf", "sheep", "grass"]):
            (true_value,) = axs[idx].plot(
                range(TIME_SPAN + 1),
                vp_data[state_var_name],
                label="true value",
                color="black",
            )

            (past_est_center_line,) = axs[idx].plot(
                range((cycle + 1) * SAMPLE_INTERVAL),
                transform_kf_to_intrinsic(
                    mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx], index=idx
                ),
                color="green",
                label="estimate of past",
            )
            past_est_range = axs[idx].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL),
                np.maximum(
                    0.0,
                    transform_kf_to_intrinsic(
                        mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx]
                        - np.sqrt(cov_matrix[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx, idx]),
                        index=idx,
                    ),
                ),
                np.minimum(
                    10 * max_scales[state_var_name],
                    transform_kf_to_intrinsic(
                        mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx]
                        + np.sqrt(cov_matrix[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx, idx]),
                        index=idx,
                    ),
                ),
                color="green",
                alpha=0.35,
            )

            (future_est_before_update_center_line,) = axs[idx].plot(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                transform_kf_to_intrinsic(
                    mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx], index=idx
                ),
                color="#d5b60a",
                label="previous future estimate",
            )
            future_est_before_update_range = axs[idx].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                np.maximum(
                    0.0,
                    transform_kf_to_intrinsic(
                        mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx]
                        - np.sqrt(cov_matrix[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]),
                        index=idx,
                    ),
                ),
                np.minimum(
                    10 * max_scales[state_var_name],
                    transform_kf_to_intrinsic(
                        mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx]
                        + np.sqrt(cov_matrix[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]),
                        index=idx,
                    ),
                ),
                color="#d5b60a",
                alpha=0.35,
            )

            (future_est_after_update_center_line,) = axs[idx].plot(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                transform_kf_to_intrinsic(
                    mean_vec[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, idx], index=idx
                ),
                label="updated future estimate",
                color="blue",
            )
            future_est_after_update_range = axs[idx].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                np.maximum(
                    0.0,
                    transform_kf_to_intrinsic(
                        mean_vec[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, idx]
                        - np.sqrt(cov_matrix[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]),
                        index=idx,
                    ),
                ),
                np.minimum(
                    10 * max_scales[state_var_name],
                    transform_kf_to_intrinsic(
                        mean_vec[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, idx]
                        + np.sqrt(cov_matrix[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]),
                        index=idx,
                    ),
                ),
                color="blue",
                alpha=0.35,
            )
            axs[idx].set_title(state_var_name, loc="left")
        # noinspection PyUnboundLocalVariable
        fig.legend(
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
            loc="outside lower center",
        )
        fig.suptitle("State Projection", ha="left")
        fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-state-kfupd.pdf")
        plt.close(fig)

################################################################################
# plot kalman updates of parameters

if GRAPHS:
    params = [
        "wolf gain from food",
        "sheep gain from food",
        "wolf reproduce",
        "sheep reproduce",
        "grass regrowth time",
    ]
    vp_param_values = dict(
        zip(
            params,
            [
                vp_wolf_gain_from_food,
                vp_sheep_gain_from_food,
                vp_wolf_reproduce,
                vp_sheep_reproduce,
                vp_grass_regrowth_time,
            ],
        )
    )

    for cycle in range(NUM_CYCLES - 1):
        fig, axs = plt.subplots(3, 2, figsize=(8, 8), sharex=True, layout="constrained")
        for idx, param_name in enumerate(params):
            row, col = divmod(idx, 3)

            (true_value,) = axs[row, col].plot(
                [0, TIME_SPAN + 1],
                [vp_param_values[param_name]] * 2,
                label="true value",
                color="black",
            )

            (past_est_center_line,) = axs[row, col].plot(
                range((cycle + 1) * SAMPLE_INTERVAL),
                transform_kf_to_intrinsic(
                    mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, 3 + idx], index=3 + idx
                ),
                label="estimate of past",
                color="green",
            )
            past_est_range = axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL),
                np.maximum(
                    0.0,
                    transform_kf_to_intrinsic(
                        mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, 3 + idx]
                        - np.sqrt(
                            cov_matrix[cycle, : (cycle + 1) * SAMPLE_INTERVAL, 3 + idx, 3 + idx]
                        ),
                        index=3 + idx,
                    ),
                ),
                np.minimum(
                    10 * vp_param_values[param_name],
                    transform_kf_to_intrinsic(
                        mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, 3 + idx]
                        + np.sqrt(
                            cov_matrix[cycle, : (cycle + 1) * SAMPLE_INTERVAL, 3 + idx, 3 + idx]
                        ),
                        index=3 + idx,
                    ),
                ),
                color="green",
                alpha=0.35,
            )

            (future_est_before_update_center_line,) = axs[row, col].plot(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                transform_kf_to_intrinsic(
                    mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, 3 + idx], index=3 + idx
                ),
                label="old estimate",
                color="#d5b60a",
            )
            future_est_before_update_range = axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                np.maximum(
                    0.0,
                    transform_kf_to_intrinsic(
                        mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, 3 + idx]
                        - np.sqrt(
                            cov_matrix[cycle, (cycle + 1) * SAMPLE_INTERVAL :, 3 + idx, 3 + idx]
                        ),
                        index=3 + idx,
                    ),
                ),
                np.minimum(
                    10 * vp_param_values[param_name],
                    transform_kf_to_intrinsic(
                        mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, 3 + idx]
                        + np.sqrt(
                            cov_matrix[cycle, (cycle + 1) * SAMPLE_INTERVAL :, 3 + idx, 3 + idx]
                        ),
                        index=3 + idx,
                    ),
                ),
                color="#d5b60a",
                alpha=0.35,
            )

            (future_est_after_update_center_line,) = axs[row, col].plot(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                transform_kf_to_intrinsic(
                    mean_vec[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, 3 + idx], index=3 + idx
                ),
                label="updated estimate",
                color="blue",
            )
            future_est_after_update_range = axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                np.maximum(
                    0.0,
                    transform_kf_to_intrinsic(
                        mean_vec[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, 3 + idx]
                        - np.sqrt(
                            cov_matrix[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, 3 + idx, 3 + idx]
                        ),
                        index=3 + idx,
                    ),
                ),
                np.minimum(
                    10 * vp_param_values[param_name],
                    transform_kf_to_intrinsic(
                        mean_vec[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, 3 + idx]
                        + np.sqrt(
                            cov_matrix[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, 3 + idx, 3 + idx]
                        ),
                        index=3 + idx,
                    ),
                ),
                color="blue",
                alpha=0.35,
                label="new future cone of uncertainty",
            )
            axs[row, col].set_title(param_name)
            # axs[row, col].legend()
        axs[2, 1].axis("off")
        # noinspection PyUnboundLocalVariable
        fig.legend(
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
            loc="lower right",
        )
        fig.suptitle("Parameter Projection", x=0, ha="left")
        fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-params-kfupd.pdf")
        plt.close(fig)

################################################################################
# calculate surprisal information

#####
# full surprisal: all state vars and params
vp_full_trajectory = transform_intrinsic_to_kf(
    np.array(
        (
            vp_wolf_counts,
            vp_sheep_counts,
            vp_grass_counts,
            [vp_wolf_gain_from_food] * (TIME_SPAN + 1),
            [vp_sheep_gain_from_food] * (TIME_SPAN + 1),
            [vp_wolf_reproduce] * (TIME_SPAN + 1),
            [vp_sheep_reproduce] * (TIME_SPAN + 1),
            [vp_grass_regrowth_time] * (TIME_SPAN + 1),
        )
    ).T
)

delta_full = mean_vec - vp_full_trajectory
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
surprisal_full_quadratic_part = np.einsum("cij,cij->ci", delta_full, sigma_inv_delta)
surprisal_full = (
    surprisal_full_quadratic_part + logdet + UNIFIED_STATE_SPACE_DIMENSION * np.log(2 * np.pi)
) / 2.0

#####
# state surprisal: restrict to just the state vars
vp_state_trajectory = vp_full_trajectory[:, :3]
delta_state = mean_vec[:, :, :3] - vp_state_trajectory
_, logdet = slogdet(cov_matrix[:, :, :3, :3])
sigma_inv_delta = np.array(
    [
        [
            np.linalg.lstsq(
                cov_matrix[cycle, t_idx, :3, :3], delta_state[cycle, t_idx, :], rcond=None
            )[0]
            for t_idx in range(cov_matrix.shape[1])
        ]
        for cycle in range(NUM_CYCLES + 1)
    ]
)

surprisal_state_quadratic_part = np.einsum("cij,cij->ci", delta_state, sigma_inv_delta)
surprisal_state = (
    surprisal_state_quadratic_part + logdet + 3 * np.log(2 * np.pi)
) / 2.0  # 3 -> 3 state vars

#####
# param surprisal: restrict to just the params
vp_param_trajectory = vp_full_trajectory[:, 3:]
delta_param = mean_vec[:, :, 3:] - vp_param_trajectory
_, logdet = slogdet(cov_matrix[:, :, 3:, 3:])
sigma_inv_delta = np.array(
    [
        [
            np.linalg.lstsq(
                cov_matrix[cycle, t_idx, 3:, 3:], delta_param[cycle, t_idx, :], rcond=None
            )[0]
            for t_idx in range(cov_matrix.shape[1])
        ]
        for cycle in range(NUM_CYCLES + 1)
    ]
)

surprisal_param_quadratic_part = np.einsum("cij,cij->ci", delta_param, sigma_inv_delta)
surprisal_param = (
    surprisal_param_quadratic_part + logdet + 5 * np.log(2 * np.pi)
) / 2.0  # 5 -> 5 params

################################################################################

# plt.plot( [ np.mean(surprisal_full[idx,50*idx:50*(idx+1)]) for idx in range(20)])
# plt.plot( [ np.mean(surprisal_state[idx,50*idx:50*(idx+1)]) for idx in range(20)])
# plt.plot( [ np.mean(surprisal_param[idx,50*idx:50*(idx+1)]) for idx in range(20)])

################################################################################
# see the dimension label information here:
# https://docs.h5py.org/en/latest/high/dims.html

with h5py.File(FILE_PREFIX + "data.hdf5", "w") as f:
    f["virtual_patient_trajectory"] = vp_full_trajectory
    f["virtual_patient_trajectory"].dims[0].label = "time"
    f["virtual_patient_trajectory"].dims[1].label = (
        "wolves,"
        "sheep,"
        "grass_proportion,"
        "wolf_gain_from_food,"
        "sheep_gain_from_food,"
        "mean_wolf_reproduce,"
        "mean_sheep_reproduce,"
        "mean_grass_regrowth_time"
    )

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

    f["surprisal_full_quad"] = surprisal_full_quadratic_part
    f["surprisal_full_quad"].dims[0].label = "kalman update number"
    f["surprisal_full_quad"].dims[1].label = "time"

    f["surprisal_state"] = surprisal_state
    f["surprisal_state"].dims[0].label = "kalman update number"
    f["surprisal_state"].dims[1].label = "time"

    f["surprisal_state_quad"] = surprisal_state_quadratic_part
    f["surprisal_state_quad"].dims[0].label = "kalman update number"
    f["surprisal_state_quad"].dims[1].label = "time"

    f["surprisal_param"] = surprisal_param
    f["surprisal_param"].dims[0].label = "kalman update number"
    f["surprisal_param"].dims[1].label = "time"

    f["surprisal_param_quad"] = surprisal_param_quadratic_part
    f["surprisal_param_quad"].dims[0].label = "kalman update number"
    f["surprisal_param_quad"].dims[1].label = "time"

    f["kalman_v"] = np.array(kalman_v)
    f["kalman_S"] = np.array(kalman_S)
    f["kalman_K"] = np.array(kalman_K)
