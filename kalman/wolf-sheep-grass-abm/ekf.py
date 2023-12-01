import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from wolf_sheep_grass import WolfSheepGrassModel

################################################################################
parser = argparse.ArgumentParser()

parser.add_argument("--prefix", type=str, default="", help="output file prefix")

parser.add_argument(
    "--measurements",
    type=str,
    choices=[
        "wolves",
        "sheep",
        "grass",
        "wolves+grass",
        "sheep+grass",
        "wolves+sheep",
        "wolves+sheep+grass",
    ],
    default="grass",
    help="which things to measure",
)

args = parser.parse_args()

################################################################################
# constants

GRID_WIDTH = 51
GRID_HEIGHT = 51
TIME_SPAN = 1000
SAMPLE_INTERVAL = 50  # how often to make measurements
ENSEMBLE_SIZE = 50
UNIFIED_STATE_SPACE_DIMENSION = 8  # 3 macrostates and 5 parameters
OBSERVABLE = (
    "grass" if not args.measurements else args.measurements
)  # "wolves+grass"  # wolves, sheep, grass, wolves+grass
RESAMPLE_MODELS = False
PARAMETER_RANDOM_WALK = True

FILE_PREFIX = "" if not args.prefix else args.prefix + "-"

################################################################################
# statistical parameters

mean_init_wolves = 50  # state variable (int valued)
std_init_wolves = 5  # expected to be <= sqrt(50) ~= 7
mean_init_sheep = 100  # state variable (int valued)
std_init_sheep = 5  # expected to be <= sqrt(100) = 10
mean_init_grass_proportion = 0.5  # state variable
std_init_grass_proportion = 0.02  # expected to be <= sqrt(0.5*51^2)/(51^2) ~= 0.04
mean_wolf_gain_from_food = 20.0  # parameter
std_wolf_gain_from_food = 1.0  # arbitrary
mean_sheep_gain_from_food = 4.0  # parameter
std_sheep_gain_from_food = 1.0  # arbitrary
mean_wolf_reproduce = 5.0  # parameter
std_wolf_reproduce = 1.0  # arbitrary
mean_sheep_reproduce = 4.0  # parameter
std_sheep_reproduce = 1.0  # arbitrary
mean_grass_regrowth_time = 30.0  # parameter
std_grass_regrowth_time = 1.0  # arbitrary

mean_vec = np.array(
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

cov_matrix = np.diag(
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
) = np.abs(multivariate_normal(mean=mean_vec, cov=cov_matrix).rvs())

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

fig = plt.figure()
ax = fig.gca()
ax.plot(vp_wolf_counts, label="wolves")
ax.plot(vp_sheep_counts, label="sheep")
ax.plot(vp_grass_counts, label="grass")
ax.legend()
fig.savefig(FILE_PREFIX + "virtual-patient.pdf")
plt.close(fig)


################################################################################


def model_ensemble_from(means, covariances):
    """
    Create an ensemble of models from a distribution

    :param means:
    :param covariances:
    :return:
    """
    mdl_ensemble = []
    distribution = multivariate_normal(mean=means, cov=covariances)
    for _ in range(ENSEMBLE_SIZE):
        (
            en_init_wolves,
            en_init_sheep,
            en_init_grass,
            en_wolf_gain_from_food,
            en_sheep_gain_from_food,
            en_wolf_reproduce,
            en_sheep_reproduce,
            en_grass_regrowth_time,
        ) = np.abs(distribution.rvs())
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
    fix_grass_clocks: bool = False,
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
    ) = np.abs(desired_state)
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
        print(f"creating {num_wolves - model.num_wolves} new wolves")
        for _ in range(num_wolves - model.num_wolves):
            model.create_wolf()
    elif num_wolves < model.num_wolves:
        print(f"killing {model.num_wolves - num_wolves} wolves")
        try:
            for _ in range(model.num_wolves - num_wolves):
                model.kill_random_wolf()
        except RuntimeError as e:
            print(e)

    num_sheep = int(num_sheep)
    if num_sheep > model.num_sheep:
        print(f"creating {num_sheep - model.num_sheep} new sheep")
        for _ in range(num_sheep - model.num_sheep):
            model.create_sheep()
    elif num_sheep < model.num_sheep:
        print(f"killing {model.num_sheep - num_sheep} sheep")
        try:
            for _ in range(model.num_sheep - num_sheep):
                model.kill_random_sheep()
        except RuntimeError as e:
            print(e)

    num_grass = int(num_grass)
    grass_present = np.sum(model.grass)
    if num_grass > grass_present:
        print(f"creating {num_grass - grass_present} new grass")
        try:
            for _ in range(num_grass - grass_present):
                model.spawn_grass()
        except RuntimeError as e:
            print(e)
    elif num_grass < grass_present:
        print(f"killing {grass_present - num_grass} grass")
        try:
            for _ in range(grass_present - num_grass):
                model.kill_random_grass()
        except RuntimeError as e:
            print(e)


################################################################################
# Kalman filter simulation
################################################################################

# create ensemble of models for kalman filter
model_ensemble = model_ensemble_from(mean_vec, cov_matrix)


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


# mean and covariances through time
mean_vec = np.zeros((TIME_SPAN + 1, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64)
cov_matrix = np.zeros(
    (TIME_SPAN + 1, UNIFIED_STATE_SPACE_DIMENSION, UNIFIED_STATE_SPACE_DIMENSION),
    dtype=np.float64,
)

# collect initial statistics
time = 0
initial_macro_data = np.array([model_macro_data(model) for model in model_ensemble])
mean_vec[time, :] = np.mean(initial_macro_data, axis=0)
cov_matrix[time, :, :] = np.cov(initial_macro_data, rowvar=False)

cycle = 0
while time < TIME_SPAN:
    cycle += 1
    # advance ensemble of models
    for _ in range(SAMPLE_INTERVAL):
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

                modify_model(model, random_walk_macrostate, ignore_state_vars=True)
        time += 1
        macro_data = np.array([model_macro_data(model) for model in model_ensemble])
        mean_vec[time, :] = np.mean(macro_data, axis=0)
        cov_matrix[time, :, :] = np.cov(macro_data, rowvar=False)

    ################################################################################
    # plot state variables

    fig, axs = plt.subplots(3, figsize=(6, 6))
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
        axs[idx].plot(
            vp_data[state_var_name][: (cycle + 1) * SAMPLE_INTERVAL + 1],
            label="true value",
            color="black",
        )
        axs[idx].plot(
            range(cycle * SAMPLE_INTERVAL + 1),
            mean_vec[: cycle * SAMPLE_INTERVAL + 1, idx],
            label="estimate",
        )
        axs[idx].fill_between(
            range(cycle * SAMPLE_INTERVAL + 1),
            np.maximum(
                0.0,
                mean_vec[: cycle * SAMPLE_INTERVAL + 1, idx]
                - np.sqrt(cov_matrix[: cycle * SAMPLE_INTERVAL + 1, idx, idx]),
            ),
            np.minimum(
                10 * max_scales[state_var_name],
                mean_vec[: cycle * SAMPLE_INTERVAL + 1, idx]
                + np.sqrt(cov_matrix[: cycle * SAMPLE_INTERVAL + 1, idx, idx]),
            ),
            color="gray",
            alpha=0.35,
        )
        axs[idx].set_title(state_var_name)
        axs[idx].legend()
    fig.tight_layout()
    fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-match.pdf")
    plt.close(fig)

    ################################################################################
    # plot state variables

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

    fig, axs = plt.subplots(3, 2, figsize=(8, 8))
    for idx, param_name in enumerate(params):
        row, col = idx % 3, idx // 3
        axs[row, col].plot(
            [0, (cycle + 1) * SAMPLE_INTERVAL + 1],
            [vp_param_values[param_name]] * 2,
            label="true value",
            color="black",
        )
        axs[row, col].plot(
            range(cycle * SAMPLE_INTERVAL + 1),
            mean_vec[: cycle * SAMPLE_INTERVAL + 1, 3 + idx],
            label="estimate",
        )
        axs[row, col].fill_between(
            range(cycle * SAMPLE_INTERVAL + 1),
            np.maximum(
                0.0,
                mean_vec[: cycle * SAMPLE_INTERVAL + 1, 3 + idx]
                - np.sqrt(cov_matrix[: cycle * SAMPLE_INTERVAL + 1, 3 + idx, 3 + idx]),
            ),
            np.minimum(
                10 * vp_param_values[param_name],
                mean_vec[: cycle * SAMPLE_INTERVAL + 1, 3 + idx]
                + np.sqrt(cov_matrix[: cycle * SAMPLE_INTERVAL + 1, 3 + idx, 3 + idx]),
            ),
            color="gray",
            alpha=0.35,
        )
        axs[row, col].set_title(param_name)
        axs[row, col].legend()
    axs[2, 1].axis("off")
    fig.tight_layout()
    fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-match-params.pdf")
    plt.close(fig)

    ################################################################################
    # Kalman filter

    match OBSERVABLE:
        case "wolves":
            R = 2.0
            H = np.zeros((1, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64)
            H[0, 0] = 1.0  # observe wolves
            observation = vp_wolf_counts[time]
        case "sheep":
            R = 2.0
            H = np.zeros((1, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64)
            H[0, 1] = 1.0  # observe sheep
            observation = vp_sheep_counts[time]
        case "grass":
            R = 1.0
            H = np.zeros((1, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64)
            H[0, 2] = 1.0  # observe grass
            observation = vp_grass_counts[time]
        case "wolves+grass":
            R = np.diag([2.0, 1.0])
            H = np.zeros((2, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64)
            H[0, 0] = 1.0  # observe wolves
            H[1, 2] = 1.0  # observe grass
            observation = np.array(
                [vp_wolf_counts[time], vp_grass_counts[time]], dtype=np.float64
            )
        case "sheep+grass":
            R = np.diag([2.0, 1.0])
            H = np.zeros((2, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64)
            H[0, 1] = 1.0  # observe sheep
            H[1, 2] = 1.0  # observe grass
            observation = np.array(
                [vp_sheep_counts[time], vp_grass_counts[time]], dtype=np.float64
            )
        case "wolves+sheep":
            R = np.diag([2.0, 2.0])
            H = np.zeros((2, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64)
            H[0, 0] = 1.0  # observe wolves
            H[1, 1] = 1.0  # observe sheep
            observation = np.array(
                [vp_wolf_counts[time], vp_sheep_counts[time]], dtype=np.float64
            )
        case "wolves+sheep+grass":
            R = np.diag([2.0, 2.0, 1.0])
            H = np.zeros((3, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64)
            H[0, 0] = 1.0  # observe wolves
            H[1, 1] = 1.0  # observe sheep
            H[2, 2] = 1.0  # observe sheep
            observation = np.array(
                [vp_wolf_counts[time], vp_sheep_counts[time], vp_grass_counts[time]],
                dtype=np.float64,
            )
        case _:
            raise RuntimeError("unknown observable?")

    v = observation - (H @ mean_vec[time, :])
    S = H @ cov_matrix[time, :, :] @ H.T + R
    K = cov_matrix[time, :, :] @ H.T @ np.linalg.inv(S)

    mean_vec[time, :] += K @ v
    cov_matrix[time, :, :] -= K @ S @ K.T

    # numerical cleanup: symmetrize and project onto pos def cone
    cov_matrix[time, :, :] = (cov_matrix[time, :, :] + cov_matrix[time, :, :].T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix[time, :, :])
    eigenvalues = np.real(eigenvalues)  # just making sure
    eigenvectors = np.real(eigenvectors)  # just making sure
    cov_matrix[time, :, :] = (
        eigenvectors @ np.diag(np.maximum(1e-6, eigenvalues)) @ eigenvectors.T
    )
    cov_matrix[time, :, :] = (cov_matrix[time, :, :] + cov_matrix[time, :, :].T) / 2.0

    # recreate ensemble
    if RESAMPLE_MODELS:
        # create an entirely new set of model instances sampled from KF-learned distribution
        model_ensemble = model_ensemble_from(mean_vec[time, :], cov_matrix[time, :, :])
    else:
        # sample from KF-learned dist and modify existing models to fit
        dist = multivariate_normal(mean=mean_vec[time, :], cov=cov_matrix[time, :, :])
        for model in model_ensemble:
            state = dist.rvs()
            modify_model(model, state)

################################################################################

vp_full_trajectory = np.array(
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

delta_full = mean_vec - vp_full_trajectory
surprisal_full = np.einsum(
    "ij,ij->i", delta_full, np.linalg.solve(cov_matrix, delta_full)
)
mean_surprisal_full = np.mean(surprisal_full)

vp_state_trajectory = np.array(
    (
        vp_wolf_counts,
        vp_sheep_counts,
        vp_grass_counts,
    )
).T
delta_state = mean_vec[:, :3] - vp_state_trajectory
surprisal_state = np.einsum(
    "ij,ij->i", delta_state, np.linalg.solve(cov_matrix[:, :3, :3], delta_state)
)
mean_surprisal_state = np.mean(surprisal_state)

vp_param_trajectory = np.array(
    (
        [vp_wolf_gain_from_food] * (TIME_SPAN + 1),
        [vp_sheep_gain_from_food] * (TIME_SPAN + 1),
        [vp_wolf_reproduce] * (TIME_SPAN + 1),
        [vp_sheep_reproduce] * (TIME_SPAN + 1),
        [vp_grass_regrowth_time] * (TIME_SPAN + 1),
    )
).T
delta_param = mean_vec[:, 3:] - vp_param_trajectory
surprisal_param = np.einsum(
    "ij,ij->i", delta_param, np.linalg.solve(cov_matrix[:, 3:, 3:], delta_param)
)
mean_surprisal_param = np.mean(surprisal_param)

plt.plot(surprisal_full, label="full surprisal")
plt.plot(surprisal_state, label="state surprisal")
plt.plot(surprisal_param, label="param surprisal")
plt.legend()
plt.tight_layout()
plt.savefig(FILE_PREFIX + f"surprisal.pdf")
plt.close()

fig, axs = plt.subplots(4, figsize=(6, 8))
plural = {"wolf": "wolves", "sheep": "sheep", "grass": "grass"}
vp_data = {
    "wolf": vp_wolf_counts,
    "sheep": vp_sheep_counts,
    "grass": vp_grass_counts,
}
max_scales = {
    "wolf": 10 * mean_init_wolves,
    "sheep": 10 * mean_init_sheep,
    "grass": 10 * mean_init_grass_proportion * GRID_HEIGHT * GRID_WIDTH,
}
for idx, state_var_name in enumerate(["wolf", "sheep", "grass"]):
    axs[idx].plot(
        vp_data[state_var_name],
        label="true value",
        color="black",
    )
    axs[idx].plot(
        range(TIME_SPAN + 1),
        mean_vec[:, idx],
        label="estimate",
    )
    axs[idx].fill_between(
        range(TIME_SPAN + 1),
        np.maximum(
            0.0,
            mean_vec[:, idx] - np.sqrt(cov_matrix[:, idx, idx]),
        ),
        np.minimum(
            max_scales[state_var_name],
            mean_vec[:, idx] + np.sqrt(cov_matrix[:, idx, idx]),
        ),
        color="gray",
        alpha=0.35,
    )
    axs[idx].set_title(state_var_name)
    axs[idx].legend()
axs[3].set_title("surprisal")
axs[3].plot(surprisal_state, label="state surprisal")
axs[3].plot(
    [0, TIME_SPAN + 1], [mean_surprisal_state, mean_surprisal_state], ":", color="black"
)
fig.tight_layout()
fig.savefig(FILE_PREFIX + f"match.pdf")
plt.close(fig)

np.savez_compressed(
    FILE_PREFIX + f"data.npz",
    vp_full_trajectory=vp_full_trajectory,
    mean_vec=mean_vec,
    cov_matrix=cov_matrix,
)
with open(FILE_PREFIX + "meansurprisal.csv", "w") as file:
    file.write(",".join(["full", "state", "param"]))
    file.write(
        ",".join(
            [
                str(mean_surprisal_full),
                str(mean_surprisal_state),
                str(mean_surprisal_param),
            ]
        )
    )
