#!/usr/bin/env python3
# Surprisal plots from bulk runs
import os

import h5py
import numpy as np
from matplotlib import pyplot as plt

from transform import transform_intrinsic_to_kf

surprisal_full = []
surprisal_state = []
surprisal_param = []
surprisal_full_quad = []
surprisal_state_quad = []
surprisal_param_quad = []
final_wolves = []
means = []
covs = []
vp_traj = []

powers = ["00.01", "00.10", "01.00", "10.00"]

for rpow_idx, rpow in enumerate(powers):
    files = [
        f
        for f in os.listdir(".")
        if os.path.isfile(f) and f[:7] == "g-" + rpow and f[-5:] == ".hdf5"
    ]

    # get shapes
    with h5py.File(files[0], "r") as h5file:
        full_shape = h5file["surprisal_full"][()].shape
        state_shape = h5file["surprisal_state"][()].shape
        param_shape = h5file["surprisal_param"][()].shape
        means_shape = h5file["means"][()].shape
        covs_shape = h5file["covs"][()].shape
        vp_traj_shape = h5file["virtual_patient_trajectory"][()].shape

    # init arrays
    surprisal_full.append(np.zeros((len(files), 20), dtype=np.float64))
    surprisal_state.append(np.zeros((len(files), 20), dtype=np.float64))
    surprisal_param.append(np.zeros((len(files), 20), dtype=np.float64))
    surprisal_full_quad.append(np.zeros((len(files), 20), dtype=np.float64))
    surprisal_state_quad.append(np.zeros((len(files), 20), dtype=np.float64))
    surprisal_param_quad.append(np.zeros((len(files), 20), dtype=np.float64))

    final_wolves.append(np.zeros(len(files)))
    means.append(np.zeros((len(files), *means_shape), dtype=np.float64))
    covs.append(np.zeros((len(files), *covs_shape), dtype=np.float64))
    vp_traj.append(np.zeros((len(files), *vp_traj_shape), dtype=np.float64))

    # collect data
    for file_idx, file in enumerate(files):
        with h5py.File(file, "r") as h5file:
            final_wolves[rpow_idx][file_idx] = h5file["virtual_patient_trajectory"][-1, 0]

            surprisal_full[rpow_idx][file_idx] = h5file["surprisal_full"][()]
            np.array(
                [h5file["surprisal_full"][idx, 50 * idx : 50 * (idx + 1)] for idx in range(20)]
            )

            surprisal_state[rpow_idx][file_idx] = h5file["surprisal_state"][()]
            surprisal_param[rpow_idx][file_idx] = h5file["surprisal_param"][()]
            surprisal_full_quad[rpow_idx][file_idx] = h5file["surprisal_full_quad"][()]
            surprisal_state_quad[rpow_idx][file_idx] = h5file["surprisal_state_quad"][()]
            surprisal_param_quad[rpow_idx][file_idx] = h5file["surprisal_param_quad"][()]

            means[rpow_idx][file_idx] = h5file["means"][()]
            covs[rpow_idx][file_idx] = h5file["covs"][()]
            vp_traj[rpow_idx][file_idx] = transform_intrinsic_to_kf(
                h5file["virtual_patient_trajectory"][()]
            )

print(f"number of sims with wolf extinction: {np.sum(final_wolves == 0.0)}")

fig = plt.figure()
ax = fig.gca()
ax.plot(
    [np.mean(np.mean(surprisal_full, axis=0)[idx, 50 * idx : 50 * (idx + 1)]) for idx in range(20)],
    label="full",
)
ax.plot(
    [
        np.mean(np.mean(surprisal_state, axis=0)[idx, 50 * idx : 50 * (idx + 1)])
        for idx in range(20)
    ],
    label="state",
)
ax.plot(
    [
        np.mean(np.mean(surprisal_param, axis=0)[idx, 50 * idx : 50 * (idx + 1)])
        for idx in range(20)
    ],
    label="param",
)
ax.legend()
fig.savefig("surprisals.pdf")

fig, axs = plt.subplots(2)
axs[0].plot(
    [
        np.mean(
            np.mean(surprisal_full[final_wolves == 0.0], axis=0)[idx, 50 * idx : 50 * (idx + 1)]
        )
        for idx in range(20)
    ],
    label="full",
)
axs[0].plot(
    [
        np.mean(
            np.mean(surprisal_state[final_wolves == 0.0], axis=0)[idx, 50 * idx : 50 * (idx + 1)]
        )
        for idx in range(20)
    ],
    label="state",
)
axs[0].plot(
    [
        np.mean(
            np.mean(surprisal_param[final_wolves == 0.0], axis=0)[idx, 50 * idx : 50 * (idx + 1)]
        )
        for idx in range(20)
    ],
    label="param",
)
axs[0].legend()
axs[0].set_title("Surprisal for sims where the wolves go extinct")

axs[1].plot(
    [
        np.mean(
            np.mean(surprisal_full[final_wolves != 0.0], axis=0)[idx, 50 * idx : 50 * (idx + 1)]
        )
        for idx in range(20)
    ],
    label="full",
)
axs[1].plot(
    [
        np.mean(
            np.mean(surprisal_state[final_wolves != 0.0], axis=0)[idx, 50 * idx : 50 * (idx + 1)]
        )
        for idx in range(20)
    ],
    label="state",
)
axs[1].plot(
    [
        np.mean(
            np.mean(surprisal_param[final_wolves != 0.0], axis=0)[idx, 50 * idx : 50 * (idx + 1)]
        )
        for idx in range(20)
    ],
    label="param",
)
axs[1].legend()
axs[1].set_title("Surprisal for sims where the wolves don't go extinct")
fig.savefig("surprisals-wolf.pdf")

components = [
    "wolves",
    "sheep",
    "grass",
    "wolf gain from food",
    "sheep gain from food",
    "wolf reproduce",
    "sheep reproduce",
    "grass regrowth time",
]
fig, axs = plt.subplots(2, 2, sharex=True)
for cpt_idx, cpt_name in enumerate(components):
    if cpt_idx < 3:
        row = 0
    else:
        row = 1
    cpt_surp = (
        covs[..., cpt_idx, cpt_idx] ** -1
        * ((means[..., cpt_idx] - vp_traj[..., cpt_idx][:, np.newaxis, :]) ** 2)
        + np.log(covs[..., cpt_idx, cpt_idx])
    ) / 2.0

    cpt_surp_wolves = cpt_surp[final_wolves != 0.0]
    cpt_surp_wolves_example_mean = np.mean(
        cpt_surp_wolves, axis=0, where=np.logical_not(np.isnan(cpt_surp_wolves))
    )
    axs[row, 0].plot(
        (
            [
                np.mean(cpt_surp_wolves_example_mean[idx, 50 * idx : 50 * (idx + 1)])
                for idx in range(20)
            ]
        ),
        label=cpt_name,
    )
    axs[row, 0].set_title("wolves")

    cpt_surp_no_wolves = cpt_surp[final_wolves == 0.0]
    cpt_surp_no_wolves_example_mean = np.mean(
        cpt_surp_no_wolves, axis=0, where=np.logical_not(np.isnan(cpt_surp_no_wolves))
    )
    axs[row, 1].plot(
        (
            [
                np.mean(cpt_surp_no_wolves_example_mean[idx, 50 * idx : 50 * (idx + 1)])
                for idx in range(20)
            ]
        ),
        label=cpt_name,
    )
    axs[row, 1].set_title("no wolves")
axs[0, 0].legend()
axs[1, 0].legend()
# fig.savefig("surprisals-components.pdf")

fig, axs = plt.subplots(2, sharex=True)
for cpt_idx, cpt_name in enumerate(components):
    if cpt_idx < 3:
        row = 0
    else:
        row = 1

    cpt_dist = (means[..., cpt_idx] - vp_traj[..., cpt_idx][:, np.newaxis, :]) ** 2
    mean_prop_dist = np.mean(cpt_dist / cpt_dist[:, :, 0][:, :, np.newaxis], axis=0)
    cycle_prop_dist = np.array(
        [np.sqrt(np.mean(mean_prop_dist[idx, 50 * idx : 50 * (idx + 1)])) for idx in range(20)]
    )

    axs[row].plot(
        (cycle_prop_dist),
        label=cpt_name,
    )

axs[0].legend()
axs[1].legend()
# fig.savefig("surprisals-components.pdf")
