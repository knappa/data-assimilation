#!/usr/bin/env python3
# Surprisal plots from bulk runs
import argparse
import os
import sys

import h5py
import numpy as np
from matplotlib import pyplot as plt

if hasattr(sys, "ps1"):
    # interactive mode
    args = object()
else:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bulk", help="do all the hdf5's in this directory", action="store_true")
    args = parser.parse_args()

BULK = True if not hasattr(args, "bulk") or args.bulk is None else args.bulk

if BULK:

    files = [f for f in os.listdir(".") if os.path.isfile(f) and f[-5:] == ".hdf5"]

    # get shapes
    with h5py.File(files[0], "r") as h5file:
        full_shape = h5file["surprisal_full"][()].shape
        state_shape = h5file["surprisal_state"][()].shape
        param_shape = h5file["surprisal_param"][()].shape
        means_shape = h5file["means"][()].shape
        covs_shape = h5file["covs"][()].shape
        vp_traj_shape = h5file["virtual_patient_trajectory"][()].shape

    # init arrays
    surprisal_full = np.zeros((len(files), *full_shape), dtype=np.float64)
    surprisal_state = np.zeros((len(files), *state_shape), dtype=np.float64)
    surprisal_param = np.zeros((len(files), *param_shape), dtype=np.float64)
    surprisal_full_quad = np.zeros((len(files), *full_shape), dtype=np.float64)
    surprisal_state_quad = np.zeros((len(files), *state_shape), dtype=np.float64)
    surprisal_param_quad = np.zeros((len(files), *param_shape), dtype=np.float64)

    final_wolves = np.zeros(len(files))
    means = np.zeros((len(files), *means_shape), dtype=np.float64)
    covs = np.zeros((len(files), *covs_shape), dtype=np.float64)
    vp_traj = np.zeros((len(files), *vp_traj_shape), dtype=np.float64)

    # collect data
    for file_idx, file in enumerate(files):
        with h5py.File(file, "r") as h5file:
            final_wolves[file_idx] = h5file["virtual_patient_trajectory"][-1, 0]
            surprisal_full[file_idx] = h5file["surprisal_full"][()]
            surprisal_state[file_idx] = h5file["surprisal_state"][()]
            surprisal_param[file_idx] = h5file["surprisal_param"][()]
            surprisal_full_quad[file_idx] = h5file["surprisal_full_quad"][()]
            surprisal_state_quad[file_idx] = h5file["surprisal_state_quad"][()]
            surprisal_param_quad[file_idx] = h5file["surprisal_param_quad"][()]

            means[file_idx] = h5file["means"][()]
            covs[file_idx] = h5file["covs"][()]
            vp_traj[file_idx] = h5file["virtual_patient_trajectory"][()]

    print(f"number of sims with wolf extinction: {np.sum(final_wolves == 0.0)}")

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(
        [
            np.mean(np.mean(surprisal_full, axis=0)[idx, 50 * idx : 50 * (idx + 1)])
            for idx in range(20)
        ],
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
                np.mean(surprisal_state[final_wolves == 0.0], axis=0)[
                    idx, 50 * idx : 50 * (idx + 1)
                ]
            )
            for idx in range(20)
        ],
        label="state",
    )
    axs[0].plot(
        [
            np.mean(
                np.mean(surprisal_param[final_wolves == 0.0], axis=0)[
                    idx, 50 * idx : 50 * (idx + 1)
                ]
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
                np.mean(surprisal_state[final_wolves != 0.0], axis=0)[
                    idx, 50 * idx : 50 * (idx + 1)
                ]
            )
            for idx in range(20)
        ],
        label="state",
    )
    axs[1].plot(
        [
            np.mean(
                np.mean(surprisal_param[final_wolves != 0.0], axis=0)[
                    idx, 50 * idx : 50 * (idx + 1)
                ]
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
    fig, axs = plt.subplots(2)
    for cpt_idx, cpt_name in enumerate(components):
        if cpt_idx < 3:
            continue
        cpt_surp = (
                covs[..., cpt_idx, cpt_idx] ** -1 * (vp_traj[..., cpt_idx] ** 2)[:, np.newaxis, :]
            + np.log(covs[..., cpt_idx, cpt_idx])
        ) / 2.0

        cpt_surp_wolves = cpt_surp[final_wolves != 0.0]
        cpt_surp_wolves_example_mean = np.mean(cpt_surp_wolves, axis=0, where=np.logical_not(np.isnan(cpt_surp_wolves)))
        axs[0].plot(
            ([np.mean(cpt_surp_wolves_example_mean[idx, 50 * idx : 50 * (idx + 1)]) for idx in range(20)]),
            label=cpt_name,
        )
        axs[0].set_title("wolves")

        cpt_surp_no_wolves = cpt_surp[final_wolves == 0.0]
        cpt_surp_no_wolves_example_mean = np.mean(cpt_surp_no_wolves, axis=0, where=np.logical_not(np.isnan(cpt_surp_no_wolves)))
        axs[1].plot(
            ([np.mean(cpt_surp_no_wolves_example_mean[idx, 50 * idx: 50 * (idx + 1)]) for idx in range(20)]),
            label=cpt_name,
        )
        axs[1].set_title("no wolves")
    axs[0].legend()
    axs[1].legend()
    fig.savefig("surprisals-components.pdf")


else:
    types = [
        ("g", "grass"),
        ("s", "sheep"),
        ("w", "wolves"),
        ("sg", "sheep, grass"),
        ("wg", "wolves, grass"),
        ("ws", "wolves, sheep"),
        ("wsg", "wolves, sheep, grass"),
    ]

    surprisal_full = dict()
    surprisal_state = dict()
    surprisal_param = dict()

    for short_name, name in types:
        files = [
            f
            for f in os.listdir(".")
            if os.path.isfile(f)
            and f[-5:] == ".hdf5"
            and f[: len(short_name) + 1] == short_name + "-"
        ]
        # get shapes
        with h5py.File(files[0], "r") as h5file:
            full_shape = h5file["surprisal_full"][()].shape
            state_shape = h5file["surprisal_state"][()].shape
            param_shape = h5file["surprisal_param"][()].shape
        # init arrays
        surprisal_full[short_name] = np.zeros((len(files), *full_shape), dtype=np.float64)
        surprisal_state[short_name] = np.zeros((len(files), *state_shape), dtype=np.float64)
        surprisal_param[short_name] = np.zeros((len(files), *param_shape), dtype=np.float64)
        # collect data
        for file_idx, file in enumerate(files):
            with h5py.File(file, "r") as h5file:
                surprisal_full[short_name][file_idx] = h5file["surprisal_full"][()]
                surprisal_state[short_name][file_idx] = h5file["surprisal_state"][()]
                surprisal_param[short_name][file_idx] = h5file["surprisal_param"][()]

    fig = plt.figure()
    ax = fig.gca()
    for short_name, name in types:
        ax.plot(
            [
                np.mean(np.mean(surprisal_full[short_name], axis=0)[idx, 50 * idx : 50 * (idx + 1)])
                for idx in range(20)
            ],
            label=name,
        )
    fig.suptitle("Full surprisal")
    fig.legend()
    fig.savefig("surp-full.pdf")
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for short_name, name in types:
        ax.plot(
            [
                np.mean(
                    np.mean(surprisal_state[short_name], axis=0)[idx, 50 * idx : 50 * (idx + 1)]
                )
                for idx in range(20)
            ],
            label=name,
        )
    fig.suptitle("State surprisal")
    fig.legend()
    fig.savefig("surp-state.pdf")
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for short_name, name in types:
        ax.plot(
            [
                np.mean(
                    np.mean(surprisal_param[short_name], axis=0)[idx, 50 * idx : 50 * (idx + 1)]
                )
                for idx in range(20)
            ],
            label=name,
        )
    fig.suptitle("Parameter surprisal")
    fig.legend()
    fig.savefig("surp-param.pdf")
    plt.close(fig)
