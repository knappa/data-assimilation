#!/usr/bin/env python3
# Surprisal plots from bulk runs
import os

import h5py
import numpy as np
from matplotlib import pyplot as plt

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
        if os.path.isfile(f) and f[-5:] == ".hdf5" and f[: len(short_name) + 1] == short_name + "-"
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
            np.mean(np.mean(surprisal_state[short_name], axis=0)[idx, 50 * idx : 50 * (idx + 1)])
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
            np.mean(np.mean(surprisal_param[short_name], axis=0)[idx, 50 * idx : 50 * (idx + 1)])
            for idx in range(20)
        ],
        label=name,
    )
fig.suptitle("Parameter surprisal")
fig.legend()
fig.savefig("surp-param.pdf")
plt.close(fig)
