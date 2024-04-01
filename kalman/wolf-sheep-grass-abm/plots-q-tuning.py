import os

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase

from transform import transform_kf_to_intrinsic

fig, axs = plt.subplots(2, 2, figsize=(6, 5), sharex=True, sharey=True)

orgs = []
finals = []
for p_idx, prefix in enumerate(["0.001", "0.005", "0.010", "0.050"]):
    files = [
        f"./st-g-{prefix}/"+f
        for f in os.listdir(f"./st-g-{prefix}")
        if os.path.isfile(f"./st-g-{prefix}/"+f) and f.startswith("st-g-") and f.endswith(prefix + "-data.hdf5")
    ]
    surp_init = []
    surp_final = []
    for file in files:
        with h5py.File(file, "r") as h5file:
            surp_init.append(h5file["surprisal_full"][0, :])
            surp_final.append(h5file["surprisal_full"][-1, :])
    surp_init = np.array(surp_init)
    surp_final = np.array(surp_final)

    orgs.append(
        axs[0, 0].plot(
            np.median(surp_init, axis=0),
            label="original surprisal",
            color=mpl.colormaps["tab10"](p_idx),
            linestyle=":",
        )[0]
    )
    finals.append(
        axs[0, 0].plot(
            np.median(surp_final, axis=0)[:-1],
            label="final " + prefix,
            color=mpl.colormaps["tab10"](p_idx),
        )[0]
    )

axs[0, 0].set_ylabel("surprisal")
axs[0, 0].set_xlabel("time")
axs[0, 0].title.set_text("Measuring Grass")

################################################################################

orgs = []
finals = []
for p_idx, prefix in enumerate(["0.001", "0.005", "0.010", "0.050"]):
    files = [
        f"./st-s-{prefix}/"+f
        for f in os.listdir(f"./st-s-{prefix}/")
        if os.path.isfile(f"./st-s-{prefix}/"+f) and f.startswith("st-s-") and f.endswith(prefix + "-data.hdf5")
    ]
    surp_init = []
    surp_final = []
    for file in files:
        with h5py.File(file, "r") as h5file:
            surp_init.append(h5file["surprisal_full"][0, :])
            surp_final.append(h5file["surprisal_full"][-1, :])
    surp_init = np.array(surp_init)
    surp_final = np.array(surp_final)
    print(surp_init.shape)

    orgs.append(
        axs[0, 1].plot(
            np.median(surp_init, axis=0),
            label="surprisal w/o KF",
            color=mpl.colormaps["tab10"](p_idx),
            linestyle=":",
        )[0]
    )
    finals.append(
        axs[0, 1].plot(
            np.median(surp_final, axis=0)[:-1],
            label="Process stoch. " + prefix,
            color=mpl.colormaps["tab10"](p_idx),
        )[0]
    )

axs[0, 1].set_ylabel("surprisal")
axs[0, 1].set_xlabel("time")
axs[0, 1].title.set_text("Measuring Sheep")

################################################################################

orgs = []
finals = []
for p_idx, prefix in enumerate(["0.001", "0.005", "0.010", "0.050"]):
    files = [
        f"./st-w-{prefix}/"+f
        for f in os.listdir(f"./st-w-{prefix}/")
        if os.path.isfile(f"./st-w-{prefix}/"+f) and f.startswith("st-w-") and f.endswith(prefix + "-data.hdf5")
    ]
    surp_init = []
    surp_final = []
    for file in files:
        with h5py.File(file, "r") as h5file:
            surp_init.append(h5file["surprisal_full"][0, :])
            surp_final.append(h5file["surprisal_full"][-1, :])
    surp_init = np.array(surp_init)
    surp_final = np.array(surp_final)
    print(surp_init.shape)

    orgs.append(
        axs[1, 0].plot(
            np.median(surp_init, axis=0),
            label="original surprisal",
            color=mpl.colormaps["tab10"](p_idx),
            linestyle=":",
        )[0]
    )
    finals.append(
        axs[1, 0].plot(
            np.median(surp_final, axis=0)[:-1],
            label="Process stoch. " + prefix,
            color=mpl.colormaps["tab10"](p_idx),
        )[0]
    )

axs[1, 0].set_ylabel("surprisal")
axs[1, 0].set_xlabel("time")
axs[1, 0].title.set_text("Measuring Wolves")

axs[1, 1].axes.set_axis_off()


class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        size = len(orig_handle)
        ls = []
        for idx, handle in enumerate(orig_handle):
            h = (size - idx) / (size + 1) * height
            ls.append(
                plt.Line2D(
                    [x0, y0 + width],
                    [h, h],
                    linestyle=handle.get_linestyle(),
                    color=handle.get_color(),
                )
            )

        return ls


axs[1,1].legend(
    tuple(zip(finals, orgs)),
    [*map(lambda x: x.get_label(), finals)],
    loc="upper left",
    handler_map={tuple: AnyObjectHandler()},
)

fig.suptitle("Surprisal after measurement")
fig.tight_layout()
fig.savefig("q-measurement-uncertainty.pdf")
plt.close(fig)

################################################################################
################################################################################
################################################################################

high_surp_files = []
for p_idx, prefix in enumerate(["10.00", "01.00", "00.10", "00.01"]):
    files = [
        f
        for f in os.listdir(".")
        if os.path.isfile(f) and f[:7] == "w-" + prefix and f[-5:] == ".hdf5"
    ]
    surp_init = []
    surp_final = []
    for file in files:
        with h5py.File(file, "r") as h5file:
            if np.any(h5file["surprisal_full"][-1, :] >= 100_000):
                high_surp_files.append(file)
# w-00.01-0107-data.hdf5

with h5py.File(high_surp_files[0], "r") as h5file:
    # plt.plot(h5file['surprisal_full'][-1, :])
    # plt.plot(transform_kf_to_intrinsic(h5file['virtual_patient_trajectory'][:, 0], index=0))

    fig, axs = plt.subplots(3, figsize=(6, 6), sharex=True, sharey=False, layout="constrained")
    plural = {"wolf": "wolves", "sheep": "sheep", "grass": "grass"}

    for idx, state_var_name in enumerate(["wolf", "sheep", "grass"]):
        (true_value,) = axs[idx].plot(
            transform_kf_to_intrinsic(h5file["virtual_patient_trajectory"][:, idx], index=idx),
            label="true value",
            # linestyle=":",
            color="black",
        )

        mu = h5file["means"][-1, :, idx]
        sigma = np.sqrt(h5file["covs"][-1, :, idx, idx])

        (prediction_center_line,) = axs[idx].plot(
            transform_kf_to_intrinsic(mu, index=idx),
            label="prediction",
            color="blue",
        )
        prediction_range = axs[idx].fill_between(
            range(len(mu)),
            transform_kf_to_intrinsic(mu - sigma, index=idx),
            transform_kf_to_intrinsic(mu + sigma, index=idx),
            color="blue",
            alpha=0.35,
        )
        axs[idx].set_title(state_var_name, loc="left")
    # noinspection PyUnboundLocalVariable
    fig.legend(
        [
            true_value,
            (prediction_center_line, prediction_range),
        ],
        [
            true_value.get_label(),
            prediction_center_line.get_label(),
        ],
        # loc="outside upper right",
    )
    fig.suptitle("State Prediction")
    fig.savefig("wolf-extinction.pdf")
    plt.close(fig)

################################################################################
################################################################################
################################################################################


normal_surp_files = []
for p_idx, prefix in enumerate(["00.01"]):
    files = [
        f
        for f in os.listdir(".")
        if os.path.isfile(f) and f[:7] == "w-" + prefix and f[-5:] == ".hdf5"
    ]
    surp_init = []
    surp_final = []
    for file in files:
        with h5py.File(file, "r") as h5file:
            if np.all(h5file["surprisal_full"][-1, :] <= 100_000):
                normal_surp_files.append(file)

# 'w-00.01-0597-data.hdf5'


with h5py.File(normal_surp_files[-1], "r") as h5file:
    fig, axs = plt.subplots(3, figsize=(6, 6), sharex=True, sharey=False, layout="constrained")
    plural = {"wolf": "wolves", "sheep": "sheep", "grass": "grass"}

    for idx, state_var_name in enumerate(["wolf", "sheep", "grass"]):
        (true_value,) = axs[idx].plot(
            transform_kf_to_intrinsic(h5file["virtual_patient_trajectory"][:, idx], index=idx),
            label="true value",
            # linestyle=":",
            color="black",
        )

        mu = h5file["means"][-1, :, idx]
        sigma = np.sqrt(h5file["covs"][-1, :, idx, idx])

        (prediction_center_line,) = axs[idx].plot(
            transform_kf_to_intrinsic(mu, index=idx),
            label="prediction",
            color="blue",
        )
        prediction_range = axs[idx].fill_between(
            range(len(mu)),
            transform_kf_to_intrinsic(mu - sigma, index=idx),
            transform_kf_to_intrinsic(mu + sigma, index=idx),
            color="blue",
            alpha=0.35,
        )
        axs[idx].set_title(state_var_name, loc="left")

        max_tv = np.max(
            transform_kf_to_intrinsic(h5file["virtual_patient_trajectory"][:, idx], index=idx)
        )
        axs[idx].set_ylim(max(0, axs[idx].get_ylim()[0]), min(2 * max_tv, axs[idx].get_ylim()[1]))
    # noinspection PyUnboundLocalVariable
    fig.legend(
        [
            true_value,
            (prediction_center_line, prediction_range),
        ],
        [
            true_value.get_label(),
            prediction_center_line.get_label(),
        ],
        # loc="outside upper right",
    )
    fig.suptitle("State Prediction")
    fig.savefig("typical-wolf-state.pdf")
    plt.close(fig)

    ################################################################################

with h5py.File(normal_surp_files[-1], "r") as h5file:
    params = [
        "wolf gain from food",
        "sheep gain from food",
        "wolf reproduce",
        "sheep reproduce",
        "grass regrowth time",
    ]

    fig, axs = plt.subplots(3, 2, figsize=(8, 8), sharex=True, sharey=False, layout="constrained")
    for idx, param_name in enumerate(params):
        row, col = idx % 3, idx // 3

        mu = h5file["means"][-1, :, idx + 3]
        sigma = np.sqrt(h5file["covs"][-1, :, idx + 3, idx + 3])

        (true_value,) = axs[row, col].plot(
            transform_kf_to_intrinsic(
                h5file["virtual_patient_trajectory"][:, idx + 3], index=idx + 3
            ),
            label="true value",
            color="black",
            # linestyle=":",
        )

        (prediction_center_line,) = axs[row, col].plot(
            transform_kf_to_intrinsic(mu, index=idx + 3),
            label="prediction",
            color="blue",
        )
        prediction_range = axs[row, col].fill_between(
            range(len(mu)),
            transform_kf_to_intrinsic(mu - sigma, index=idx + 3),
            transform_kf_to_intrinsic(mu + sigma, index=idx + 3),
            color="blue",
            alpha=0.35,
        )

        max_tv = np.max(
            transform_kf_to_intrinsic(
                h5file["virtual_patient_trajectory"][:, idx + 3], index=idx + 3
            )
        )
        axs[row, col].set_ylim(
            max(0, axs[row, col].get_ylim()[0]), min(5 * max_tv, axs[row, col].get_ylim()[1])
        )
        axs[row, col].set_title(param_name, loc="left")
    axs[2, 1].axis("off")
    # noinspection PyUnboundLocalVariable
    fig.legend(
        [
            true_value,
            (prediction_center_line, prediction_range),
        ],
        [
            true_value.get_label(),
            prediction_center_line.get_label(),
        ],
        loc="lower right",
    )
    fig.suptitle("Parameters")
    fig.savefig("typical-wolf-param.pdf")
    plt.close(fig)

################################################################################
################################################################################
################################################################################

normal_surp_files = []
for p_idx, prefix in enumerate(["00.01"]):
    files = [
        f
        for f in os.listdir(".")
        if os.path.isfile(f) and f[:7] == "s-" + prefix and f[-5:] == ".hdf5"
    ]
    surp_init = []
    surp_final = []
    for file in files:
        with h5py.File(file, "r") as h5file:
            if np.all(h5file["surprisal_full"][-1, :] <= 100_000):
                normal_surp_files.append(file)

# 'w-00.01-0597-data.hdf5'


with h5py.File(normal_surp_files[-1], "r") as h5file:
    fig, axs = plt.subplots(3, figsize=(6, 6), sharex=True, sharey=False, layout="constrained")
    plural = {"wolf": "wolves", "sheep": "sheep", "grass": "grass"}

    for idx, state_var_name in enumerate(["wolf", "sheep", "grass"]):
        (true_value,) = axs[idx].plot(
            transform_kf_to_intrinsic(h5file["virtual_patient_trajectory"][:, idx], index=idx),
            label="true value",
            # linestyle=":",
            color="black",
        )

        mu = h5file["means"][-1, :, idx]
        sigma = np.sqrt(h5file["covs"][-1, :, idx, idx])

        (prediction_center_line,) = axs[idx].plot(
            transform_kf_to_intrinsic(mu, index=idx),
            label="prediction",
            color="blue",
        )
        prediction_range = axs[idx].fill_between(
            range(len(mu)),
            transform_kf_to_intrinsic(mu - sigma, index=idx),
            transform_kf_to_intrinsic(mu + sigma, index=idx),
            color="blue",
            alpha=0.35,
        )
        axs[idx].set_title(state_var_name, loc="left")

        max_tv = np.max(
            transform_kf_to_intrinsic(h5file["virtual_patient_trajectory"][:, idx], index=idx)
        )
        axs[idx].set_ylim(max(0, axs[idx].get_ylim()[0]), min(2 * max_tv, axs[idx].get_ylim()[1]))
    # noinspection PyUnboundLocalVariable
    fig.legend(
        [
            true_value,
            (prediction_center_line, prediction_range),
        ],
        [
            true_value.get_label(),
            prediction_center_line.get_label(),
        ],
        # loc="outside upper right",
    )
    fig.suptitle("State Prediction")
    fig.savefig("typical-sheep-state.pdf")
    plt.close(fig)

    ################################################################################

    params = [
        "wolf gain from food",
        "sheep gain from food",
        "wolf reproduce",
        "sheep reproduce",
        "grass regrowth time",
    ]

    fig, axs = plt.subplots(3, 2, figsize=(8, 8), sharex=True, sharey=False, layout="constrained")
    for idx, param_name in enumerate(params):
        row, col = idx % 3, idx // 3

        mu = h5file["means"][-1, :, idx + 3]
        sigma = np.sqrt(h5file["covs"][-1, :, idx + 3, idx + 3])

        (true_value,) = axs[row, col].plot(
            transform_kf_to_intrinsic(
                h5file["virtual_patient_trajectory"][:, idx + 3], index=idx + 3
            ),
            label="true value",
            color="black",
            # linestyle=":",
        )

        (prediction_center_line,) = axs[row, col].plot(
            transform_kf_to_intrinsic(mu, index=idx + 3),
            label="prediction",
            color="blue",
        )
        prediction_range = axs[row, col].fill_between(
            range(len(mu)),
            transform_kf_to_intrinsic(mu - sigma, index=idx + 3),
            transform_kf_to_intrinsic(mu + sigma, index=idx + 3),
            color="blue",
            alpha=0.35,
        )

        max_tv = np.max(
            transform_kf_to_intrinsic(
                h5file["virtual_patient_trajectory"][:, idx + 3], index=idx + 3
            )
        )
        axs[row, col].set_ylim(
            max(0, axs[row, col].get_ylim()[0]), min(5 * max_tv, axs[row, col].get_ylim()[1])
        )
        axs[row, col].set_title(param_name, loc="left")
    axs[2, 1].axis("off")
    # noinspection PyUnboundLocalVariable
    fig.legend(
        [
            true_value,
            (prediction_center_line, prediction_range),
        ],
        [
            true_value.get_label(),
            prediction_center_line.get_label(),
        ],
        loc="lower right",
    )
    fig.suptitle("Parameters")
    fig.savefig("typical-sheep-param.pdf")
    plt.close(fig)

################################################################################
################################################################################
################################################################################

normal_surp_files = []
for p_idx, prefix in enumerate(["00.01"]):
    files = [
        f
        for f in os.listdir(".")
        if os.path.isfile(f) and f[:7] == "g-" + prefix and f[-5:] == ".hdf5"
    ]
    surp_init = []
    surp_final = []
    for file in files:
        with h5py.File(file, "r") as h5file:
            if np.all(h5file["surprisal_full"][-1, :] <= 100_000):
                normal_surp_files.append(file)

# 'w-00.01-0597-data.hdf5'


with h5py.File(normal_surp_files[-1], "r") as h5file:
    fig, axs = plt.subplots(3, figsize=(6, 6), sharex=True, sharey=False, layout="constrained")
    plural = {"wolf": "wolves", "sheep": "sheep", "grass": "grass"}

    for idx, state_var_name in enumerate(["wolf", "sheep", "grass"]):
        (true_value,) = axs[idx].plot(
            transform_kf_to_intrinsic(h5file["virtual_patient_trajectory"][:, idx], index=idx),
            label="true value",
            # linestyle=":",
            color="black",
        )

        mu = h5file["means"][-1, :, idx]
        sigma = np.sqrt(h5file["covs"][-1, :, idx, idx])

        (prediction_center_line,) = axs[idx].plot(
            transform_kf_to_intrinsic(mu, index=idx),
            label="prediction",
            color="blue",
        )
        prediction_range = axs[idx].fill_between(
            range(len(mu)),
            transform_kf_to_intrinsic(mu - sigma, index=idx),
            transform_kf_to_intrinsic(mu + sigma, index=idx),
            color="blue",
            alpha=0.35,
        )
        axs[idx].set_title(state_var_name, loc="left")

        max_tv = np.max(
            transform_kf_to_intrinsic(h5file["virtual_patient_trajectory"][:, idx], index=idx)
        )
        axs[idx].set_ylim(max(0, axs[idx].get_ylim()[0]), min(2 * max_tv, axs[idx].get_ylim()[1]))
    # noinspection PyUnboundLocalVariable
    fig.legend(
        [
            true_value,
            (prediction_center_line, prediction_range),
        ],
        [
            true_value.get_label(),
            prediction_center_line.get_label(),
        ],
        # loc="outside upper right",
    )
    fig.suptitle("State Prediction")
    fig.savefig("typical-grass-state.pdf")
    plt.close(fig)

    ################################################################################

    params = [
        "wolf gain from food",
        "sheep gain from food",
        "wolf reproduce",
        "sheep reproduce",
        "grass regrowth time",
    ]

    fig, axs = plt.subplots(3, 2, figsize=(8, 8), sharex=True, sharey=False, layout="constrained")
    for idx, param_name in enumerate(params):
        row, col = idx % 3, idx // 3

        mu = h5file["means"][-1, :, idx + 3]
        sigma = np.sqrt(h5file["covs"][-1, :, idx + 3, idx + 3])

        (true_value,) = axs[row, col].plot(
            transform_kf_to_intrinsic(
                h5file["virtual_patient_trajectory"][:, idx + 3], index=idx + 3
            ),
            label="true value",
            color="black",
            # linestyle=":",
        )

        (prediction_center_line,) = axs[row, col].plot(
            transform_kf_to_intrinsic(mu, index=idx + 3),
            label="prediction",
            color="blue",
        )
        prediction_range = axs[row, col].fill_between(
            range(len(mu)),
            transform_kf_to_intrinsic(mu - sigma, index=idx + 3),
            transform_kf_to_intrinsic(mu + sigma, index=idx + 3),
            color="blue",
            alpha=0.35,
        )

        max_tv = np.max(
            transform_kf_to_intrinsic(
                h5file["virtual_patient_trajectory"][:, idx + 3], index=idx + 3
            )
        )
        axs[row, col].set_ylim(
            max(0, axs[row, col].get_ylim()[0]), min(5 * max_tv, axs[row, col].get_ylim()[1])
        )
        axs[row, col].set_title(param_name, loc="left")
    axs[2, 1].axis("off")
    # noinspection PyUnboundLocalVariable
    fig.legend(
        [
            true_value,
            (prediction_center_line, prediction_range),
        ],
        [
            true_value.get_label(),
            prediction_center_line.get_label(),
        ],
        loc="lower right",
    )
    fig.suptitle("Parameters")
    fig.savefig("typical-grass-param.pdf")
    plt.close(fig)

################################################################################
################################################################################
################################################################################


low_surp_files = []
for p_idx, prefix in enumerate(["00.01"]):
    files = [
        f
        for f in os.listdir(".")
        if os.path.isfile(f) and f[:7] == "w-" + prefix and f[-5:] == ".hdf5"
    ]
    surp_init = []
    surp_final = []
    for file in files:
        with h5py.File(file, "r") as h5file:
            if np.all(h5file["surprisal_full"][-1, :] <= 20):
                low_surp_files.append(file)

#
# def sing_val_projs(M, idx):
#     evals, evecs = np.linalg.eigh(M)
#     return np.abs(evecs[:,idx,:] * evals)
