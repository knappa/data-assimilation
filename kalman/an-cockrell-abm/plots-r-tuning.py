import os

import h5py
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase

from transform import transform_kf_to_intrinsic

fig, axs = plt.subplots(1, 1, figsize=(6.5, 5), sharex=True)

finals = []
for p_idx, prefix in enumerate(["10.00", "01.00", "00.10", "00.01"]):
    files = [
        f
        for f in os.listdir(".")
        if os.path.isfile(f) and f.startswith("cyt-" + prefix) and f.endswith(".hdf5")
    ]
    surp_final = []
    for file in files:
        with h5py.File(file, "r") as h5file:
            if np.all(h5file["surprisal_full"][-1, :] < 100_000):
                surp_final.append(h5file["surprisal_full"][-1, :])
            else:
                print(files.index(file))
    surp_final = np.array(surp_final)

    finals.append(
        axs.plot(
            np.median(surp_final, axis=0)[:-1],
            label="final " + prefix,
            color=mpl.colormaps["tab10"](p_idx),
        )[0]
    )

axs.set_ylabel("surprisal")
axs.set_xlabel("time")
axs.title.set_text("Measuring Cytokines")


# noinspection PyTypeChecker
class AnyObjectHandler(HandlerBase):
    def create_artists(
        self, legend, orig_handle, x0, y0, width, height, fontsize, trans
    ):
        size = len(orig_handle)
        ls = []
        # noinspection PyShadowingNames
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


axs.legend(
    finals,
    [*map(lambda x: x.get_label(), finals)],
    loc="upper left",
    # handler_map={tuple: AnyObjectHandler()},
)

fig.suptitle("Surprisal after measurement")
fig.tight_layout()
fig.savefig("r-measurement-uncertainty.pdf")
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
    fig = plt.figure(figsize=(6.5, 6), constrained_layout=True)
    gs_root = gridspec.GridSpec(nrows=1, ncols=2, figure=fig, width_ratios=[3, 2])

    gs_state = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_root[0])
    axs_state = [fig.add_subplot(gs_state[0, :])]
    for idx in range(1, 3):
        axs_state.append(fig.add_subplot(gs_state[idx, :], sharex=axs_state[0]))
    for idx in range(3 - 1):
        plt.setp(axs_state[idx].get_xticklabels(), visible=False)

    gs_params = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs_root[1])
    axs_params = [fig.add_subplot(gs_params[0, :])]
    for idx in range(1, 5):
        axs_params.append(fig.add_subplot(gs_params[idx, :], sharex=axs_params[0]))
    for idx in range(5 - 1):
        plt.setp(axs_params[idx].get_xticklabels(), visible=False)

    plural = {"wolf": "wolves", "sheep": "sheep", "grass": "grass"}

    for idx, state_var_name in enumerate(["wolf", "sheep", "grass"]):
        axs_state[idx].set_title(state_var_name, loc="left")
        axs_state[idx].set_xlabel("time")
        axs_state[idx].set_ylabel("count")

        (true_value,) = axs_state[idx].plot(
            transform_kf_to_intrinsic(
                h5file["virtual_patient_trajectory"][:, idx], index=idx
            ),
            label="true value",
            # linestyle=":",
            color="black",
        )

        mu = h5file["means"][-1, :, idx]
        sigma = np.sqrt(h5file["covs"][-1, :, idx, idx])

        (prediction_center_line,) = axs_state[idx].plot(
            transform_kf_to_intrinsic(mu, index=idx),
            label="prediction",
            color="blue",
        )
        prediction_range = axs_state[idx].fill_between(
            range(len(mu)),
            transform_kf_to_intrinsic(mu - sigma, index=idx),
            transform_kf_to_intrinsic(mu + sigma, index=idx),
            color="blue",
            alpha=0.35,
        )

    params = [
        "wolf gain from food",
        "sheep gain from food",
        "wolf reproduce",
        "sheep reproduce",
        "grass regrowth time",
    ]
    for idx, param_name in enumerate(params):
        axs_params[idx].set_title(param_name, loc="left")
        axs_params[idx].set_xlabel("time")

        axs_params[idx].plot(
            h5file["virtual_patient_trajectory"][:, 3 + idx],
            label="true value",
            color="black",
            linestyle=":",
        )

        mu = h5file["means"][-1, :, 3 + idx]
        sigma = np.sqrt(h5file["covs"][-1, :, 3 + idx, 3 + idx])

        axs_params[idx].plot(
            transform_kf_to_intrinsic(mu, index=3 + idx),
            color="blue",
            label="prediction",
        )
        TIME_SPAN_p1 = h5file["virtual_patient_trajectory"].shape[0]
        axs_params[idx].fill_between(
            range(TIME_SPAN_p1),
            np.maximum(
                0.0,
                transform_kf_to_intrinsic(
                    mu - sigma,
                    index=3 + idx,
                ),
            ),
            transform_kf_to_intrinsic(
                mu + sigma,
                index=3 + idx,
            ),
            color="blue",
            alpha=0.35,
            label="prediction cone",
        )

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
        loc="outside lower center",
    )
    # fig.suptitle("State Prediction")
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
    fig = plt.figure(figsize=(6.5, 6), constrained_layout=True)
    gs_root = gridspec.GridSpec(nrows=1, ncols=2, figure=fig, width_ratios=[3, 2])

    gs_state = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_root[0])
    axs_state = [fig.add_subplot(gs_state[0, :])]
    for idx in range(1, 3):
        axs_state.append(fig.add_subplot(gs_state[idx, :], sharex=axs_state[0]))
    for idx in range(3 - 1):
        plt.setp(axs_state[idx].get_xticklabels(), visible=False)

    gs_params = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs_root[1])
    axs_params = [fig.add_subplot(gs_params[0, :])]
    for idx in range(1, 5):
        axs_params.append(fig.add_subplot(gs_params[idx, :], sharex=axs_params[0]))
    for idx in range(5 - 1):
        plt.setp(axs_params[idx].get_xticklabels(), visible=False)

    plural = {"wolf": "wolves", "sheep": "sheep", "grass": "grass"}

    for idx, state_var_name in enumerate(["wolf", "sheep", "grass"]):
        axs_state[idx].set_title(state_var_name, loc="left")
        axs_state[idx].set_xlabel("time")
        axs_state[idx].set_ylabel("count")

        (true_value,) = axs_state[idx].plot(
            transform_kf_to_intrinsic(
                h5file["virtual_patient_trajectory"][:, idx], index=idx
            ),
            label="true value",
            # linestyle=":",
            color="black",
        )

        mu = h5file["means"][-1, :, idx]
        sigma = np.sqrt(h5file["covs"][-1, :, idx, idx])

        (prediction_center_line,) = axs_state[idx].plot(
            transform_kf_to_intrinsic(mu, index=idx),
            label="prediction",
            color="blue",
        )
        prediction_range = axs_state[idx].fill_between(
            range(len(mu)),
            transform_kf_to_intrinsic(mu - sigma, index=idx),
            transform_kf_to_intrinsic(mu + sigma, index=idx),
            color="blue",
            alpha=0.35,
        )

        ymin = max(0.0, axs_state[idx].get_ylim()[0])
        ymax = min(
            axs_state[idx].get_ylim()[1],
            1.2
            * np.max(
                transform_kf_to_intrinsic(
                    h5file["virtual_patient_trajectory"][:, idx], index=idx
                )
            ),
        )
        axs_state[idx].set_ylim(bottom=ymin, top=ymax)

    params = [
        "wolf gain from food",
        "sheep gain from food",
        "wolf reproduce",
        "sheep reproduce",
        "grass regrowth time",
    ]
    for idx, param_name in enumerate(params):
        axs_params[idx].set_title(param_name, loc="left")
        axs_params[idx].set_xlabel("time")

        axs_params[idx].plot(
            h5file["virtual_patient_trajectory"][:, 3 + idx],
            label="true value",
            color="black",
            linestyle=":",
        )

        mu = h5file["means"][-1, :, 3 + idx]
        sigma = np.sqrt(h5file["covs"][-1, :, 3 + idx, 3 + idx])

        axs_params[idx].plot(
            transform_kf_to_intrinsic(mu, index=3 + idx),
            color="blue",
            label="prediction",
        )
        TIME_SPAN_p1 = h5file["virtual_patient_trajectory"].shape[0]
        axs_params[idx].fill_between(
            range(TIME_SPAN_p1),
            np.maximum(
                0.0,
                transform_kf_to_intrinsic(
                    mu - sigma,
                    index=3 + idx,
                ),
            ),
            transform_kf_to_intrinsic(
                mu + sigma,
                index=3 + idx,
            ),
            color="blue",
            alpha=0.35,
            label="prediction cone",
        )

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
        loc="outside lower center",
    )
    # fig.suptitle("State Prediction")
    fig.savefig("typical-wolf-state.pdf")
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

# 's-00.01-0254-data.hdf5'


with h5py.File(normal_surp_files[0], "r") as h5file:
    fig = plt.figure(figsize=(6.5, 6), constrained_layout=True)
    gs_root = gridspec.GridSpec(nrows=1, ncols=2, figure=fig, width_ratios=[3, 2])

    gs_state = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_root[0])
    axs_state = [fig.add_subplot(gs_state[0, :])]
    for idx in range(1, 3):
        axs_state.append(fig.add_subplot(gs_state[idx, :], sharex=axs_state[0]))
    for idx in range(3 - 1):
        plt.setp(axs_state[idx].get_xticklabels(), visible=False)

    gs_params = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs_root[1])
    axs_params = [fig.add_subplot(gs_params[0, :])]
    for idx in range(1, 5):
        axs_params.append(fig.add_subplot(gs_params[idx, :], sharex=axs_params[0]))
    for idx in range(5 - 1):
        plt.setp(axs_params[idx].get_xticklabels(), visible=False)

    plural = {"wolf": "wolves", "sheep": "sheep", "grass": "grass"}

    for idx, state_var_name in enumerate(["wolf", "sheep", "grass"]):
        axs_state[idx].set_title(state_var_name, loc="left")
        axs_state[idx].set_xlabel("time")
        axs_state[idx].set_ylabel("count")

        (true_value,) = axs_state[idx].plot(
            transform_kf_to_intrinsic(
                h5file["virtual_patient_trajectory"][:, idx], index=idx
            ),
            label="true value",
            # linestyle=":",
            color="black",
        )

        mu = h5file["means"][-1, :, idx]
        sigma = np.sqrt(h5file["covs"][-1, :, idx, idx])

        (prediction_center_line,) = axs_state[idx].plot(
            transform_kf_to_intrinsic(mu, index=idx),
            label="prediction",
            color="blue",
        )
        prediction_range = axs_state[idx].fill_between(
            range(len(mu)),
            transform_kf_to_intrinsic(mu - sigma, index=idx),
            transform_kf_to_intrinsic(mu + sigma, index=idx),
            color="blue",
            alpha=0.35,
        )

        ymin = max(0.0, axs_state[idx].get_ylim()[0])
        ymax = min(
            axs_state[idx].get_ylim()[1],
            1.2
            * np.max(
                transform_kf_to_intrinsic(
                    h5file["virtual_patient_trajectory"][:, idx], index=idx
                )
            ),
        )
        axs_state[idx].set_ylim(bottom=ymin, top=ymax)

    params = [
        "wolf gain from food",
        "sheep gain from food",
        "wolf reproduce",
        "sheep reproduce",
        "grass regrowth time",
    ]
    for idx, param_name in enumerate(params):
        axs_params[idx].set_title(param_name, loc="left")
        axs_params[idx].set_xlabel("time")

        axs_params[idx].plot(
            h5file["virtual_patient_trajectory"][:, 3 + idx],
            label="true value",
            color="black",
            linestyle=":",
        )

        mu = h5file["means"][-1, :, 3 + idx]
        sigma = np.sqrt(h5file["covs"][-1, :, 3 + idx, 3 + idx])

        axs_params[idx].plot(
            transform_kf_to_intrinsic(mu, index=3 + idx),
            color="blue",
            label="prediction",
        )
        TIME_SPAN_p1 = h5file["virtual_patient_trajectory"].shape[0]
        axs_params[idx].fill_between(
            range(TIME_SPAN_p1),
            np.maximum(
                0.0,
                transform_kf_to_intrinsic(
                    mu - sigma,
                    index=3 + idx,
                ),
            ),
            transform_kf_to_intrinsic(
                mu + sigma,
                index=3 + idx,
            ),
            color="blue",
            alpha=0.35,
            label="prediction cone",
        )

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
        loc="outside lower center",
    )
    # fig.suptitle("State Prediction")
    fig.savefig("typical-sheep-state.pdf")
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
    fig = plt.figure(figsize=(6.5, 6), constrained_layout=True)
    gs_root = gridspec.GridSpec(nrows=1, ncols=2, figure=fig, width_ratios=[3, 2])

    gs_state = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_root[0])
    axs_state = [fig.add_subplot(gs_state[0, :])]
    for idx in range(1, 3):
        axs_state.append(fig.add_subplot(gs_state[idx, :], sharex=axs_state[0]))
    for idx in range(3 - 1):
        plt.setp(axs_state[idx].get_xticklabels(), visible=False)

    gs_params = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs_root[1])
    axs_params = [fig.add_subplot(gs_params[0, :])]
    for idx in range(1, 5):
        axs_params.append(fig.add_subplot(gs_params[idx, :], sharex=axs_params[0]))
    for idx in range(5 - 1):
        plt.setp(axs_params[idx].get_xticklabels(), visible=False)

    plural = {"wolf": "wolves", "sheep": "sheep", "grass": "grass"}

    for idx, state_var_name in enumerate(["wolf", "sheep", "grass"]):
        axs_state[idx].set_title(state_var_name, loc="left")
        axs_state[idx].set_xlabel("time")
        axs_state[idx].set_ylabel("count")

        (true_value,) = axs_state[idx].plot(
            transform_kf_to_intrinsic(
                h5file["virtual_patient_trajectory"][:, idx], index=idx
            ),
            label="true value",
            # linestyle=":",
            color="black",
        )

        mu = h5file["means"][-1, :, idx]
        sigma = np.sqrt(h5file["covs"][-1, :, idx, idx])

        (prediction_center_line,) = axs_state[idx].plot(
            transform_kf_to_intrinsic(mu, index=idx),
            label="prediction",
            color="blue",
        )
        prediction_range = axs_state[idx].fill_between(
            range(len(mu)),
            transform_kf_to_intrinsic(mu - sigma, index=idx),
            transform_kf_to_intrinsic(mu + sigma, index=idx),
            color="blue",
            alpha=0.35,
        )

        ymin = max(0.0, axs_state[idx].get_ylim()[0])
        ymax = min(
            axs_state[idx].get_ylim()[1],
            1.2
            * np.max(
                transform_kf_to_intrinsic(
                    h5file["virtual_patient_trajectory"][:, idx], index=idx
                )
            ),
        )
        axs_state[idx].set_ylim(bottom=ymin, top=ymax)

    params = [
        "wolf gain from food",
        "sheep gain from food",
        "wolf reproduce",
        "sheep reproduce",
        "grass regrowth time",
    ]
    for idx, param_name in enumerate(params):
        axs_params[idx].set_title(param_name, loc="left")
        axs_params[idx].set_xlabel("time")

        axs_params[idx].plot(
            h5file["virtual_patient_trajectory"][:, 3 + idx],
            label="true value",
            color="black",
            linestyle=":",
        )

        mu = h5file["means"][-1, :, 3 + idx]
        sigma = np.sqrt(h5file["covs"][-1, :, 3 + idx, 3 + idx])

        axs_params[idx].plot(
            transform_kf_to_intrinsic(mu, index=3 + idx),
            color="blue",
            label="prediction",
        )
        TIME_SPAN_p1 = h5file["virtual_patient_trajectory"].shape[0]
        axs_params[idx].fill_between(
            range(TIME_SPAN_p1),
            np.maximum(
                0.0,
                transform_kf_to_intrinsic(
                    mu - sigma,
                    index=3 + idx,
                ),
            ),
            transform_kf_to_intrinsic(
                mu + sigma,
                index=3 + idx,
            ),
            color="blue",
            alpha=0.35,
            label="prediction cone",
        )

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
        loc="outside lower center",
    )
    # fig.suptitle("State Prediction")
    fig.savefig("typical-grass-state.pdf")
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
