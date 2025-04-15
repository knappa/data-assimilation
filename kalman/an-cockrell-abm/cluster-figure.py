# coding: utf-8
import itertools

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance_nd
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC

colors = [
    "#004488",  # blue
    "#ddaa33",  # yellow
    "#bb5566",  # red
]
colorsmix = [
    "#228833",  # 0,1 blue/yellow -> green
    "#aa3377",  # 0,2 blue/red -> purple
    "#66ccee",  # 1,2 yellow/red -> cyan...? orange? ee7733
]
markers = [".", "1", "+"]

with h5py.File("virt_pop.hdf5", "r") as h5file:
    data = np.log(1e-3 + h5file["virt_pop"][()])


pca_traj_3d = PCA(n_components=3)
pca_traj_3d.fit(data.reshape(-1, 2017 * 41))
t_data = pca_traj_3d.transform(data.reshape(-1, 2017 * 41))


gm = GaussianMixture(
    n_components=4,
    means_init=np.array(
        [
            [57.74094909, -128.1007097, 167.05671305],
            [-477.80472188, 96.23611442, -102.65933337],
            [769.88102858, -219.00299126, -229.74449402],
            [1074.40147663, 1240.15156448, 114.09776011],
        ]
    ),
)
labels = gm.fit_predict(t_data)
# collapse two groups
labels = 0 * (labels == 0) + 0 * (labels == 1) + 1 * (labels == 2) + 2 * (labels == 3)


################################################################################
# low-dimensional data for parameters

# 3d pca, but we throw away the 1st coord
param_pca = PCA(n_components=3)
param_pca.fit(data[:, :, 22:].reshape((data.shape[0], -1)))

t_param_data = param_pca.transform(data[:, :, 22:].reshape((data.shape[0], -1)))

lsvc = LinearSVC(dual="auto", class_weight="balanced")
lsvc.fit(t_param_data[labels != 2, 1:], labels[labels != 2])

################################################################################
# plotting setup

fig = plt.figure(figsize=plt.figaspect(1 / 2))

ax3d = fig.add_subplot(1, 2, 1, projection="3d")
ax = fig.add_subplot(1, 2, 2)

################################################################################
# 3d plot

ax3d.set_title("Trajectory clustering")

for c_idx in range(3):
    ax3d.scatter(
        *t_data[labels == c_idx, :].T,
        c=colors[c_idx],
        marker=markers[c_idx],
        label=f"Cluster {c_idx}",
    )
ax3d.xaxis.set_ticklabels([])
ax3d.yaxis.set_ticklabels([])
ax3d.zaxis.set_ticklabels([])
for line in ax3d.xaxis.get_ticklines():
    line.set_visible(False)
for line in ax3d.yaxis.get_ticklines():
    line.set_visible(False)
for line in ax3d.zaxis.get_ticklines():
    line.set_visible(False)

################################################################################
# 2d plot

ax.set_title("Parameters of Clusters")
ax.set_axis_off()

for c_idx, mkr in enumerate(markers):
    ax.scatter(
        *t_param_data[labels == c_idx, 1:].T,
        marker=mkr,
        s=16,
        c=colors[c_idx],
        # label=f"Cluster {c_idx}",
    )

DecisionBoundaryDisplay.from_estimator(
    lsvc,
    t_param_data[:, 1:],
    ax=ax,
    grid_resolution=50,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
)

fig.tight_layout()
fig.legend(loc="lower right")
plt.savefig("clustering-low-dim.pdf")
plt.show()


################################################################################
# plot for mean cluster trajectories in components

state_vars = [
    "total_P_DAMPS",
    "total_T1IFN",
    "total_TNF",
    "total_IFNg",
    "total_IL6",
    "total_IL1",
    "total_IL8",
    "total_IL10",
    "total_IL12",
    "total_IL18",
    "total_extracellular_virus",
    "total_intracellular_virus",
    "apoptosis_eaten_counter",
    "empty_epithelium_count",
    "healthy_epithelium_count",
    "infected_epithelium_count",
    "dead_epithelium_count",
    "apoptosed_epithelium_count",
    # "system_health",
    "dc_count",
    "nk_count",
    "pmn_count",
    "macro_count",
]
variational_params = [
    # INSENSITIVE "macro_phago_recovery",
    # INSENSITIVE "macro_phago_limit",
    "inflammasome_activation_threshold",
    # INSENSITIVE "inflammasome_priming_threshold",
    # INSENSITIVE "viral_carrying_capacity",
    # INSENSITIVE "susceptibility_to_infection",
    "human_endo_activation",
    "human_metabolic_byproduct",
    "resistance_to_infection",
    "viral_incubation_threshold",
    "epi_apoptosis_threshold_lower",
    # INSENSITIVE "epi_apoptosis_threshold_range",
    # INSENSITIVE "epi_apoptosis_threshold_lower_regrow",
    # INSENSITIVE "epi_apoptosis_threshold_range_regrow",
    # INSENSITIVE "epi_regrowth_counter_threshold",
    # INSENSITIVE "epi_cell_membrane_init_lower",
    # INSENSITIVE "epi_cell_membrane_init_range",
    # INSENSITIVE "infected_epithelium_ros_damage_counter_threshold",
    "epithelium_ros_damage_counter_threshold",
    "epithelium_pdamps_secretion_on_death",
    # INSENSITIVE "dead_epithelium_pdamps_burst_secretion",
    # INSENSITIVE "dead_epithelium_pdamps_secretion",
    # INSENSITIVE "epi_max_tnf_uptake",
    # INSENSITIVE "epi_max_il1_uptake",
    # INSENSITIVE "epi_t1ifn_secretion",
    # INSENSITIVE "epi_t1ifn_secretion_prob",
    # INSENSITIVE "epi_pdamps_secretion_prob",
    "infected_epi_t1ifn_secretion",
    # INSENSITIVE "infected_epi_il18_secretion",
    # INSENSITIVE "infected_epi_il6_secretion",
    "activated_endo_death_threshold",
    # INSENSITIVE "activated_endo_adhesion_threshold",
    "activated_endo_pmn_spawn_prob",
    "activated_endo_pmn_spawn_dist",
    "extracellular_virus_init_amount_lower",
    # INSENSITIVE "extracellular_virus_init_amount_range",
    # INSENSITIVE "human_t1ifn_effect_scale",
    # INSENSITIVE "pmn_max_age",
    "pmn_ros_secretion_on_death",
    "pmn_il1_secretion_on_death",
    # INSENSITIVE "nk_ifng_secretion",
    # INSENSITIVE "macro_max_virus_uptake",
    "macro_activation_threshold",
    # INSENSITIVE "macro_antiactivation_threshold",
    # INSENSITIVE "activated_macro_il8_secretion",
    # INSENSITIVE "activated_macro_il12_secretion",
    "activated_macro_tnf_secretion",
    # INSENSITIVE "activated_macro_il6_secretion",
    # INSENSITIVE "activated_macro_il10_secretion",
    # INSENSITIVE "antiactivated_macro_il10_secretion",
    "inflammasome_il1_secretion",
    "inflammasome_macro_pre_il1_secretion",
    # INSENSITIVE "inflammasome_il18_secretion",
    # INSENSITIVE "inflammasome_macro_pre_il18_secretion",
    # INSENSITIVE "pyroptosis_macro_pdamps_secretion",
    # INSENSITIVE "dc_t1ifn_activation_threshold",
    # INSENSITIVE "dc_il12_secretion",
    # INSENSITIVE "dc_ifng_secretion",
    # INSENSITIVE "dc_il6_secretion",
    # INSENSITIVE "dc_il6_max_uptake",
    # # ACK's Executive Judgement: These are physics-like parameters and won't vary between individuals.
    # # They also include model-intrinsic things like a cleanup thresholds which don't precisely
    # # correspond to read world objects.
    # "human_viral_lower_bound", # 0.0
    # "extracellular_virus_diffusion_const",
    # "T1IFN_diffusion_const",
    # "PAF_diffusion_const",
    # "ROS_diffusion_const",
    # "P_DAMPS_diffusion_const",
    # "IFNg_diffusion_const",
    # "TNF_diffusion_const",
    # "IL6_diffusion_const",
    # "IL1_diffusion_const",
    # "IL10_diffusion_const",
    # "IL12_diffusion_const",
    # "IL18_diffusion_const",
    # "IL8_diffusion_const",
    # "extracellular_virus_cleanup_threshold",
    # "cleanup_threshold",
    # "evap_const_1",
    # "evap_const_2",
]


def fix_title(s: str, *, break_len=14):
    """
    Fix variable name titles.

    :param s: a title with _'s and maybe too long
    :param break_len: where to look for line breaks
    :return: a title without _'s and with \n's in reasonable places
    """
    s = s.replace("_", " ")
    if len(s) > 1.5 * break_len:
        idx = s[break_len:].find(" ")
        if idx >= 0:
            idx += break_len
        else:
            idx = s.find(" ")
        if idx != -1:
            s = s[:idx] + "\n" + s[idx + 1 :]
    return s


mean_c0 = np.mean(data[labels == 0, :, :], axis=0)
mean_c1 = np.mean(data[labels == 1, :, :], axis=0)
mean_c2 = np.mean(data[labels == 2, :, :], axis=0)

state_var_graphs_cols = 5
state_var_graphs_rows = int(np.ceil(len(state_vars) / state_var_graphs_cols))
state_var_graphs_figsize = (
    1.8 * state_var_graphs_rows,
    1.8 * state_var_graphs_cols,
)

fig, axs = plt.subplots(
    nrows=state_var_graphs_rows,
    ncols=state_var_graphs_cols,
    figsize=state_var_graphs_figsize,
    sharex=True,
    sharey=False,
)
for idx, state_var_name in enumerate(state_vars):
    row, col = divmod(idx, state_var_graphs_cols)
    axs[row, col].plot(np.exp(mean_c0[:, idx]) - 1e-3, color=colors[0])
    axs[row, col].plot(np.exp(mean_c1[:, idx]) - 1e-3, color=colors[1])
    axs[row, col].plot(np.exp(mean_c2[:, idx]) - 1e-3, color=colors[2])
    axs[row, col].set_title(
        fix_title(state_var_name),
        loc="center",
        wrap=True,
    )
    # axs[row, col].set_ylim(bottom=max(0.0, axs[row, col].get_ylim()[0]))
for idx in range(len(state_vars), state_var_graphs_rows * state_var_graphs_cols):
    row, col = divmod(idx, state_var_graphs_cols)
    axs[row, col].set_axis_off()
fig.tight_layout()
plt.show()


fig, axs = plt.subplots(
    nrows=state_var_graphs_rows,
    ncols=state_var_graphs_cols,
    figsize=state_var_graphs_figsize,
    sharex=True,
    sharey=False,
)
for idx, state_var_name in enumerate(state_vars):
    row, col = divmod(idx, state_var_graphs_cols)
    axs[row, col].plot(mean_c0[:, idx], color=colors[0])
    axs[row, col].plot(mean_c1[:, idx], color=colors[1])
    axs[row, col].plot(mean_c2[:, idx], color=colors[2])
    axs[row, col].set_title(
        fix_title(state_var_name),
        loc="center",
        wrap=True,
    )
    # axs[row, col].set_ylim(bottom=max(0.0, axs[row, col].get_ylim()[0]))
for idx in range(len(state_vars), state_var_graphs_rows * state_var_graphs_cols):
    row, col = divmod(idx, state_var_graphs_cols)
    axs[row, col].set_axis_off()
fig.tight_layout()
plt.show()


################################################################################
# plot for Wasserstein distances between clusters


# time-parameterized distances
mus = np.zeros((3, 2017, 41))
covs = np.zeros((3, 2017, 41, 41))
dists = np.zeros((3, 2017))
for t_idx in range(2017):
    for l_idx in range(3):
        mus[l_idx, t_idx, :] = np.mean(data[labels == l_idx, t_idx, :], axis=0)
        covs[l_idx, t_idx, :, :] = np.cov(data[labels == l_idx, t_idx, :], rowvar=False)

    for k, (i, j) in enumerate(itertools.combinations(range(3), 2)):
        m1 = mus[i, t_idx, :]
        m2 = mus[j, t_idx, :]
        C1 = covs[i, t_idx, :, :]
        C2 = covs[j, t_idx, :, :]

        dists[k, t_idx] = np.sqrt(
            (m1 - m2) @ (m1 - m2).T + np.trace(C1 + C2 - 2 * sqrtm(sqrtm(C2) @ C1 @ sqrtm(C2)))
        )


fig = plt.figure()
ax = fig.add_subplot()
for k, (i, j) in enumerate(itertools.combinations(range(3), 2)):
    # ax.plot(
    #     dists[k, :],
    #     label=f"Distance between clusters {i} and {j}",
    #     c=colors[i],
    #     linestyle="-",
    #     dashes=(5, 5),
    #     gapcolor=colors[j]
    # )
    ax.plot(
        dists[k, :],
        label=f"Distance between clusters {i} and {j}",
        c=colorsmix[k],
    )

ax.set_title("Wasserstein distance")
ax.legend(loc="best")
fig.tight_layout()
plt.savefig("cluster-wasserstein-dist.pdf")
plt.show()


################################################################################
# combo figure for wasserstein dist and selected state components


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


fig = plt.figure(figsize=(5, 5), constrained_layout=True)
subfigs = fig.subfigures(2, 1, wspace=0.07, height_ratios=[1, 1.5])

ax_wd = subfigs[0].subplots(1, 1)

axs = subfigs[1].subplots(2, 3, sharex=True)
(ax_tnf, ax_t1ifn, ax_il12), (ax_exvir, ax_hepi, blank) = axs


blank.set_axis_off()

# wasserstein-dist
for k, (i, j) in enumerate(itertools.combinations(range(3), 2)):
    ax_wd.plot(
        dists[k, :],
        label=f"Dist. clusters {i} and {j}",
        c=colorsmix[k],
    )

ax_wd.set_title("Wasserstein distance")
ax_wd.legend(loc="best")


# log tnf
state_var_name = "total_TNF"
idx = state_vars.index(state_var_name)
ax_tnf.plot(mean_c0[:, idx], color=colors[0])
ax_tnf.plot(mean_c1[:, idx], color=colors[1])
ax_tnf.plot(mean_c2[:, idx], color=colors[2])
ax_tnf.set_title(
    "$\log$ TNF",
    loc="center",
    wrap=True,
)


# t1ifn
state_var_name = "total_T1IFN"
idx = state_vars.index(state_var_name)
ax_t1ifn.plot(np.exp(mean_c0[:, idx]) - 1e-3, color=colors[0])
ax_t1ifn.plot(np.exp(mean_c1[:, idx]) - 1e-3, color=colors[1])
ax_t1ifn.plot(np.exp(mean_c2[:, idx]) - 1e-3, color=colors[2])
ax_t1ifn.set_title(
    "T1IFN",
    loc="center",
    wrap=True,
)


# il12
state_var_name = "total_IL12"
idx = state_vars.index(state_var_name)
ax_il12.plot(np.exp(mean_c0[:, idx]) - 1e-3, color=colors[0])
ax_il12.plot(np.exp(mean_c1[:, idx]) - 1e-3, color=colors[1])
ax_il12.plot(np.exp(mean_c2[:, idx]) - 1e-3, color=colors[2])
ax_il12.set_title(
    "IL12",
    loc="center",
    wrap=True,
)


# exvir
state_var_name = "total_extracellular_virus"
idx = state_vars.index(state_var_name)
ax_exvir.plot(mean_c0[:, idx], color=colors[0])
ax_exvir.plot(mean_c1[:, idx], color=colors[1])
ax_exvir.plot(mean_c2[:, idx], color=colors[2])
ax_exvir.set_title(
    "$\log$ extracellular virus",
    loc="center",
    wrap=True,
)
ax_exvir.set_xticks(
    np.arange(0, 2017, 250),
    [str(t) if k % 3 == 0 else "" for k, t in enumerate(np.arange(0, 2017, 250))],
)


# hepi
state_var_name = "healthy_epithelium_count"
idx = state_vars.index(state_var_name)
line1 = ax_hepi.plot(np.exp(mean_c0[:, idx]) - 1e-3, color=colors[0])
line2 = ax_hepi.plot(np.exp(mean_c1[:, idx]) - 1e-3, color=colors[1])
line3 = ax_hepi.plot(np.exp(mean_c2[:, idx]) - 1e-3, color=colors[2])
ax_hepi.set_title(
    "healthy epithelium",
    loc="center",
    wrap=True,
)

blank.legend([line1[0], line2[0], line3[0]], [f"Cluster {i}" for i in range(3)], loc="upper left")

plt.show()
