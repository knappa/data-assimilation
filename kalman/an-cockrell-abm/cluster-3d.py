# coding: utf-8
import h5py
import matplotlib.pyplot as plt
import numpy as np

colors = ["#004488", "#ddaa33", "#bb5566"]
markers = [".", "1", "+"]

with h5py.File("virt_pop.hdf5", "r") as h5file:
    data = np.log(1e-3 + h5file["virt_pop"][()])

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(data.reshape(-1, 2017 * 41))
t_data = pca.transform(data.reshape(-1, 2017 * 41))

# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=5).fit(data.reshape(-1,2017*41))

from sklearn.mixture import GaussianMixture

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

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
for c_idx in range(3):
    ax.scatter(
        *t_data[labels == c_idx, :].T,
        c=colors[c_idx],
        marker=markers[c_idx],
        label=f"Cluster {c_idx}",
    )
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
for line in ax.xaxis.get_ticklines():
    line.set_visible(False)
for line in ax.yaxis.get_ticklines():
    line.set_visible(False)
for line in ax.zaxis.get_ticklines():
    line.set_visible(False)
fig.legend()
fig.tight_layout()
plt.show()


################################################################################

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
