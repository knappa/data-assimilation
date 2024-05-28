#!/usr/bin/env python3
# coding: utf-8

# # An-Cockrell model reimplementation

import h5py
import matplotlib.pyplot as plt
import numpy as np

from transform import transform_intrinsic_to_kf

GRAPHS = False

# must match an-cockrell-sensitivity-runner.py
variational_params = [
    "init_inoculum",
    "init_dcs",
    "init_nks",
    "init_macros",
    "macro_phago_recovery",
    "macro_phago_limit",
    "inflammasome_activation_threshold",
    "inflammasome_priming_threshold",
    "viral_carrying_capacity",
    "susceptibility_to_infection",
    "human_endo_activation",
    "human_metabolic_byproduct",
    "viral_incubation_threshold",
    "epi_apoptosis_threshold_lower",
    "epi_apoptosis_threshold_range",
    "epi_apoptosis_threshold_lower_regrow",
    "epi_apoptosis_threshold_range_regrow",
    "epi_regrowth_counter_threshold",
    "epi_cell_membrane_init_lower",
    "epi_cell_membrane_init_range",
    "infected_epithelium_ros_damage_counter_threshold",
    "epithelium_ros_damage_counter_threshold",
    "epithelium_pdamps_secretion_on_death",
    "dead_epithelium_pdamps_burst_secretion",
    "dead_epithelium_pdamps_secretion",
    "epi_max_tnf_uptake",
    "epi_max_il1_uptake",
    "epi_t1ifn_secretion",
    "epi_t1ifn_secretion_prob",
    "epi_pdamps_secretion_prob",
    "infected_epi_t1ifn_secretion",
    "infected_epi_il18_secretion",
    "infected_epi_il6_secretion",
    "activated_endo_death_threshold",
    "activated_endo_adhesion_threshold",
    "activated_endo_pmn_spawn_prob",
    "activated_endo_pmn_spawn_dist",
    "extracellular_virus_init_amount_lower",
    "extracellular_virus_init_amount_range",
    "human_t1ifn_effect_scale",
    "pmn_max_age",
    "pmn_ros_secretion_on_death",
    "pmn_il1_secretion_on_death",
    "nk_ifng_secretion",
    "macro_max_virus_uptake",
    "macro_activation_threshold",
    "activated_macro_il8_secretion",
    "activated_macro_il12_secretion",
    "activated_macro_tnf_secretion",
    "activated_macro_il6_secretion",
    "activated_macro_il10_secretion",
    "macro_antiactivation_threshold",
    "antiactivated_macro_il10_secretion",
    "inflammasome_il1_secretion",
    "inflammasome_macro_pre_il1_secretion",
    "inflammasome_il18_secretion",
    "inflammasome_macro_pre_il18_secretion",
    "pyroptosis_macro_pdamps_secretion",
    "dc_t1ifn_activation_threshold",
    "dc_il12_secretion",
    "dc_ifng_secretion",
    "dc_il6_secretion",
    "dc_il6_max_uptake",
    # # ACK's Executive Judgement: These are physics-like parameters and won't vary between individuals.
    # "human_viral_lower_bound", 0.0
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
    s = s.split("_")
    lines = []
    line = ""
    for word in s:
        if len(s) + len(line) + 1 > break_len:
            lines.append(line)
            line = word
        else:
            line = line + " " + word
    if len(line) > 0:
        lines.append(line)
    return "\n".join(lines)


variational_params_aug = np.array(variational_params + ["bias"])
varp_idcs = {str(vp): i for i, vp in enumerate(variational_params_aug)}

with h5py.File("simulation-statistics.hdf5", "r") as h5file:
    param_list = h5file["param_list"][()]
    system_health = h5file["system_health"][()]
    key_to_idx = {
        k: i for i, k in enumerate([k for k in h5file.keys() if k != "param_list"])
    }
    full_data = np.zeros((len(key_to_idx), *system_health.shape), dtype=np.float64)
    for k, i in key_to_idx.items():
        full_data[i, :] = h5file[k][()]
    full_data = np.transpose(full_data, axes=[1, 0, 2])
    # total_P_DAMPS = h5file["total_P_DAMPS"][()]
    # total_T1IFN = h5file["total_T1IFN"][()]
    # total_TNF = h5file["total_TNF"][()]
    # total_IFNg = h5file["total_IFNg"][()]
    # total_IL6 = h5file["total_IL6"][()]
    # total_IL1 = h5file["total_IL1"][()]
    # total_IL8 = h5file["total_IL8"][()]
    # total_IL10 = h5file["total_IL10"][()]
    # total_IL12 = h5file["total_IL12"][()]
    # total_IL18 = h5file["total_IL18"][()]
    # total_extracellular_virus = h5file["total_extracellular_virus"][()]
    # total_intracellular_virus = h5file["total_intracellular_virus"][()]
    # apoptosis_eaten_counter = h5file["apoptosis_eaten_counter"][()]
    # infected_epis = h5file["infected_epis"][()]
    # dead_epis = h5file["dead_epis"][()]
    # apoptosed_epis = h5file["apoptosed_epis"][()]

min_sys_health = np.min(system_health, axis=1)

param_list_aug = np.vstack([param_list.T, np.ones(len(param_list))[np.newaxis]])

least_squares_sol, residuals, rank, sing_vals = np.linalg.lstsq(
    param_list_aug.T, min_sys_health, rcond=None
)

sensitivity_dict = dict(zip(variational_params_aug, least_squares_sol))
print(sensitivity_dict)

sort_indices = np.argsort(np.abs(least_squares_sol))
print(variational_params_aug[sort_indices[:10]])
if GRAPHS:
    plt.plot(np.abs(least_squares_sol[sort_indices]))

log_min_sys_health = transform_intrinsic_to_kf(min_sys_health)
log_param_list = np.log(param_list)
log_param_list_aug = np.vstack(
    [log_param_list.T, np.ones(len(log_param_list))[np.newaxis]]
)

log_least_squares_sol, log_residuals, log_rank, log_sing_vals = np.linalg.lstsq(
    log_param_list_aug.T, log_min_sys_health, rcond=None
)

log_sort_indices = np.argsort(np.abs(log_least_squares_sol))
print(variational_params_aug[log_sort_indices[:10]])
if GRAPHS:
    plt.plot(np.abs(log_least_squares_sol[log_sort_indices]))

log_sensitivity_dict = dict(
    zip(
        variational_params_aug,
        log_least_squares_sol,
    )
)
print(log_sensitivity_dict)


if GRAPHS:
    fig, ax = plt.subplots()
    ax.scatter(np.abs(least_squares_sol)[:-1], np.abs(log_least_squares_sol)[:-1])
    plt.xlabel("abs sensitivity")
    plt.ylabel("abs log sensitivity")
    for i, txt in enumerate(variational_params):
        ax.annotate(
            txt, (np.abs(least_squares_sol)[i], np.abs(log_least_squares_sol)[i])
        )

foo = np.sqrt(least_squares_sol**2 + log_least_squares_sol**2)
foo_sort_indices = np.argsort(np.abs(log_least_squares_sol))

################################################################################

if GRAPHS:
    plt.hist(system_health.reshape(-1), bins=100, density=True)

# compute sensitivity matrix
system_health_sensitivity, sh_residuals, sh_rank, sh_sing_vals = np.linalg.lstsq(
    param_list_aug.T, system_health, rcond=None
)

system_health_sensitivity_column_norm = np.linalg.norm(
    system_health_sensitivity, axis=1
)

# compute log sensitivity matrix
log_system_health = np.log(np.maximum(np.exp(-25), system_health))
(
    system_health_log_sensitivity,
    log_sh_residuals,
    log_sh_rank,
    log_sh_sing_vals,
) = np.linalg.lstsq(log_param_list_aug.T, log_system_health, rcond=None)
system_health_log_sensitivity_column_norm = np.linalg.norm(
    system_health_log_sensitivity, axis=1
)

if GRAPHS:
    # matrix plot of sensitivity matrix, in chunks
    fig, axs = plt.subplots(8, figsize=(8, 20))
    for plt_idx in range(8):
        axs[plt_idx].imshow(
            np.abs(system_health_sensitivity[:, plt_idx * 252 : (plt_idx + 1) * 252]),
            aspect=1,
        )
    fig.tight_layout()

if GRAPHS:
    # matrix plot of sensitivity matrix; relative time contribution, in chunks
    fig, axs = plt.subplots(8, figsize=(8, 20))
    for plt_idx in range(8):
        axs[plt_idx].imshow(
            np.abs(system_health_sensitivity[:, plt_idx * 252 : (plt_idx + 1) * 252])
            / system_health_sensitivity_column_norm[:, np.newaxis],
            aspect=1,
        )
    fig.tight_layout()

if GRAPHS:
    # matrix plot of log sensitivity matrix, in chunks
    fig, axs = plt.subplots(8, figsize=(8, 20))
    for plt_idx in range(8):
        axs[plt_idx].imshow(
            np.abs(
                system_health_log_sensitivity[:, plt_idx * 252 : (plt_idx + 1) * 252]
            ),
            aspect=1,
        )
    fig.tight_layout()

if GRAPHS:
    # matrix plot of log sensitivity matrix; relative time contribution, in chunks
    fig, axs = plt.subplots(8, figsize=(8, 20))
    for plt_idx in range(8):
        axs[plt_idx].imshow(
            np.abs(
                system_health_log_sensitivity[:, plt_idx * 252 : (plt_idx + 1) * 252]
            )
            / system_health_log_sensitivity_column_norm[:, np.newaxis],
            aspect=1,
        )
    fig.tight_layout()

if GRAPHS:
    # plot log sensitivity of components that are ever above min_sensitivity
    min_sensitivity = 0.25
    max_log_sensitivity = np.max(np.abs(system_health_log_sensitivity), axis=1)
    plt.figure()
    plt.plot(
        system_health_log_sensitivity.T[:, max_log_sensitivity > min_sensitivity][
            :, :-1
        ],
        label=variational_params_aug[max_log_sensitivity > min_sensitivity][:-1],
    )
    plt.legend()
    plt.tight_layout()

# compute column-wise norms
norm_sensitivity = np.linalg.norm(np.abs(system_health_sensitivity), axis=1)
norm_log_sensitivity = np.linalg.norm(np.abs(system_health_log_sensitivity), axis=1)

norm_sensitivity_sort_indices = np.argsort(norm_sensitivity)
norm_log_sensitivity_sort_indices = np.argsort(norm_log_sensitivity)


# indices of the top 10 by column-wise norm
indices_top_ten = np.argsort(norm_sensitivity[:-1])[-10:]
indices_top_ten_log = np.argsort(norm_log_sensitivity[:-1])[-10:]

# collect names
names_top_ten = set(variational_params_aug[indices_top_ten])
names_top_ten_log = set(variational_params_aug[indices_top_ten_log])

if GRAPHS:
    # plot columns of log sensitivity matrix as time series, for top ten components
    plt.figure()
    plt.plot(
        system_health_log_sensitivity.T[:, indices_top_ten_log],
        label=variational_params_aug[indices_top_ten_log],
    )
    plt.legend()
    plt.tight_layout()

# how different are these?
print(set(names_top_ten).intersection(names_top_ten_log))
print(set(names_top_ten).union(names_top_ten_log))

if GRAPHS:
    # scatter
    fig, ax = plt.subplots()
    ax.scatter(norm_sensitivity[:-1], norm_log_sensitivity[:-1])
    plt.xlabel("sensitivity")
    plt.ylabel("log sensitivity")
    for i, txt in enumerate(variational_params):
        ax.annotate(txt, (norm_sensitivity[i], norm_log_sensitivity[i]))


################################################################################
# sensitivity by early/late infection

early_cutoff = 750

# compute log sensitivity matrix
early_system_health_log_sensitivity, _, _, _ = np.linalg.lstsq(
    log_param_list_aug.T, log_system_health[:, :early_cutoff], rcond=None
)
early_system_health_log_sensitivity_column_norm = np.linalg.norm(
    early_system_health_log_sensitivity, axis=1
)

if GRAPHS:
    fig = plt.figure()
    plt.plot(early_system_health_log_sensitivity.T[:, :-1])

early_indices_top_ten_log = np.argsort(
    early_system_health_log_sensitivity_column_norm[:-1]
)[-10:]

################################################################################

full_data_log_sensitivity, _, _, _ = np.linalg.lstsq(
    log_param_list_aug.T,
    np.log(np.maximum(np.exp(-25), full_data.reshape((full_data.shape[0], -1)))),
    rcond=None,
)
full_data_log_sensitivity_column_norm = np.linalg.norm(
    full_data_log_sensitivity, axis=1
)
indices_full = np.argsort(full_data_log_sensitivity_column_norm[:-1])[-10:]
names_full = set(variational_params_aug[indices_full])

if GRAPHS:
    fig, axs = plt.subplots(5, 4)
    for idx, name in enumerate(key_to_idx.keys()):
        row, col = divmod(idx, 4)
        axs[row, col].plot(
            full_data_log_sensitivity.reshape(
                full_data_log_sensitivity.shape[0], *full_data.shape[1:]
            )[:, idx, :].T,
            label=name,
        )
        axs[row, col].set_title(name)
    fig.tight_layout()

if GRAPHS:
    fig, axs = plt.subplots(4, 5, sharex=True, layout="constrained", figsize=(8, 4))
    lines = None
    for idx, name in enumerate(key_to_idx.keys()):
        row, col = divmod(idx, 5)
        if idx >= 16:
            row, col = divmod(idx + 3, 5)
        lines = axs[row, col].plot(
            full_data_log_sensitivity.reshape(
                full_data_log_sensitivity.shape[0], *full_data.shape[1:]
            )[indices_full[::-1], idx, :].T,
            label=variational_params_aug[indices_full[::-1]],
        )
        axs[row, col].set_title(fix_title(name))
    for idx in [16, 17, 18]:
        row, col = divmod(idx, 5)
        axs[row, col].set_axis_off()
    fig.legend(
        lines,
        variational_params_aug[indices_full][::-1],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
    )

full_data_sort_indices = np.argsort(full_data_log_sensitivity_column_norm)


################################################################################
# print table for inclusion in supplementary material

# log(min-system-health), |log(system-health)|, log(full)

for idx in range(len(variational_params_aug)):
    # noinspection PyUnresolvedReferences
    name = variational_params_aug[idx].replace("_", "\\_")
    if (
        idx in log_sort_indices[-11:]
        or idx in norm_log_sensitivity_sort_indices[-11:]
        or idx in full_data_sort_indices[-11:]
    ):
        name += "\\textsuperscript{"
        if idx in log_sort_indices[-11:]:
            name += "$\star$"
        if idx in norm_log_sensitivity_sort_indices[-11:]:
            name += "$\dagger$"
        if idx in full_data_sort_indices[-11:]:
            name += "$\ddagger$"
        name += "}"
    print(
        f"{name} & ${log_least_squares_sol[idx]:.4g}$ & ${norm_log_sensitivity[idx]:.4g}$ & ${full_data_log_sensitivity_column_norm[idx]:.4g}$ \\\\ \\hline"
    )
