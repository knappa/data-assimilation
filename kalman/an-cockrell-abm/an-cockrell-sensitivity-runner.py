#!/usr/bin/env python3
# coding: utf-8

# # An-Cockrell model reimplementation

import h5py
import matplotlib.pyplot as plt
import numpy as np

# from consts import variational_params

GRAPHS = False

# true parameters of the model
variational_params = [
    "macro_phago_recovery",
    "macro_phago_limit",
    "inflammasome_activation_threshold",
    "inflammasome_priming_threshold",
    "viral_carrying_capacity",
    "susceptibility_to_infection",
    "human_endo_activation",
    "human_metabolic_byproduct",
    "resistance_to_infection",  # NOT USED
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
    "macro_antiactivation_threshold",
    "activated_macro_il8_secretion",
    "activated_macro_il12_secretion",
    "activated_macro_tnf_secretion",
    "activated_macro_il6_secretion",
    "activated_macro_il10_secretion",
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

variational_params_filtered = [
    s for s in variational_params if s != "human_viral_lower_bound"
]
variational_params_aug = np.array(variational_params + ["bias"])
variational_params_filtered_aug = np.array(variational_params_filtered + ["bias"])
varp_idcs = {str(vp): i for i, vp in enumerate(variational_params_aug)}
varpf_idcs = {str(vp): i for i, vp in enumerate(variational_params_filtered_aug)}

# with h5py.File("run-statistics.hdf5", "r") as f:

f = h5py.File("run-statistics.hdf5", "r")

min_sys_health = np.min(f["system_health"][()], axis=1)
param_list_filtered = f["param_list"][()][
    :,
    [
        i
        for i in range(len(variational_params))
        if i != varp_idcs["human_viral_lower_bound"]
    ],
]

param_list_filtered_aug = np.vstack(
    [param_list_filtered.T, np.ones(len(param_list_filtered))[np.newaxis]]
)

least_squares_sol, residuals, rank, sing_vals = np.linalg.lstsq(
    param_list_filtered_aug.T, min_sys_health, rcond=None
)

sort_indices = np.argsort(np.abs(least_squares_sol))
sensitivity_dict = dict(
    zip(variational_params_filtered_aug[sort_indices], least_squares_sol[sort_indices])
)
print(sensitivity_dict)

if GRAPHS:
    plt.plot(np.abs(least_squares_sol[np.argsort(np.abs(least_squares_sol))]))

epsilon = np.exp(-25)
log_min_sys_health = np.log(np.maximum(epsilon, np.min(f["system_health"][()], axis=1)))

# log_param_list_filtered = log_param_list[
#     :, [i for i in range(len(variational_params)) if i != varp_idcs["human_viral_lower_bound"]]
# ]
log_param_list_filtered = np.log(
    f["param_list"][
        :,
        [
            i
            for i in range(len(variational_params))
            if i != varp_idcs["human_viral_lower_bound"]
        ],
    ]
)
log_param_list_filtered_aug = np.vstack(
    [log_param_list_filtered.T, np.ones(len(log_param_list_filtered))[np.newaxis]]
)

log_least_squares_sol, log_residuals, log_rank, log_sing_vals = np.linalg.lstsq(
    log_param_list_filtered_aug.T, log_min_sys_health, rcond=None
)

if GRAPHS:
    plt.plot(np.abs(least_squares_sol[np.argsort(np.abs(least_squares_sol))]))

log_sort_indices = np.argsort(np.abs(log_least_squares_sol))
log_sensitivity_dict = dict(
    zip(
        variational_params_filtered_aug[log_sort_indices],
        log_least_squares_sol[log_sort_indices],
    )
)
print(log_sensitivity_dict)

if GRAPHS:
    plt.plot(np.abs(least_squares_sol[np.argsort(np.abs(least_squares_sol))]))

if GRAPHS:
    fig, ax = plt.subplots()
    ax.scatter(np.abs(least_squares_sol)[:-1], np.abs(log_least_squares_sol)[:-1])
    plt.xlabel("abs sensitivity")
    plt.ylabel("abs log sensitivity")
    for i, txt in enumerate(variational_params_filtered):
        ax.annotate(
            txt, (np.abs(least_squares_sol)[i], np.abs(log_least_squares_sol)[i])
        )

################################################################################

system_health = f["system_health"][()]

if GRAPHS:
    plt.hist(system_health.reshape(-1), bins=100, density=True)

# compute sensitivity matrix
system_health_sensitivity, sh_residuals, sh_rank, sh_sing_vals = np.linalg.lstsq(
    param_list_filtered_aug.T, system_health, rcond=None
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
) = np.linalg.lstsq(log_param_list_filtered_aug.T, log_system_health, rcond=None)
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
        label=variational_params_filtered_aug[max_log_sensitivity > min_sensitivity][
            :-1
        ],
    )
    plt.legend()
    plt.tight_layout()

# compute column-wise norms
norm_sensitivity = np.linalg.norm(np.abs(system_health_sensitivity), axis=1)
norm_log_sensitivity = np.linalg.norm(np.abs(system_health_log_sensitivity), axis=1)

# indices of the top 10 by column-wise norm
indices_top_ten = np.argsort(norm_sensitivity[:-1])[-10:]
indices_top_ten_log = np.argsort(norm_log_sensitivity[:-1])[-10:]

# collect names
names_top_ten = set(variational_params_filtered_aug[indices_top_ten])
names_top_ten_log = set(variational_params_filtered_aug[indices_top_ten_log])

if GRAPHS:
    # plot columns of log sensitivity matrix as time series, for top ten components
    plt.figure()
    plt.plot(
        system_health_log_sensitivity.T[:, indices_top_ten_log],
        label=variational_params_filtered_aug[indices_top_ten_log],
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
    for i, txt in enumerate(variational_params_filtered):
        ax.annotate(txt, (norm_sensitivity[i], norm_log_sensitivity[i]))

################################################################################
# sensitivity by early/late infection

early_cutoff = 750

# compute log sensitivity matrix
early_system_health_log_sensitivity, _, _, _ = np.linalg.lstsq(
    log_param_list_filtered_aug.T, log_system_health[:, :early_cutoff], rcond=None
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

key_to_idx = {k: i for i, k in enumerate([k for k in f.keys() if k != "param_list"])}
full_data = np.zeros((len(key_to_idx), *f["system_health"][()].shape), dtype=np.float64)
for k, i in key_to_idx.items():
    full_data[i, :] = f[k][()]
full_data = np.transpose(full_data, axes=[1, 0, 2])

full_data_log_sensitivity, _, _, _ = np.linalg.lstsq(
    log_param_list_filtered_aug.T,
    np.log(np.maximum(np.exp(-25), full_data.reshape((full_data.shape[0], -1)))),
    rcond=None,
)
full_data_log_sensitivity_column_norm = np.linalg.norm(
    full_data_log_sensitivity, axis=1
)
indices_full = np.argsort(full_data_log_sensitivity_column_norm[:-1])[-10:]
names_full = set(variational_params_filtered_aug[indices_full])

if GRAPHS:
    fig, axs = plt.subplots(4, 4)
    for idx, name in enumerate([k for k in f.keys() if k != "param_list"]):
        row, col = divmod(idx, 4)
        axs[row, col].plot(
            full_data_log_sensitivity.reshape(
                full_data_log_sensitivity.shape[0], *full_data.shape[1:]
            )[:, idx, :].T
        )
        axs[row, col].set_title(name)
    fig.tight_layout()

if GRAPHS:
    fig, axs = plt.subplots(4, 4, sharex=True, layout="constrained")
    lines = None
    for idx, name in enumerate([k for k in f.keys() if k != "param_list"]):
        row, col = divmod(idx, 4)
        lines = axs[row, col].plot(
            full_data_log_sensitivity.reshape(
                full_data_log_sensitivity.shape[0], *full_data.shape[1:]
            )[indices_full[::-1], idx, :].T,
            label=variational_params_filtered_aug[indices_full[::-1]],
        )
        axs[row, col].set_title(name)
    fig.legend(
        lines, variational_params_filtered_aug[indices_full][::-1], loc="outside right"
    )
