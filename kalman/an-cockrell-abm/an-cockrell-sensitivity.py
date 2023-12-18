#!/usr/bin/env python3
# coding: utf-8

# # An-Cockrell model reimplementation

import h5py
import matplotlib.pyplot as plt
import numpy as np

# must match params from an-cockrell-runner.py
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
    "resistance_to_infection",
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
    "human_viral_lower_bound",
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
variational_params_filtered = [s for s in variational_params if s != "human_viral_lower_bound"]
variational_params_aug = np.array(variational_params + ["bias"])
variational_params_filtered_aug = np.array(variational_params_filtered + ["bias"])
varp_idcs = {str(vp): i for i, vp in enumerate(variational_params_aug)}
varpf_idcs = {str(vp): i for i, vp in enumerate(variational_params_filtered_aug)}

# with h5py.File("run-statistics.hdf5", "r") as f:

f = h5py.File("run-statistics.hdf5", "r")

min_sys_health = np.min(f["system_health"][()], axis=1)
param_list_filtered = f["param_list"][()][
    :, [i for i in range(len(variational_params)) if i != varp_idcs["human_viral_lower_bound"]]
]

param_list_filtered_aug = np.vstack(
    [param_list_filtered.T, np.ones(len(param_list_filtered))[np.newaxis]]
)

lsq_sol, residuals, rank, sing_vals = np.linalg.lstsq(
    param_list_filtered_aug.T, min_sys_health, rcond=None
)

sort_idcs = np.argsort(np.abs(lsq_sol))
sensitivity_dict = dict(zip(variational_params_filtered_aug[sort_idcs], lsq_sol[sort_idcs]))
print(sensitivity_dict)

plt.plot(np.abs(lsq_sol[np.argsort(np.abs(lsq_sol))]))

epsilon = np.exp(-25)
log_min_sys_health = np.log(np.maximum(epsilon, np.min(f["system_health"][()], axis=1)))

# log_param_list_filtered = log_param_list[
#     :, [i for i in range(len(variational_params)) if i != varp_idcs["human_viral_lower_bound"]]
# ]
log_param_list_filtered = np.log(
    f["param_list"][
        :, [i for i in range(len(variational_params)) if i != varp_idcs["human_viral_lower_bound"]]
    ]
)
log_param_list_filtered_aug = np.vstack(
    [log_param_list_filtered.T, np.ones(len(log_param_list_filtered))[np.newaxis]]
)

log_lsq_sol, log_residuals, log_rank, log_sing_vals = np.linalg.lstsq(
    log_param_list_filtered_aug.T, log_min_sys_health, rcond=None
)

plt.plot(np.abs(lsq_sol[np.argsort(np.abs(lsq_sol))]))

log_sort_idcs = np.argsort(np.abs(log_lsq_sol))
log_sensitivity_dict = dict(
    zip(variational_params_filtered_aug[log_sort_idcs], log_lsq_sol[log_sort_idcs])
)
print(log_sensitivity_dict)
plt.plot(np.abs(lsq_sol[np.argsort(np.abs(lsq_sol))]))

fig, ax = plt.subplots()
ax.scatter(np.abs(lsq_sol)[:-1], np.abs(log_lsq_sol)[:-1])
plt.xlabel("abs sensitivity")
plt.ylabel("abs log sensitivity")
for i, txt in enumerate(variational_params_filtered):
    ax.annotate(txt, (np.abs(lsq_sol)[i], np.abs(log_lsq_sol)[i]))


################################################################################

system_health = f["system_health"][()]

plt.hist(system_health.reshape(-1), bins=100, density=True)

sh_lsq_sol, sh_residuals, sh_rank, sh_sing_vals = np.linalg.lstsq(
    param_list_filtered_aug.T, system_health, rcond=None
)

fig, axs = plt.subplots(8, figsize=(8, 20))
for plt_idx in range(8):
    axs[plt_idx].imshow(np.abs(sh_lsq_sol[:, plt_idx * 252:(plt_idx + 1) * 252]), aspect=1)
fig.tight_layout()

sh_lsq_sol_col_norm = np.linalg.norm(sh_lsq_sol,axis=1)
fig, axs = plt.subplots(8, figsize=(8, 20))
for plt_idx in range(8):
    axs[plt_idx].imshow(np.abs(sh_lsq_sol[:, plt_idx * 252:(plt_idx + 1) * 252])/sh_lsq_sol_col_norm[:,np.newaxis], aspect=1)
fig.tight_layout()



log_system_health = np.log(np.maximum(np.exp(-25), system_health))
log_sh_lsq_sol, log_sh_residuals, log_sh_rank, log_sh_sing_vals = np.linalg.lstsq(
    log_param_list_filtered_aug.T, log_system_health, rcond=None
)

fig, axs = plt.subplots(8, figsize=(8, 20))
for plt_idx in range(8):
    axs[plt_idx].imshow(np.abs(log_sh_lsq_sol[:, plt_idx * 252:(plt_idx + 1) * 252]), aspect=1)
fig.tight_layout()


log_sh_lsq_sol_col_norm = np.linalg.norm(log_sh_lsq_sol,axis=1)
fig, axs = plt.subplots(8, figsize=(8, 20))
for plt_idx in range(8):
    axs[plt_idx].imshow(np.abs(log_sh_lsq_sol[:, plt_idx * 252:(plt_idx + 1) * 252])/log_sh_lsq_sol_col_norm[:,np.newaxis], aspect=1)
fig.tight_layout()


# log sensitivity is ever above min_sensitivity
max_log_sh_lsq_sol = np.max(np.abs(log_sh_lsq_sol), axis=1)
# plt.gca().clear()
min_sensitivity = 0.25
plt.plot(log_sh_lsq_sol.T[:,max_log_sh_lsq_sol > min_sensitivity][:,:-1], label=variational_params_filtered_aug[max_log_sh_lsq_sol > min_sensitivity][:-1])
plt.legend()
plt.tight_layout()

max_sh_lsq_sol = np.max(np.abs(sh_lsq_sol), axis=1)
# plt.gca().clear()
# min_sensitivity = 0.0
plt.plot(sh_lsq_sol.T[:,:-1], label=variational_params_filtered_aug[:-1])
# plt.plot(log_sh_lsq_sol.T[:,max_log_sh_lsq_sol > min_sensitivity][:,:-1], label=variational_params_filtered_aug[max_log_sh_lsq_sol > min_sensitivity][:-1])
plt.legend()
plt.tight_layout()

idcs = np.argsort(max_sh_lsq_sol[:-1])[-10:]
plt.plot(sh_lsq_sol.T[:,idcs], label=variational_params_filtered_aug[idcs])
# plt.plot(log_sh_lsq_sol.T[:,max_log_sh_lsq_sol > min_sensitivity][:,:-1], label=variational_params_filtered_aug[max_log_sh_lsq_sol > min_sensitivity][:-1])
plt.legend()
plt.tight_layout()


log_idcs = np.argsort(max_log_sh_lsq_sol[:-1])[-10:]
top_ten = set(variational_params_filtered_aug[idcs])
log_top_ten = set(variational_params_filtered_aug[log_idcs])
plt.plot(log_sh_lsq_sol.T[:,log_idcs], label=variational_params_filtered_aug[log_idcs])
plt.legend()
plt.tight_layout()


norm_sh_lsq_sol = np.linalg.norm(np.abs(sh_lsq_sol), axis=1)
norm_log_sh_lsq_sol = np.linalg.norm(np.abs(log_sh_lsq_sol), axis=1)
norm_idcs = np.argsort(norm_sh_lsq_sol[:-1])[-10:]
norm_log_idcs = np.argsort(norm_log_sh_lsq_sol[:-1])[-10:]
norm_top_ten = set(variational_params_filtered_aug[norm_idcs])
norm_log_top_ten = set(variational_params_filtered_aug[norm_log_idcs])
plt.plot(log_sh_lsq_sol.T[:,norm_log_idcs], label=variational_params_filtered_aug[norm_log_idcs])
plt.legend()
plt.tight_layout()


print(set(top_ten).intersection(log_top_ten))
print(set(top_ten).union(log_top_ten))


fig, ax = plt.subplots()
ax.scatter(norm_sh_lsq_sol[:-1], norm_log_sh_lsq_sol[:-1])
plt.xlabel("sensitivity")
plt.ylabel("log sensitivity")
for i, txt in enumerate(variational_params_filtered):
    ax.annotate(txt, (norm_sh_lsq_sol[i], norm_log_sh_lsq_sol[i]))
