import numpy as np

# KF macrostate variables as of april 1, 2024:
# state_vars = [
#     "total_P_DAMPS",
#     "total_T1IFN",
#     "total_TNF",
#     "total_IFNg",
#     "total_IL6",
#     "total_IL1",
#     "total_IL8",
#     "total_IL10",
#     "total_IL12",
#     "total_IL18",
#     "total_extracellular_virus",
#     "total_intracellular_virus",
#     "apoptosis_eaten_counter",
#     "empty_epithelium_count",
#     "healthy_epithelium_count",
#     "infected_epithelium_count",
#     "dead_epithelium_count",
#     "apoptosed_epithelium_count",
#     # "system_health",
#     "dc_count",
#     "nk_count",
#     "pmn_count",
#     "macro_count",
# ]
# variational_params = [
#     # INSENSITIVE "macro_phago_recovery",
#     # INSENSITIVE "macro_phago_limit",
#     "inflammasome_activation_threshold",
#     # INSENSITIVE "inflammasome_priming_threshold",
#     # INSENSITIVE "viral_carrying_capacity",
#     # INSENSITIVE "susceptibility_to_infection",
#     "human_endo_activation",
#     "human_metabolic_byproduct",
#     # NOT USED "resistance_to_infection",
#     "viral_incubation_threshold",
#     "epi_apoptosis_threshold_lower",
#     # INSENSITIVE "epi_apoptosis_threshold_range",
#     # INSENSITIVE "epi_apoptosis_threshold_lower_regrow",
#     # INSENSITIVE "epi_apoptosis_threshold_range_regrow",
#     # INSENSITIVE "epi_regrowth_counter_threshold",
#     # INSENSITIVE "epi_cell_membrane_init_lower",
#     # INSENSITIVE "epi_cell_membrane_init_range",
#     # INSENSITIVE "infected_epithelium_ros_damage_counter_threshold",
#     "epithelium_ros_damage_counter_threshold",
#     "epithelium_pdamps_secretion_on_death",
#     # INSENSITIVE "dead_epithelium_pdamps_burst_secretion",
#     # INSENSITIVE "dead_epithelium_pdamps_secretion",
#     # INSENSITIVE "epi_max_tnf_uptake",
#     # INSENSITIVE "epi_max_il1_uptake",
#     # INSENSITIVE "epi_t1ifn_secretion",
#     # INSENSITIVE "epi_t1ifn_secretion_prob",
#     # INSENSITIVE "epi_pdamps_secretion_prob",
#     "infected_epi_t1ifn_secretion",
#     # INSENSITIVE "infected_epi_il18_secretion",
#     # INSENSITIVE "infected_epi_il6_secretion",
#     "activated_endo_death_threshold",
#     # INSENSITIVE "activated_endo_adhesion_threshold",
#     "activated_endo_pmn_spawn_prob",
#     "activated_endo_pmn_spawn_dist",
#     "extracellular_virus_init_amount_lower",
#     # INSENSITIVE "extracellular_virus_init_amount_range",
#     # INSENSITIVE "human_t1ifn_effect_scale",
#     # INSENSITIVE "pmn_max_age",
#     "pmn_ros_secretion_on_death",
#     "pmn_il1_secretion_on_death",
#     # INSENSITIVE "nk_ifng_secretion",
#     # INSENSITIVE "macro_max_virus_uptake",
#     "macro_activation_threshold",
#     # INSENSITIVE "macro_antiactivation_threshold",
#     # INSENSITIVE "activated_macro_il8_secretion",
#     # INSENSITIVE "activated_macro_il12_secretion",
#     "activated_macro_tnf_secretion",
#     # INSENSITIVE "activated_macro_il6_secretion",
#     # INSENSITIVE "activated_macro_il10_secretion",
#     # INSENSITIVE "antiactivated_macro_il10_secretion",
#     "inflammasome_il1_secretion",
#     "inflammasome_macro_pre_il1_secretion",
#     # INSENSITIVE "inflammasome_il18_secretion",
#     # INSENSITIVE "inflammasome_macro_pre_il18_secretion",
#     # INSENSITIVE "pyroptosis_macro_pdamps_secretion",
#     # INSENSITIVE "dc_t1ifn_activation_threshold",
#     # INSENSITIVE "dc_il12_secretion",
#     # INSENSITIVE "dc_ifng_secretion",
#     # INSENSITIVE "dc_il6_secretion",
#     # INSENSITIVE "dc_il6_max_uptake",
#     # # ACK's Executive Judgement: These are physics-like parameters and won't vary between individuals.
#     # # They also include model-intrinsic things like a cleanup thresholds which don't precisely
#     # # correspond to read world objects.
#     # "human_viral_lower_bound", # 0.0
#     # "extracellular_virus_diffusion_const",
#     # "T1IFN_diffusion_const",
#     # "PAF_diffusion_const",
#     # "ROS_diffusion_const",
#     # "P_DAMPS_diffusion_const",
#     # "IFNg_diffusion_const",
#     # "TNF_diffusion_const",
#     # "IL6_diffusion_const",
#     # "IL1_diffusion_const",
#     # "IL10_diffusion_const",
#     # "IL12_diffusion_const",
#     # "IL18_diffusion_const",
#     # "IL8_diffusion_const",
#     # "extracellular_virus_cleanup_threshold",
#     # "cleanup_threshold",
#     # "evap_const_1",
#     # "evap_const_2",
# ]
################################################################################
# So the state variable indices are:
# 0    "total_P_DAMPS"
# 1    "total_T1IFN"
# 2    "total_TNF"
# 3    "total_IFNg"
# 4    "total_IL6"
# 5    "total_IL1"
# 6    "total_IL8"
# 7    "total_IL10"
# 8    "total_IL12"
# 9    "total_IL18"
# 10    "total_extracellular_virus"
# 11    "total_intracellular_virus"
# 12    "apoptosis_eaten_counter"
# 13    "empty_epithelium_count"
# 14    "healthy_epithelium_count"
# 15    "infected_epithelium_count"
# 16    "dead_epithelium_count"
# 17    "apoptosed_epithelium_count"
# 18    "dc_count"
# 19    "nk_count"
# 20    "pmn_count"
# 21    "macro_count"
# 22    "inflammasome_activation_threshold"
# 23    "human_endo_activation"
# 24    "human_metabolic_byproduct"
# PREVIOUS 25 resistance to infection, was unused in model
# 25    "viral_incubation_threshold"
# 26    "epi_apoptosis_threshold_lower"
# 27    "epithelium_ros_damage_counter_threshold"
# 28    "epithelium_pdamps_secretion_on_death"
# 29    "infected_epi_t1ifn_secretion"
# 30    "activated_endo_death_threshold"
# 31    "activated_endo_pmn_spawn_prob"
# 32    "activated_endo_pmn_spawn_dist"
# 33    "extracellular_virus_init_amount_lower"
# 34    "pmn_ros_secretion_on_death"
# 35    "pmn_il1_secretion_on_death"
# 36    "macro_activation_threshold"
# 37    "activated_macro_tnf_secretion"
# 38    "inflammasome_il1_secretion"
# 30    "inflammasome_macro_pre_il1_secretion"


# use a slight shift on the log-transform as variables can be exactly zero
# and arithmetic using -inf=np.log(0.0) gives poor results.
__EPSILON__ = 1e-3


def transform_intrinsic_to_kf(
    macrostate_intrinsic: np.ndarray, *, index=-1
) -> np.ndarray:
    """
    Transform an intrinsic macrostate to a normalized one for the KF.

    :param macrostate_intrinsic: intrinsic macrostate
    :param index: which index to transform, for arrays with single components
    :return: normalized macrostate for kf
    """
    if index == -1:
        # full state
        retval = np.log(__EPSILON__ + macrostate_intrinsic)
        return retval
    else:
        # parameters
        return np.log(__EPSILON__ + macrostate_intrinsic)


def transform_kf_to_intrinsic(macrostate_kf: np.ndarray, *, index=-1) -> np.ndarray:
    """
    Transform a normalized macrostate to an intrinsic one.

    :param macrostate_kf: normalized macrostate for kf
    :param index: which index to transform, for arrays with single components
    :return: intrinsic macrostate
    """
    if index == -1:
        # full state
        retval = np.maximum(0.0, np.exp(macrostate_kf) - __EPSILON__)
        return retval
    else:
        # parameters
        return np.maximum(0.0, np.exp(macrostate_kf) - __EPSILON__)
