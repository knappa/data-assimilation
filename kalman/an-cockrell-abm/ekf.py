import argparse
import csv
import sys
from typing import Dict

import an_cockrell
import matplotlib.pyplot as plt
import numpy as np
import scipy
from an_cockrell import AnCockrellModel, EndoType, EpiType
from scipy.stats import multivariate_normal
from tqdm.auto import tqdm

################################################################################

if hasattr(sys, "ps1"):
    # interactive mode
    args = object()
else:
    parser = argparse.ArgumentParser()

    parser.add_argument("--prefix", type=str, default="", help="output file prefix")

    parser.add_argument(
        "--measurements",
        type=str,
        choices=[
            "T1IFN",
            "TNF",
            "IFNg",
            "IL6",
            "IL1",
            "IL8",
            "IL10",
            "IL12",
            "IL18",
            "extracellular_virus",
        ],
        nargs="+",
        required=True,
        help="which things to measure (required)",
    )

    parser.add_argument(
        "--matchmaker",
        help="try to match resampled macrostates with microstate "
        "models to minimize change magnitude",
        type=str,
        choices=["yes", "no"],
        required=True,
    )

    parser.add_argument("--graphs", help="make pdf graphs", action="store_true")

    args = parser.parse_args()

VERBOSE = False

################################################################################
# constants

default_params = dict(
    GRID_WIDTH=51,
    GRID_HEIGHT=51,
    is_bat=False,
    init_inoculum=100,
    init_dcs=50,
    init_nks=25,
    init_macros=50,
    macro_phago_recovery=0.5,
    macro_phago_limit=1_000,
    inflammasome_activation_threshold=10,  # default 50 for bats
    inflammasome_priming_threshold=1.0,  # default 5.0 for bats
    viral_carrying_capacity=500,
    susceptibility_to_infection=77,
    human_endo_activation=5,
    human_metabolic_byproduct=0.2,
    resistance_to_infection=75,
    viral_incubation_threshold=60,
    epi_apoptosis_threshold_lower=450,
    epi_apoptosis_threshold_range=100,
    epi_apoptosis_threshold_lower_regrow=475,
    epi_apoptosis_threshold_range_regrow=51,
    epi_regrowth_counter_threshold=432,
    epi_cell_membrane_init_lower=975,
    epi_cell_membrane_init_range=51,
    infected_epithelium_ros_damage_counter_threshold=10,
    epithelium_ros_damage_counter_threshold=2,
    epithelium_pdamps_secretion_on_death=10.0,
    dead_epithelium_pdamps_burst_secretion=10.0,
    dead_epithelium_pdamps_secretion=1.0,
    epi_max_tnf_uptake=0.1,
    epi_max_il1_uptake=0.1,
    epi_t1ifn_secretion=0.75,
    epi_t1ifn_secretion_prob=0.01,
    epi_pdamps_secretion_prob=0.01,
    infected_epi_t1ifn_secretion=1.0,
    infected_epi_il18_secretion=0.11,
    infected_epi_il6_secretion=0.10,
    activated_endo_death_threshold=0.5,
    activated_endo_adhesion_threshold=36.0,
    activated_endo_pmn_spawn_prob=0.1,
    activated_endo_pmn_spawn_dist=5.0,
    extracellular_virus_init_amount_lower=80,
    extracellular_virus_init_amount_range=40,
    human_viral_lower_bound=0.0,
    human_t1ifn_effect_scale=0.01,
    pmn_max_age=36,
    pmn_ros_secretion_on_death=10.0,
    pmn_il1_secretion_on_death=1.0,
    nk_ifng_secretion=1.0,
    macro_max_virus_uptake=10.0,
    macro_activation_threshold=5.0,
    macro_antiactivation_threshold=5.0,
    activated_macro_il8_secretion=1.0,
    activated_macro_il12_secretion=0.5,
    activated_macro_tnf_secretion=1.0,
    activated_macro_il6_secretion=0.4,
    activated_macro_il10_secretion=1.0,
    antiactivated_macro_il10_secretion=0.5,
    inflammasome_il1_secretion=1.0,
    inflammasome_macro_pre_il1_secretion=5.0,
    inflammasome_il18_secretion=1.0,
    inflammasome_macro_pre_il18_secretion=0.5,
    pyroptosis_macro_pdamps_secretion=10.0,
    dc_t1ifn_activation_threshold=1.0,
    dc_il12_secretion=0.5,
    dc_ifng_secretion=0.5,
    dc_il6_secretion=0.4,
    dc_il6_max_uptake=0.1,
    extracellular_virus_diffusion_const=0.05,
    T1IFN_diffusion_const=0.1,
    PAF_diffusion_const=0.1,
    ROS_diffusion_const=0.1,
    P_DAMPS_diffusion_const=0.1,
    IFNg_diffusion_const=0.2,
    TNF_diffusion_const=0.2,
    IL6_diffusion_const=0.2,
    IL1_diffusion_const=0.2,
    IL10_diffusion_const=0.2,
    IL12_diffusion_const=0.2,
    IL18_diffusion_const=0.2,
    IL8_diffusion_const=0.3,
    extracellular_virus_cleanup_threshold=0.05,
    cleanup_threshold=0.1,
    evap_const_1=0.99,
    evap_const_2=0.9,
)

# variables which reflect the macrostate of the model
state_vars = [
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
    "infected_epithelium_count",
    "dead_epithelium_count",
    "apoptosed_epithelium_count",
    "healthy_epithelium_count",
    # "system_health",
    "dc_count",
    "nk_count",
    "pmn_count",
    "macro_count",
]

state_var_indices = {s: i for i, s in enumerate(state_vars)}

# layout for graphing state variables.
# Attempts to be mostly square, with possibly more rows than columns
state_var_graphs_cols: int = int(np.floor(np.sqrt(len(state_vars))))
state_var_graphs_rows: int = int(np.ceil(len(state_vars) / state_var_graphs_cols))
state_var_graphs_figsize = (1.6 * state_var_graphs_rows, 1.6 * state_var_graphs_cols)

# parameters of the model used only for initialization
init_only_params = [
    "init_inoculum",
    "init_dcs",
    "init_nks",
    "init_macros",
]

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

variational_params_indices = {s: i + len(state_vars) for i, s in enumerate(state_vars)}

# layout for graphing parameters.
# Attempts to be mostly square, with possibly more rows than columns
variational_params_graphs_cols: int = int(np.floor(np.sqrt(len(state_vars))))
variational_params_graphs_rows: int = int(np.ceil(len(state_vars) / state_var_graphs_cols))
variational_params_graphs_figsize = (
    1.6 * variational_params_graphs_rows,
    1.6 * variational_params_graphs_cols,
)

assert all(param in default_params for param in variational_params)

TIME_SPAN = 2016
SAMPLE_INTERVAL = 48  # how often to make measurements
ENSEMBLE_SIZE = 50
UNIFIED_STATE_SPACE_DIMENSION = len(state_vars) + len(variational_params)
OBSERVABLES = ["extracellular_virus"] if not hasattr(args, "measurements") else args.measurements
OBSERVABLE_VAR_NAMES = ["total_" + name for name in OBSERVABLES]

RESAMPLE_MODELS = False

# if we are altering the models (as opposed to resampling) try to match the
# models to minimize the changes necessary.
MODEL_MATCHMAKER = True if not hasattr(args, "matchmaker") else (args.matchmaker == "yes")

# have the models' parameters do a random walk over time (should help
# with covariance starvation)
PARAMETER_RANDOM_WALK = True

FILE_PREFIX = "" if not hasattr(args, "prefix") else args.prefix + "-"

GRAPHS = False if not hasattr(args, "graphs") else bool(args.graphs)

################################################################################
# statistical parameters

init_mean_vec = np.array(
    [default_params[param] for param in (init_only_params + variational_params)]
)

init_cov_matrix = np.diag(
    np.array(
        [0.75 * np.sqrt(default_params[param]) for param in (init_only_params + variational_params)]
    )
)


################################################################################


def model_macro_data(model: AnCockrellModel):
    """
    Collect macroscale data from a model
    :param model:
    :return:
    """
    macroscale_data = np.zeros(UNIFIED_STATE_SPACE_DIMENSION, dtype=np.float64)

    for idx, param in enumerate(state_vars):
        macroscale_data[idx] = getattr(model, param)

    for idx, param in enumerate(variational_params):
        macroscale_data[len(state_vars) + idx] = getattr(model, param)

    return macroscale_data


################################################################################
# sample a virtual patient

# sampled virtual patient parameters
vp_init_params = default_params.copy()
vp_init_param_sample = np.abs(multivariate_normal(mean=init_mean_vec, cov=init_cov_matrix).rvs())
for sample_component, param_name in zip(
    vp_init_param_sample,
    (init_only_params + variational_params),
):
    vp_init_params[param_name] = (
        round(sample_component) if isinstance(default_params[param_name], int) else sample_component
    )

# create model for virtual patient
virtual_patient_model = an_cockrell.AnCockrellModel(**vp_init_params)

# evaluate the virtual patient's trajectory
vp_trajectory = np.zeros((TIME_SPAN + 1, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64)

vp_trajectory[0, :] = model_macro_data(virtual_patient_model)
# noinspection PyTypeChecker
for t in tqdm(range(1, TIME_SPAN + 1), desc="create virtual patient"):
    virtual_patient_model.time_step()
    vp_trajectory[t, :] = model_macro_data(virtual_patient_model)

################################################################################
# plot virtual patient

if GRAPHS:
    fig, axs = plt.subplots(
        nrows=state_var_graphs_rows,
        ncols=state_var_graphs_cols,
        figsize=state_var_graphs_figsize,
        sharex=True,
        sharey=False,
    )
    for idx, state_var_name in enumerate(state_vars):
        row, col = idx // state_var_graphs_cols, idx % state_var_graphs_cols
        axs[row, col].plot(vp_trajectory[:, idx])
        axs[row, col].set_title(state_var_name)
    fig.tight_layout()
    fig.savefig(FILE_PREFIX + "virtual-patient.pdf")
    plt.close(fig)


################################################################################


def model_ensemble_from(means, covariances):
    """
    Create an ensemble of models from a distribution. Uses init-only
    and variational parameters

    :param means:
    :param covariances:
    :return:
    """
    mdl_ensemble = []
    distribution = multivariate_normal(mean=means, cov=covariances, allow_singular=True)
    for _ in range(ENSEMBLE_SIZE):
        model_param_dict = default_params.copy()
        sampled_params = np.abs(distribution.rvs())
        for sample_component, param_name in zip(
            sampled_params,
            (init_only_params + variational_params),
        ):
            model_param_dict[param_name] = (
                round(sample_component)
                if isinstance(default_params[param_name], int)
                else sample_component
            )
        # create model for virtual patient
        model = AnCockrellModel(**model_param_dict)
        mdl_ensemble.append(model)

    return mdl_ensemble


################################################################################


def modify_model(
    model: AnCockrellModel,
    desired_state: np.ndarray,
    *,
    ignore_state_vars: bool = False,
):
    """
    Modify a model's microstate to fit a given macrostate

    :param model: model instance (encodes microstate)
    :param desired_state: desired macrostate for the model
    :param ignore_state_vars: if True, only alter parameters, not state variables
    :return: None
    """
    np.abs(desired_state, out=desired_state)  # in-place absolute value

    for param_idx, param_name in enumerate(variational_params, start=len(state_vars)):
        prev_value = getattr(model, param_name, None)
        assert prev_value is not None
        if isinstance(prev_value, int):
            setattr(model, param_name, round(float(desired_state[param_idx])))
        else:
            setattr(model, param_name, desired_state[param_idx])

    if ignore_state_vars:
        return

    # it's worth pointing out that, because of the cleanup-below-a-threshold code,
    # these reset-by-scaling fields may not go to quite the right thing on the next
    # time step. I'm not seeing an easy fix/alternative here, so ¯\_(ツ)_/¯ ?

    desired_t1ifn = desired_state[state_var_indices["total_T1IFN"]]
    model.T1IFN *= desired_t1ifn / model.total_T1IFN

    desired_tnf = desired_state[state_var_indices["total_TNF"]]
    model.TNF *= desired_tnf / model.total_TNF

    desired_ifn_g = desired_state[state_var_indices["total_IFNg"]]
    model.IFNg *= desired_ifn_g / model.total_IFNg

    desired_il6 = desired_state[state_var_indices["total_IL6"]]
    model.IL6 *= desired_il6 / model.total_IL6

    desired_il1 = desired_state[state_var_indices["total_IL1"]]
    model.IL1 *= desired_il1 / model.total_IL1

    desired_il8 = desired_state[state_var_indices["total_IL8"]]
    model.IL8 *= desired_il8 / model.total_IL8

    desired_il10 = desired_state[state_var_indices["total_IL10"]]
    model.IL10 *= desired_il10 / model.total_IL10

    desired_il12 = desired_state[state_var_indices["total_IL12"]]
    model.IL12 *= desired_il12 / model.total_IL12

    desired_il18 = desired_state[state_var_indices["total_IL18"]]
    model.IL18 *= desired_il18 / model.total_IL18

    desired_total_extracellular_virus = desired_state[
        state_var_indices["total_extracellular_virus"]
    ]
    model.extracellular_virus *= desired_total_extracellular_virus / model.total_extracellular_virus

    desired_total_intracellular_virus = desired_state[
        state_var_indices["total_intracellular_virus"]
    ]
    # TODO: how close does this get? do we have to deal with discretization error?
    np.rint(
        model.epi_intracellular_virus[:]
        * (desired_total_intracellular_virus / model.total_intracellular_virus),
        out=model.epi_intracellular_virus,
    )

    model.apoptosis_eaten_counter = int(
        desired_state[state_var_indices["apoptosis_eaten_counter"]]
    )  # no internal state here

    # epithelium: infected, dead, apoptosed, healthy
    desired_epithelium = np.array(
        np.round(
            desired_state[
                [
                    state_var_indices["infected_epithelium_count"],
                    state_var_indices["dead_epithelium_count"],
                    state_var_indices["apoptosed_epithelium_count"],
                    state_var_indices["healthy_epithelium_count"],
                ]
            ]
        ),
        dtype=np.int64,
    )
    desired_total_epithelium = np.sum(desired_epithelium)
    # Since these just samples from a normal distribution, the sampling might request more epithelium than
    # there is room for. We try to do our best...
    if desired_total_epithelium > model.GRID_WIDTH * model.GRID_HEIGHT:
        # try a proportional rescale
        np.rint(
            desired_epithelium * (model.GRID_WIDTH * model.GRID_HEIGHT / desired_total_epithelium),
            out=desired_epithelium,
        )
        assert np.all(desired_epithelium >= 0)
        desired_total_epithelium = np.sum(desired_epithelium)
        # if that didn't go all the way (b/c e.g. rounding) knock off random individuals until it's ok
        assert model.GRID_WIDTH * model.GRID_HEIGHT > 0
        while desired_total_epithelium > model.GRID_WIDTH * model.GRID_HEIGHT:
            rand_idx = np.random.randint(4)
            if desired_epithelium[rand_idx] > 0:
                desired_epithelium[rand_idx] -= 1
                desired_total_epithelium -= 1
    # now we are certain that desired_epithelium holds attainable values

    # first, kill off (empty) any excess in the epithelial categories
    if model.infected_epithelium_count > desired_epithelium[0]:
        infected_locations = np.array(np.where(model.epithelium == EpiType.Infected))
        empty_locs = np.random.choice(
            infected_locations.shape[1],
            model.infected_epithelium_count - desired_epithelium[0],
            replace=False,
        )
        for loc in empty_locs:
            model.epithelium[infected_locations[:, loc]] = EpiType.Empty
            model.epi_intracellular_virus[infected_locations[:, loc]] = 0

    if model.dead_epithelium_count > desired_epithelium[1]:
        dead_locations = np.array(np.where(model.epithelium == EpiType.Dead))
        empty_locs = np.random.choice(
            dead_locations.shape[1],
            model.dead_epithelium_count - desired_epithelium[1],
            replace=False,
        )
        for loc in empty_locs:
            model.epithelium[dead_locations[:, loc]] = EpiType.Empty
            model.epi_intracellular_virus[dead_locations[:, loc]] = 0

    if model.apoptosed_epithelium_count > desired_epithelium[2]:
        apoptosed_locations = np.array(np.where(model.epithelium == EpiType.Apoptosed))
        empty_locs = np.random.choice(
            apoptosed_locations.shape[1],
            model.apoptosed_epithelium_count - desired_epithelium[2],
            replace=False,
        )
        for loc in empty_locs:
            model.epithelium[apoptosed_locations[:, loc]] = EpiType.Empty
            model.epi_intracellular_virus[apoptosed_locations[:, loc]] = 0

    if model.healthy_epithelium_count > desired_epithelium[3]:
        healthy_locations = np.array(np.where(model.epithelium == EpiType.Healthy))
        empty_locs = np.random.choice(
            healthy_locations.shape[1],
            model.healthy_epithelium_count - desired_epithelium[3],
            replace=False,
        )
        for loc in empty_locs:
            model.epithelium[healthy_locations[:, loc]] = EpiType.Empty
            model.epi_intracellular_virus[healthy_locations[:, loc]] = 0

    # second, spawn to make up for deficiency in the epithelial categories
    # TODO: check the following epithelium state variables
    #  epithelium_ros_damage_counter, epi_regrow_counter, epi_apoptosis_counter,
    #  epi_intracellular_virus, epi_cell_membrane, epi_apoptosis_threshold,
    #  epithelium_apoptosis_counter
    if model.infected_epithelium_count < desired_epithelium[0]:
        empty_locations = np.array(np.where(model.epithelium == EpiType.Empty))
        infected_locs = np.random.choice(
            empty_locations.shape[1],
            desired_epithelium[0] - model.infected_epithelium_count,
            replace=False,
        )
        for loc in infected_locs:
            model.epithelium[empty_locations[:, loc]] = EpiType.Infected
            # virus_invade_epi_cell says also: self.epi_intracellular_virus[invasion_mask] += 1
            model.epi_intracellular_virus[empty_locations[:, loc]] += 1

    if model.dead_epithelium_count < desired_epithelium[1]:
        empty_locations = np.array(np.where(model.epithelium == EpiType.Empty))
        dead_locs = np.random.choice(
            empty_locations.shape[1],
            desired_epithelium[1] - model.dead_epithelium_count,
            replace=False,
        )
        for loc in dead_locs:
            model.epithelium[empty_locations[:, loc]] = EpiType.Dead
            model.epi_intracellular_virus[empty_locations[:, loc]] = 0

    if model.apoptosed_epithelium_count < desired_epithelium[2]:
        empty_locations = np.array(np.where(model.epithelium == EpiType.Empty))
        apoptosed_locs = np.random.choice(
            empty_locations.shape[1],
            desired_epithelium[2] - model.apoptosed_epithelium_count,
            replace=False,
        )
        for loc in apoptosed_locs:
            model.epithelium[empty_locations[:, loc]] = EpiType.Apoptosed
            model.epi_intracellular_virus[empty_locations[:, loc]] = 0

    if model.healthy_epithelium_count < desired_epithelium[3]:
        empty_locations = np.array(np.where(model.epithelium == EpiType.Empty))
        healthy_locs = np.random.choice(
            empty_locations.shape[1],
            desired_epithelium[3] - model.healthy_epithelium_count,
            replace=False,
        )
        for loc in healthy_locs:
            model.epithelium[empty_locations[:, loc]] = EpiType.Healthy
            model.epi_intracellular_virus[empty_locations[:, loc]] = 0

    dc_delta = desired_state[state_var_indices["dc_count"]] - model.dc_count
    if dc_delta > 0:
        for _ in range(dc_delta):
            model.create_dc()
    elif dc_delta < 0:
        # need fewer dcs, kill them randomly
        dcs_to_kill = np.random.choice(model.num_dcs, min(-dc_delta, model.num_dcs), replace=False)
        dc_idcs = np.where(model.dc_mask)[0]
        model.dc_mask[dc_idcs[dcs_to_kill]] = False

    nk_delta = desired_state[state_var_indices["nk_count"]] - model.nk_count
    if nk_delta > 0:
        model.create_nk(number=int(nk_delta))
    elif nk_delta < 0:
        # need fewer nks, kill them randomly
        nks_to_kill = np.random.choice(model.num_nks, min(-nk_delta, model.num_nks), replace=False)
        nk_idcs = np.where(model.nk_mask)[0]
        model.nk_mask[nk_idcs[nks_to_kill]] = False

    pmn_delta = desired_state[state_var_indices["pmn_count"]] - model.pmn_count
    if pmn_delta > 0:
        # need more pmns
        # adapted from activated_endo_update
        pmn_spawn_list = list(
            zip(
                *np.where(
                    (model.endothelial_adhesion_counter > model.activated_endo_adhesion_threshold)
                    & (model.endothelial_activation == EndoType.Activated)
                    & (np.random.rand(*model.geometry) < model.activated_endo_pmn_spawn_prob)
                )
            )
        )
        if len(pmn_spawn_list) > 0:
            for loc_idx in np.random.choice(len(pmn_spawn_list), pmn_delta):
                model.create_pmn(
                    location=pmn_spawn_list[loc_idx],
                    age=0,
                    jump_dist=model.activated_endo_pmn_spawn_dist,
                )
        else:
            if VERBOSE:
                print("Nowhere to put desired pmns")
    elif pmn_delta < 0:
        # need fewer pmns, kill them randomly
        pmns_to_kill = np.random.choice(
            model.num_pmns, min(-pmn_delta, model.num_pmns), replace=False
        )
        pmn_idcs = np.where(model.pmn_mask)[0]
        model.pmn_mask[pmn_idcs[pmns_to_kill]] = False

    macro_delta = desired_state[state_var_indices["macro_count"]] - model.macro_count
    if macro_delta > 0:
        # need more macrophages, create them as in init in random locations
        for _ in range(macro_delta):
            model.create_macro(
                pre_il1=0,
                pre_il18=0,
                inflammasome_primed=False,
                inflammasome_active=False,
                macro_activation_level=0,
                pyroptosis_counter=0,
                virus_eaten=0,
                cells_eaten=0,
            )
    elif macro_delta < 0:
        # need fewer macrophages, kill them randomly
        macros_to_kill = np.random.choice(
            model.num_macros, min(-macro_delta, model.num_macros), replace=False
        )
        macro_idcs = np.where(model.macro_mask)[0]
        model.macro_mask[macro_idcs[macros_to_kill]] = False


################################################################################
# Kalman filter simulation
################################################################################

# create ensemble of models for kalman filter
model_ensemble = model_ensemble_from(init_mean_vec, init_cov_matrix)

# mean and covariances through time
mean_vec = np.full((TIME_SPAN + 1, UNIFIED_STATE_SPACE_DIMENSION), -1, dtype=np.float64)
cov_matrix = np.full(
    (TIME_SPAN + 1, UNIFIED_STATE_SPACE_DIMENSION, UNIFIED_STATE_SPACE_DIMENSION),
    -1,
    dtype=np.float64,
)

# collect initial statistics
time = 0
initial_macro_data = np.array([model_macro_data(model) for model in model_ensemble])
mean_vec[time, :] = np.mean(initial_macro_data, axis=0)
cov_matrix[time, :, :] = np.cov(initial_macro_data, rowvar=False)

cycle = 0
while time < TIME_SPAN:
    cycle += 1
    print(f" *** {cycle=} *** ")
    # advance ensemble of models
    for _ in range(SAMPLE_INTERVAL):
        for model in model_ensemble:
            model.time_step()
            if PARAMETER_RANDOM_WALK:
                macrostate = model_macro_data(model)
                random_walk_macrostate = np.abs(
                    macrostate
                    + multivariate_normal(
                        mean=np.zeros_like(macrostate),
                        cov=np.diag(0.01 * np.ones_like(macrostate)),
                    ).rvs()
                )
                modify_model(model, random_walk_macrostate, ignore_state_vars=True)
        time += 1
        macro_data = np.array([model_macro_data(model) for model in model_ensemble])
        mean_vec[time, :] = np.mean(macro_data, axis=0)
        cov_matrix[time, :, :] = np.cov(macro_data, rowvar=False)

    ################################################################################
    # plot state variables

    if GRAPHS:
        fig, axs = plt.subplots(
            nrows=state_var_graphs_rows,
            ncols=state_var_graphs_cols,
            figsize=state_var_graphs_figsize,
            sharex=True,
            sharey=False,
        )
        for idx, state_var_name in enumerate(state_vars):
            row, col = idx // state_var_graphs_cols, idx % state_var_graphs_cols
            axs[row, col].plot(
                vp_trajectory[: (cycle + 1) * SAMPLE_INTERVAL + 1, idx],
                label="true value",
                color="black",
            )
            axs[row, col].plot(
                range(cycle * SAMPLE_INTERVAL + 1),
                mean_vec[: cycle * SAMPLE_INTERVAL + 1, idx],
                label="estimate",
            )
            axs[row, col].fill_between(
                range(cycle * SAMPLE_INTERVAL + 1),
                np.maximum(
                    0.0,
                    mean_vec[: cycle * SAMPLE_INTERVAL + 1, idx]
                    - np.sqrt(cov_matrix[: cycle * SAMPLE_INTERVAL + 1, idx, idx]),
                ),
                mean_vec[: cycle * SAMPLE_INTERVAL + 1, idx]
                + np.sqrt(cov_matrix[: cycle * SAMPLE_INTERVAL + 1, idx, idx]),
                color="gray",
                alpha=0.35,
            )
            axs[row, col].set_title(state_var_name)
            axs[row, col].legend()
        fig.tight_layout()
        fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-match.pdf")
        plt.close(fig)

    ################################################################################
    # plot parameters

    if GRAPHS:
        fig, axs = plt.subplots(
            nrows=variational_params_graphs_rows,
            ncols=variational_params_graphs_cols,
            figsize=variational_params_graphs_figsize,
            sharex=True,
            sharey=False,
        )
        for idx, param_name in enumerate(state_vars):
            row, col = idx // variational_params_graphs_cols, idx % variational_params_graphs_cols

            axs[row, col].plot(
                [0, (cycle + 1) * SAMPLE_INTERVAL + 1],
                [vp_init_params[param_name]] * 2,
                label="true value",
                color="black",
            )
            axs[row, col].plot(
                range(cycle * SAMPLE_INTERVAL + 1),
                mean_vec[: cycle * SAMPLE_INTERVAL + 1, len(state_vars) + idx],
                label="estimate",
            )
            axs[row, col].fill_between(
                range(cycle * SAMPLE_INTERVAL + 1),
                np.maximum(
                    0.0,
                    mean_vec[: cycle * SAMPLE_INTERVAL + 1, len(state_vars) + idx]
                    - np.sqrt(
                        cov_matrix[
                            : cycle * SAMPLE_INTERVAL + 1,
                            len(state_vars) + idx,
                            len(state_vars) + idx,
                        ]
                    ),
                ),
                np.minimum(
                    10 * vp_init_params[param_name],
                    mean_vec[: cycle * SAMPLE_INTERVAL + 1, len(state_vars) + idx]
                    + np.sqrt(
                        cov_matrix[
                            : cycle * SAMPLE_INTERVAL + 1,
                            len(state_vars) + idx,
                            len(state_vars) + idx,
                        ]
                    ),
                ),
                color="gray",
                alpha=0.35,
            )
            axs[row, col].set_title(param_name)
            axs[row, col].legend()
        axs[2, 1].axis("off")
        fig.tight_layout()
        fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-match-params.pdf")
        plt.close(fig)

    ################################################################################
    # Kalman filter

    num_observables = len(OBSERVABLE_VAR_NAMES)

    # rs encodes the uncertainty in the various observations
    rs: Dict[str, float] = {
        "total_T1IFN": 1.0,
        "total_TNF": 1.0,
        "total_IFNg": 1.0,
        "total_IL6": 1.0,
        "total_IL1": 1.0,
        "total_IL8": 1.0,
        "total_IL10": 1.0,
        "total_IL12": 1.0,
        "total_IL18": 1.0,
        "total_extracellular_virus": 1.0,
    }
    R = np.diag([rs[obs_name] for obs_name in OBSERVABLE_VAR_NAMES])

    H = np.zeros((num_observables, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64)
    for h_idx, obs_name in enumerate(OBSERVABLE_VAR_NAMES):
        H[h_idx, state_var_indices[obs_name]] = 1.0

    observation = np.array(
        [vp_trajectory[time, state_var_indices[obs_name]] for obs_name in OBSERVABLE_VAR_NAMES],
        dtype=np.float64,
    )

    v = observation - (H @ mean_vec[time, :])
    S = H @ cov_matrix[time, :, :] @ H.T + R
    K = cov_matrix[time, :, :] @ H.T @ np.linalg.inv(S)

    mean_vec[time, :] += K @ v
    cov_matrix[time, :, :] -= K @ S @ K.T

    # numerical cleanup: symmetrize and project onto pos def cone
    cov_matrix[time, :, :] = np.nan_to_num(
        (np.nan_to_num(cov_matrix[time, :, :]) + np.nan_to_num(cov_matrix[time, :, :].T)) / 2.0
    )
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        cov_matrix[time, :, :], lower=True, check_finite=False
    )
    eigenvalues[:] = np.real(eigenvalues)  # just making sure
    eigenvectors[:, :] = np.real(eigenvectors)  # just making sure
    # spectrum must be positive.
    # from the scipy code, it also can't have a max/min e-val ratio bigger than 1/(1e6*double machine epsilon)
    # and that's ~4503599627.370496=1/(1e6*np.finfo('d').eps), so a ratio bounded by 1e9 is ok.
    cov_matrix[time, :, :] = (
        eigenvectors @ np.diag(np.minimum(1e5, np.maximum(1e-4, eigenvalues))) @ eigenvectors.T
    )
    cov_matrix[time, :, :] = np.nan_to_num(
        (np.nan_to_num(cov_matrix[time, :, :]) + np.nan_to_num(cov_matrix[time, :, :].T)) / 2.0
    )

    # recreate ensemble
    if RESAMPLE_MODELS:
        # create an entirely new set of model instances sampled from KF-learned distribution
        model_ensemble = model_ensemble_from(mean_vec[time, :], cov_matrix[time, :, :])
    else:
        dist = multivariate_normal(
            mean=mean_vec[time, :], cov=cov_matrix[time, :, :], allow_singular=True
        )
        if MODEL_MATCHMAKER:
            new_sample = dist.rvs(size=ENSEMBLE_SIZE)
            # Gale-Shapely matching algorithm to try and pair up the models and these new samples

            # fill out preference lists for the models
            prefs = np.zeros((ENSEMBLE_SIZE, ENSEMBLE_SIZE), dtype=np.int64)
            for idx in range(ENSEMBLE_SIZE):
                # noinspection PyUnboundLocalVariable
                dists = np.linalg.norm(new_sample - macro_data[idx], axis=1)
                prefs[idx, :] = np.argsort(dists)

            # arrays to record pairings
            model_to_sample_pairing = np.full(ENSEMBLE_SIZE, -1, dtype=np.int64)
            sample_to_model_pairing = np.full(ENSEMBLE_SIZE, -1, dtype=np.int64)

            all_paired = False
            while not all_paired:
                all_paired = True
                for model_idx in range(ENSEMBLE_SIZE):
                    if model_to_sample_pairing[model_idx] != -1:
                        # skip already paired models
                        continue
                    # found an unpaired model, find the first thing not yet
                    # checked on its preference list
                    min_pref_idx = np.argmax(prefs[model_idx, :] >= 0)
                    for pref_idx in range(min_pref_idx, ENSEMBLE_SIZE):
                        possible_sample_pair = prefs[model_idx, pref_idx]
                        competitor_model_idx = sample_to_model_pairing[possible_sample_pair]
                        if competitor_model_idx == -1:
                            # if the sample is unpaired, pair the two
                            sample_to_model_pairing[possible_sample_pair] = model_idx
                            model_to_sample_pairing[model_idx] = possible_sample_pair
                            # erase this possibility for future pairings
                            prefs[model_idx, pref_idx] = -1
                            break  # stop looking now
                        else:
                            # compare preferences
                            established_pair_dist = np.linalg.norm(
                                macro_data[competitor_model_idx, :]
                                - new_sample[possible_sample_pair, :]
                            )
                            proposed_pair_dist = np.linalg.norm(
                                macro_data[model_idx, :] - new_sample[possible_sample_pair, :]
                            )
                            if proposed_pair_dist < established_pair_dist:
                                model_to_sample_pairing[
                                    competitor_model_idx
                                ] = -1  # free the competitor
                                all_paired = False
                                # make new pair
                                sample_to_model_pairing[possible_sample_pair] = model_idx
                                model_to_sample_pairing[model_idx] = possible_sample_pair
                                # erase this possibility for future pairings
                                prefs[model_idx, pref_idx] = -1
                                break  # stop looking now
                            else:
                                prefs[model_idx, pref_idx] = -1  # this one didn't work
                                continue

            # now do the model modifications
            for model_idx in range(ENSEMBLE_SIZE):
                modify_model(
                    model_ensemble[model_idx],
                    new_sample[model_to_sample_pairing[model_idx], :],
                )
        else:
            # sample from KF-learned dist and modify existing models to fit
            for model in model_ensemble:
                state = dist.rvs()
                modify_model(model, state)

################################################################################

vp_full_trajectory = np.array(
    (
        vp_wolf_counts,
        vp_sheep_counts,
        vp_grass_counts,
        [vp_wolf_gain_from_food] * (TIME_SPAN + 1),
        [vp_sheep_gain_from_food] * (TIME_SPAN + 1),
        [vp_wolf_reproduce] * (TIME_SPAN + 1),
        [vp_sheep_reproduce] * (TIME_SPAN + 1),
        [vp_grass_regrowth_time] * (TIME_SPAN + 1),
    )
).T

delta_full = mean_vec - vp_full_trajectory
surprisal_full = np.einsum("ij,ij->i", delta_full, np.linalg.solve(cov_matrix, delta_full))
mean_surprisal_full = np.mean(surprisal_full)

vp_state_trajectory = np.array(
    (
        vp_wolf_counts,
        vp_sheep_counts,
        vp_grass_counts,
    )
).T
delta_state = mean_vec[:, :3] - vp_state_trajectory
surprisal_state = np.einsum(
    "ij,ij->i", delta_state, np.linalg.solve(cov_matrix[:, :3, :3], delta_state)
)
mean_surprisal_state = np.mean(surprisal_state)

vp_param_trajectory = np.array(
    (
        [vp_wolf_gain_from_food] * (TIME_SPAN + 1),
        [vp_sheep_gain_from_food] * (TIME_SPAN + 1),
        [vp_wolf_reproduce] * (TIME_SPAN + 1),
        [vp_sheep_reproduce] * (TIME_SPAN + 1),
        [vp_grass_regrowth_time] * (TIME_SPAN + 1),
    )
).T
delta_param = mean_vec[:, 3:] - vp_param_trajectory
surprisal_param = np.einsum(
    "ij,ij->i", delta_param, np.linalg.solve(cov_matrix[:, 3:, 3:], delta_param)
)
mean_surprisal_param = np.mean(surprisal_param)

if GRAPHS:
    plt.plot(surprisal_full, label="full surprisal")
    plt.plot(surprisal_state, label="state surprisal")
    plt.plot(surprisal_param, label="param surprisal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FILE_PREFIX + f"surprisal.pdf")
    plt.close()

if GRAPHS:
    fig, axs = plt.subplots(4, figsize=(6, 8))
    plural = {"wolf": "wolves", "sheep": "sheep", "grass": "grass"}
    vp_data = {
        "wolf": vp_wolf_counts,
        "sheep": vp_sheep_counts,
        "grass": vp_grass_counts,
    }
    max_scales = {
        "wolf": 10 * mean_init_wolves,
        "sheep": 10 * mean_init_sheep,
        "grass": 10 * mean_init_grass_proportion * GRID_HEIGHT * GRID_WIDTH,
    }
    for idx, state_var_name in enumerate(["wolf", "sheep", "grass"]):
        axs[idx].plot(
            vp_data[state_var_name],
            label="true value",
            color="black",
        )
        axs[idx].plot(
            range(TIME_SPAN + 1),
            mean_vec[:, idx],
            label="estimate",
        )
        axs[idx].fill_between(
            range(TIME_SPAN + 1),
            np.maximum(
                0.0,
                mean_vec[:, idx] - np.sqrt(cov_matrix[:, idx, idx]),
            ),
            np.minimum(
                max_scales[state_var_name],
                mean_vec[:, idx] + np.sqrt(cov_matrix[:, idx, idx]),
            ),
            color="gray",
            alpha=0.35,
        )
        axs[idx].set_title(state_var_name)
        axs[idx].legend()
    axs[3].set_title("surprisal")
    axs[3].plot(surprisal_state, label="state surprisal")
    axs[3].plot(
        [0, TIME_SPAN + 1], [mean_surprisal_state, mean_surprisal_state], ":", color="black"
    )
    fig.tight_layout()
    fig.savefig(FILE_PREFIX + f"match.pdf")
    plt.close(fig)

np.savez_compressed(
    FILE_PREFIX + f"data.npz",
    vp_full_trajectory=vp_full_trajectory,
    mean_vec=mean_vec,
    cov_matrix=cov_matrix,
)
with open(FILE_PREFIX + "meansurprisal.csv", "w") as file:
    csvwriter = csv.writer(file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(["full", "state", "param"])
    csvwriter.writerow([mean_surprisal_full, mean_surprisal_state, mean_surprisal_param])
