#!/usr/bin/env python3
# coding: utf-8
import itertools
from typing import Tuple

import an_cockrell
import h5py
import numpy as np
from an_cockrell import AnCockrellModel, EpiType
from scipy.stats.qmc import LatinHypercube
from tqdm.auto import trange

# Compute Moore neighborhood covariance for spatial variables


# # An-Cockrell model reimplementation

# constants
init_inoculum = 100
num_sims = 10_000
num_steps = 2016  # <- full run value

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
    bat_endo_activation=10,
    bat_metabolic_byproduct=2.0,
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
    activated_macro_il8_secretion=1.0,
    activated_macro_il12_secretion=0.5,
    activated_macro_tnf_secretion=1.0,
    activated_macro_il6_secretion=0.4,
    activated_macro_il10_secretion=1.0,
    macro_antiactivation_threshold=5.0,
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


def update_stats(
    model: AnCockrellModel, mean: np.ndarray, cov_mat: np.ndarray, num_samples: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    # omitting the epithelial type which, as a categorical variable, is handled separately
    spatial_vars = [
        model.epithelium_ros_damage_counter,
        model.epi_regrow_counter,
        model.epi_apoptosis_counter,
        model.epi_intracellular_virus,
        model.epi_cell_membrane,
        model.endothelial_activation,
        model.endothelial_adhesion_counter,
        model.extracellular_virus,
        model.P_DAMPS,
        model.ROS,
        model.PAF,
        model.TNF,
        model.IL1,
        model.IL6,
        model.IL8,
        model.IL10,
        model.IL12,
        model.IL18,
        model.IFNg,
        model.T1IFN,
    ]
    block_dim = 5 + len(spatial_vars)

    mean = np.copy(mean)
    cov_mat = np.copy(cov_mat)

    # Welford's online algorithm for mean and covariance calculation. See Knuth Vol 2, pg 232
    for sample_idx, (row_idx, col_idx) in enumerate(
        itertools.product(*map(range, model.geometry)), start=num_samples + 1
    ):
        # assemble the sample as a vector
        sample = np.zeros(shape=(9 * block_dim,), dtype=np.float64)
        for block_idx, (row_delta, col_delta) in enumerate(
            (
                (0, 0),
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                # (0,0) put first
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            )
        ):
            block_row = (row_idx + row_delta) % model.geometry[0]
            block_col = (col_idx + col_delta) % model.geometry[1]
            sample[0 + block_idx * block_dim] = (
                model.epithelium[block_row, block_col] == EpiType.Empty
            )
            sample[1 + block_idx * block_dim] = (
                model.epithelium[block_row, block_col] == EpiType.Healthy
            )
            sample[2 + block_idx * block_dim] = (
                model.epithelium[block_row, block_col] == EpiType.Infected
            )
            sample[3 + block_idx * block_dim] = (
                model.epithelium[block_row, block_col] == EpiType.Dead
            )
            sample[4 + block_idx * block_dim] = (
                model.epithelium[block_row, block_col] == EpiType.Apoptosed
            )
            for idx, spatial_var in enumerate(spatial_vars, 5):
                sample[idx + block_idx * block_dim] = spatial_var[block_row, block_col]

        old_mean = np.copy(mean)
        mean += (sample - mean) / sample_idx
        # use variant formula (mean of two of the standard updates) to
        # increase symmetry in the fp error (1e-18) range
        cov_mat[:, :] += (
            (sample - mean)[:,np.newaxis] * (sample - old_mean)[:,np.newaxis].transpose()
            + (sample - old_mean)[:,np.newaxis] * (sample - mean)[:,np.newaxis].transpose()
        ) / 2.0
        num_samples += 1

    return mean, cov_mat, num_samples


mean = np.zeros(225, dtype=np.float64)
cov_mat = np.zeros((225, 225), dtype=np.float64)
num_samples = 0

with h5py.File("local-nbhd-statistics.hdf5", "w") as f:
    f.create_dataset(
        "mean",
        mean.shape,
        dtype=np.float64,
        data=mean,
    )
    f.create_dataset(
        "cov_mat",
        cov_mat.shape,
        dtype=np.float64,
        data=cov_mat,
    )
    f.create_dataset(
        "num_samples",
        (),
        dtype=np.int64,
        data=num_samples,
    )

lhc = LatinHypercube(len(variational_params))
sample = 1.0 + 0.5 * (lhc.random(n=num_sims) - 0.5)  # between 75% and 125%

# noinspection PyTypeChecker
for sim_idx in trange(num_sims, desc="simulation"):
    # generate a perturbation of the default parameters
    params = default_params.copy()
    pct_perturbation = sample[sim_idx]
    for pert_idx, param in enumerate(variational_params):
        if isinstance(params[param], int):
            params[param] = int(np.round(pct_perturbation[pert_idx] * params[param]))
        else:
            params[param] = float(pct_perturbation[pert_idx] * params[param])

    model = an_cockrell.AnCockrellModel(**params)

    run_mean = np.zeros(225, dtype=np.float64)
    run_cov_mat_unscaled = np.zeros((225, 225), dtype=np.float64)
    run_num_samples = 0

    # noinspection PyTypeChecker
    for step_idx in trange(num_steps):
        model.time_step()
        run_mean, run_cov_mat_unscaled, run_num_samples = update_stats(
            model, run_mean, run_cov_mat_unscaled, run_num_samples
        )

    # combine https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    delta = (mean - run_mean)[:, np.newaxis]
    delta_scale = (
        num_samples
        * run_num_samples
        / (num_samples + run_num_samples)
        / (num_samples + run_num_samples - 1)
    ) if num_samples + run_num_samples > 1 else 1.0
    prev_cov_scale = (num_samples - 1) / (num_samples + run_num_samples - 1)
    new_cov_scale = 1 / (num_samples + run_num_samples - 1)
    cov_mat[:, :] = (
        prev_cov_scale * cov_mat
        + new_cov_scale * run_cov_mat_unscaled
        + (delta @ delta.transpose()) * delta_scale
    )

    # mean[:] = (mean * num_samples + run_mean * run_num_samples)/(num_samples+run_num_samples)
    mean[:] = mean * (num_samples / (num_samples + run_num_samples)) + run_mean * (
        run_num_samples / (num_samples + run_num_samples)
    )

    num_samples += run_num_samples

    print(f"{mean=}")
    print(f"{cov_mat=}")
    print(f"{num_samples=}")

    with h5py.File("local-nbhd-statistics.hdf5", "r+") as f:
        f["mean"][:] = mean
        f["cov_mat"][:, :] = cov_mat
        f["num_samples"][()] = num_samples
