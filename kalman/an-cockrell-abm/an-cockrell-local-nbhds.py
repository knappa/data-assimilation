#!/usr/bin/env python3
# coding: utf-8

# Compute Moore neighborhood covariance for spatial variables

import itertools
from typing import Tuple

import an_cockrell
import h5py
import numpy as np
from an_cockrell import AnCockrellModel, EpiType
from scipy.stats.qmc import LatinHypercube
from tqdm.auto import trange

from consts import default_params, variational_params

# constants
init_inoculum = 100
num_sims = 500
num_steps = 2016  # <- full run value


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
            (sample - mean)[:, np.newaxis]
            * (sample - old_mean)[:, np.newaxis].transpose()
            + (sample - old_mean)[:, np.newaxis]
            * (sample - mean)[:, np.newaxis].transpose()
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
        (
            num_samples
            * run_num_samples
            / (num_samples + run_num_samples)
            / (num_samples + run_num_samples - 1)
        )
        if num_samples + run_num_samples > 1
        else 1.0
    )
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
