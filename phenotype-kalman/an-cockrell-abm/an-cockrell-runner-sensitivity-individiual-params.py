#!/usr/bin/env python3
# coding: utf-8


# https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Two-sample_Kolmogorov%E2%80%93Smirnov_test
import argparse

import an_cockrell
import h5py
import numpy as np
from an_cockrell import EpiType
from tqdm import trange

from consts import default_params, variational_params

parser = argparse.ArgumentParser()
parser.add_argument("param_idx", type=int)
args = parser.parse_args()

# constants
init_inoculum = 100
num_sims = 2_000
num_steps = 2016  # <- full run value

chunk_size = min(num_sims, 100)


param_idx = args.param_idx
assert 0 <= param_idx < len(variational_params)
param_name = variational_params[param_idx]

# for param_idx, param_name in enumerate(variational_params):
for group_num in trange(5, desc="groups"):
    pct_perturbation = [0.8, 0.9, 1.0, 1.1, 1.2][group_num]

    # allocate arrays
    param_list = np.full((num_sims, len(variational_params)), -1, dtype=np.float64)
    total_T1IFN = np.full((num_sims, num_steps), -1, dtype=np.float64)
    total_TNF = np.full((num_sims, num_steps), -1, dtype=np.float64)
    total_IFNg = np.full((num_sims, num_steps), -1, dtype=np.float64)
    total_IL6 = np.full((num_sims, num_steps), -1, dtype=np.float64)
    total_IL1 = np.full((num_sims, num_steps), -1, dtype=np.float64)
    total_IL8 = np.full((num_sims, num_steps), -1, dtype=np.float64)
    total_IL10 = np.full((num_sims, num_steps), -1, dtype=np.float64)
    total_IL12 = np.full((num_sims, num_steps), -1, dtype=np.float64)
    total_IL18 = np.full((num_sims, num_steps), -1, dtype=np.float64)
    total_extracellular_virus = np.full((num_sims, num_steps), -1, dtype=np.float64)
    total_intracellular_virus = np.full((num_sims, num_steps), -1, dtype=np.float64)
    apoptosis_eaten_counter = np.full((num_sims, num_steps), -1, dtype=np.float64)
    infected_epis = np.full((num_sims, num_steps), -1, dtype=np.float64)
    dead_epis = np.full((num_sims, num_steps), -1, dtype=np.float64)
    apoptosed_epis = np.full((num_sims, num_steps), -1, dtype=np.float64)
    system_health = np.full((num_sims, num_steps), -1, dtype=np.float64)

    # create HDF5 file
    hdf5_filename = (
        f"statistics-param-{param_idx}-{param_name}-group-{pct_perturbation}.hdf5"
    )
    with h5py.File(hdf5_filename, "w") as f:
        f.create_dataset(
            "param_list",
            (num_sims, len(variational_params)),
            dtype=np.float64,
            data=param_list,
            chunks=(chunk_size, len(variational_params)),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
            fletcher32=True,
        )
        f.create_dataset(
            "total_T1IFN",
            (num_sims, num_steps),
            dtype=np.float64,
            data=total_T1IFN,
            chunks=(chunk_size, num_steps),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
            fletcher32=True,
        )
        f.create_dataset(
            "total_TNF",
            (num_sims, num_steps),
            dtype=np.float64,
            data=total_TNF,
            chunks=(chunk_size, num_steps),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
            fletcher32=True,
        )
        f.create_dataset(
            "total_IFNg",
            (num_sims, num_steps),
            dtype=np.float64,
            data=total_IFNg,
            chunks=(chunk_size, num_steps),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
            fletcher32=True,
        )
        f.create_dataset(
            "total_IL6",
            (num_sims, num_steps),
            dtype=np.float64,
            data=total_IL6,
            chunks=(chunk_size, num_steps),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
            fletcher32=True,
        )
        f.create_dataset(
            "total_IL1",
            (num_sims, num_steps),
            dtype=np.float64,
            data=total_IL1,
            chunks=(chunk_size, num_steps),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
            fletcher32=True,
        )
        f.create_dataset(
            "total_IL8",
            (num_sims, num_steps),
            dtype=np.float64,
            data=total_IL8,
            chunks=(chunk_size, num_steps),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
            fletcher32=True,
        )
        f.create_dataset(
            "total_IL10",
            (num_sims, num_steps),
            dtype=np.float64,
            data=total_IL10,
            chunks=(chunk_size, num_steps),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
            fletcher32=True,
        )
        f.create_dataset(
            "total_IL12",
            (num_sims, num_steps),
            dtype=np.float64,
            data=total_IL12,
            chunks=(chunk_size, num_steps),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
            fletcher32=True,
        )
        f.create_dataset(
            "total_IL18",
            (num_sims, num_steps),
            dtype=np.float64,
            data=total_IL18,
            chunks=(chunk_size, num_steps),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
            fletcher32=True,
        )
        f.create_dataset(
            "total_extracellular_virus",
            (num_sims, num_steps),
            dtype=np.float64,
            data=total_extracellular_virus,
            chunks=(chunk_size, num_steps),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
            fletcher32=True,
        )
        f.create_dataset(
            "total_intracellular_virus",
            (num_sims, num_steps),
            dtype=np.float64,
            data=total_intracellular_virus,
            chunks=(chunk_size, num_steps),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
            fletcher32=True,
        )
        f.create_dataset(
            "apoptosis_eaten_counter",
            (num_sims, num_steps),
            dtype=np.float64,
            data=apoptosis_eaten_counter,
            chunks=(chunk_size, num_steps),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
            fletcher32=True,
        )
        f.create_dataset(
            "infected_epis",
            (num_sims, num_steps),
            dtype=np.float64,
            data=infected_epis,
            chunks=(chunk_size, num_steps),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
            fletcher32=True,
        )
        f.create_dataset(
            "dead_epis",
            (num_sims, num_steps),
            dtype=np.float64,
            data=dead_epis,
            chunks=(chunk_size, num_steps),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
            fletcher32=True,
        )
        f.create_dataset(
            "apoptosed_epis",
            (num_sims, num_steps),
            dtype=np.float64,
            data=apoptosed_epis,
            chunks=(chunk_size, num_steps),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
            fletcher32=True,
        )
        f.create_dataset(
            "system_health",
            (num_sims, num_steps),
            dtype=np.float64,
            data=system_health,
            chunks=(chunk_size, num_steps),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
            fletcher32=True,
        )
        # end: create HDF5 file

    for sim_idx in trange(num_sims, desc=f"group {group_num} simulations"):
        # generate a perturbation of the default parameters
        params = default_params.copy()

        if isinstance(params[param_name], int):
            params[param_name] = int(round(pct_perturbation * params[param_name], 0))
        else:
            params[param_name] = float(pct_perturbation * params[param_name])

        param_list[sim_idx, :] = np.array(
            [params[param] for param in variational_params]
        )

        model = an_cockrell.AnCockrellModel(**params)

        # noinspection PyTypeChecker
        for step_idx in trange(num_steps):
            model.time_step()

            total_T1IFN[sim_idx, step_idx] = model.total_T1IFN
            total_TNF[sim_idx, step_idx] = model.total_TNF
            total_IFNg[sim_idx, step_idx] = model.total_IFNg
            total_IL6[sim_idx, step_idx] = model.total_IL6
            total_IL1[sim_idx, step_idx] = model.total_IL1
            total_IL8[sim_idx, step_idx] = model.total_IL8
            total_IL10[sim_idx, step_idx] = model.total_IL10
            total_IL12[sim_idx, step_idx] = model.total_IL12
            total_IL18[sim_idx, step_idx] = model.total_IL18
            total_extracellular_virus[sim_idx, step_idx] = (
                model.total_extracellular_virus
            )
            total_intracellular_virus[sim_idx, step_idx] = (
                model.total_intracellular_virus
            )
            apoptosis_eaten_counter[sim_idx, step_idx] = model.apoptosis_eaten_counter
            infected_epis[sim_idx, step_idx] = np.sum(
                model.epithelium == EpiType.Infected
            )
            dead_epis[sim_idx, step_idx] = np.sum(model.epithelium == EpiType.Dead)
            apoptosed_epis[sim_idx, step_idx] = np.sum(
                model.epithelium == EpiType.Apoptosed
            )
            system_health[sim_idx, step_idx] = model.system_health

        with h5py.File(hdf5_filename, "r+") as f:
            f["param_list"][sim_idx, :] = param_list[sim_idx, :]
            f["total_T1IFN"][sim_idx, :] = total_T1IFN[sim_idx, :]
            f["total_TNF"][sim_idx, :] = total_TNF[sim_idx, :]
            f["total_IFNg"][sim_idx, :] = total_IFNg[sim_idx, :]
            f["total_IL6"][sim_idx, :] = total_IL6[sim_idx, :]
            f["total_IL1"][sim_idx, :] = total_IL1[sim_idx, :]
            f["total_IL8"][sim_idx, :] = total_IL8[sim_idx, :]
            f["total_IL10"][sim_idx, :] = total_IL10[sim_idx, :]
            f["total_IL12"][sim_idx, :] = total_IL12[sim_idx, :]
            f["total_IL18"][sim_idx, :] = total_IL18[sim_idx, :]
            f["total_extracellular_virus"][sim_idx, :] = total_extracellular_virus[
                sim_idx, :
            ]
            f["total_intracellular_virus"][sim_idx, :] = total_intracellular_virus[
                sim_idx, :
            ]
            f["apoptosis_eaten_counter"][sim_idx, :] = apoptosis_eaten_counter[
                sim_idx, :
            ]
            f["infected_epis"][sim_idx, :] = infected_epis[sim_idx, :]
            f["dead_epis"][sim_idx, :] = dead_epis[sim_idx, :]
            f["apoptosed_epis"][sim_idx, :] = apoptosed_epis[sim_idx, :]
            f["system_health"][sim_idx, :] = system_health[sim_idx, :]
