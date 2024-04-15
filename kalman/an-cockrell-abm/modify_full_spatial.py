import itertools
from typing import Callable, Iterable, Tuple

import h5py
import numpy as np
from an_cockrell import AnCockrellModel, EndoType, EpiType, epitype_one_hot_encoding
from scipy.optimize import Bounds, OptimizeResult, lsq_linear

from util import compute_desired_epi_counts, smooth_random_field

################################################################################

spatial_vars = [
    "epithelium_ros_damage_counter",  # idx: 5
    "epi_regrow_counter",  # idx: 6
    "epi_apoptosis_counter",  # idx: 7
    "epi_intracellular_virus",  # idx: 8
    "epi_cell_membrane",  # idx: 9
    "endothelial_activation",  # idx: 10
    "endothelial_adhesion_counter",  # idx: 11
    "extracellular_virus",  # idx: 12
    "P_DAMPS",  # idx: 13
    "ROS",  # idx: 14
    "PAF",  # idx: 15
    "TNF",  # idx: 16
    "IL1",  # idx: 17
    "IL6",  # idx: 18
    "IL8",  # idx: 19
    "IL10",  # idx: 20
    "IL12",  # idx: 21
    "IL18",  # idx: 22
    "IFNg",  # idx: 23
    "T1IFN",  # idx: 24
]
block_dim = len(EpiType) + len(spatial_vars)


def quantization_maker(
    *,
    quantization_similarity_loss_weight: float,
    typical_neighborhood_loss_weight: float,
    neighbor_similarity_loss_weight: float,
    spatial_regularization: float,
    statistics_hdf5_file: str,
) -> Callable[[AnCockrellModel, np.ndarray, Iterable[EpiType], int, int], np.ndarray]:
    with h5py.File(statistics_hdf5_file, "r") as h5file:
        mean = h5file["mean"][:]
        cov_mat = h5file["cov_mat"][:, :]
        # num_samples = h5file['num_samples'][()]

    cov_mat_inv = np.linalg.pinv(cov_mat)

    def _quantizer(
        model: AnCockrellModel,
        full_state: np.ndarray,
        available_epitypes: Iterable[EpiType],
        row_idx: int,
        col_idx: int,
    ) -> np.ndarray:
        moore_neighborhood = (
            (0, 0),
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            # (0,0) put first, otherwise lexicographic order
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )

        # assemble the sample as a vector
        sample = np.zeros(shape=(9 * block_dim,), dtype=np.float64)
        for block_idx, (row_delta, col_delta) in enumerate(moore_neighborhood):
            block_row = (row_idx + row_delta) % model.geometry[0]
            block_col = (col_idx + col_delta) % model.geometry[1]
            sample[block_idx * block_dim : (block_idx + 1) * block_dim] = full_state[
                block_row, block_col, :
            ]

        # compute neighborhood state counts, pre-seeded by 1's as a normalization
        neighbor_states = np.full(len(EpiType), 1.0, dtype=np.float64)
        for row_delta, col_delta in moore_neighborhood:
            block_row = (row_idx + row_delta) % model.geometry[0]
            block_col = (col_idx + col_delta) % model.geometry[1]
            neighbor_states[model.epithelium[block_row, block_col]] += 1

        # evaluate the various quantizations
        min_state = sample[:block_dim].copy()
        min_neg_log_likelihood = float("inf")
        variable_minima = np.array(
            [
                0.0 if spatial_var != "endothelial_activation" else -np.inf
                for spatial_var in spatial_vars
            ],
            dtype=np.float64,
        )
        variable_bounds = Bounds(
            lb=variable_minima,
            ub=np.inf,
        )
        second_block_start_idx = len(EpiType)
        third_block_start_idx = block_dim
        for epitype in available_epitypes:
            # setup for quantized state
            quantized_state = sample.copy()
            quantized_state[:second_block_start_idx] = epitype_one_hot_encoding(epitype)

            # find optimal non-quantized values e.g. cytokine levels
            # Goal here is to find a state s which comes in 3 blocks such that
            # 1. the first block is held constant (this is the quantization of the epitype)
            # 2. the second block is allowed to vary
            #    (these are the non-quantized variables at (r,c) )
            # 3. the third block is held constant (these are the combined states of nearby pixels)
            # 4. s minimizes s.T @ cov_mat_inv @ s + λ²||s-s_{prev}||^2 subject to the above
            #    where λ is `spatial_regularization`
            #    the equations below are straightforward to derive, but a little tedious.
            #    Symmetry of cov_mat_inv is used.
            #    Watch out for the - on b! scipy.optimize.lsq_linear minimizes ||Ax-b||^2
            # 5. s is further constrained to be non-negative in (most) coordinates (see `variable_bounds`)
            # noinspection PyPep8Naming
            A = cov_mat_inv[
                second_block_start_idx:third_block_start_idx,
                second_block_start_idx:third_block_start_idx,
            ]
            b = -(
                quantized_state[:second_block_start_idx].T
                @ cov_mat_inv[
                    :second_block_start_idx,
                    second_block_start_idx:third_block_start_idx,
                ]
                + quantized_state[third_block_start_idx:].T
                @ cov_mat_inv[
                    third_block_start_idx:, second_block_start_idx:third_block_start_idx
                ]
            )
            # TODO: explore options (specifically solver options and tolerance)
            # noinspection PyTypeChecker
            result: OptimizeResult = lsq_linear(
                np.vstack([A, spatial_regularization * np.identity(len(spatial_vars))]),
                np.concatenate(
                    [
                        b,
                        spatial_regularization
                        * quantized_state[second_block_start_idx:third_block_start_idx],
                    ]
                ),
                bounds=variable_bounds,
            )
            solution: np.ndarray = result.x
            if not result.success:
                # fallback is to use the existing state (poss. fixed to be within bounds)
                solution[:] = np.maximum(
                    variable_minima,
                    quantized_state[second_block_start_idx:third_block_start_idx],
                )
                print("using fallback approx for spatial variables")
            # round integer fields to integers (fp integers, that is, not ints)
            for idx, spatial_var in enumerate(spatial_vars):
                if np.issubdtype(getattr(model, spatial_var).dtype, np.integer):
                    solution[idx] = np.rint(solution[idx])

            # replace in quantized state
            quantized_state[second_block_start_idx:third_block_start_idx] = solution

            # # Compute the negative log likelihood of this quantization

            # prefer to be close to the natural quantization
            quantized_neg_log_likelihood = quantization_similarity_loss_weight * np.sum(
                (
                    quantized_state[:third_block_start_idx]
                    - sample[:third_block_start_idx]
                )
                ** 2
            )
            # prefer to be a typical local neighborhood
            quantized_neg_log_likelihood += (
                typical_neighborhood_loss_weight
                * (quantized_state - mean)
                @ cov_mat_inv
                @ (quantized_state - mean)
            )
            # prefer to be like one's neighbors
            # omitting an + np.log(np.sum(neighbor_states)) term since it is a per neighborhood constant
            quantized_neg_log_likelihood += neighbor_similarity_loss_weight * (
                -np.log(neighbor_states[epitype])
            )

            # compare to previously explored quantizations
            if quantized_neg_log_likelihood < min_neg_log_likelihood:
                min_state[:] = quantized_state[:block_dim]
                min_neg_log_likelihood = quantized_neg_log_likelihood

        return min_state

    return _quantizer


quantizer = quantization_maker(
    quantization_similarity_loss_weight=1.0,
    typical_neighborhood_loss_weight=0.002,
    neighbor_similarity_loss_weight=1.0,
    spatial_regularization=1.0,
    statistics_hdf5_file="local-nbhd-statistics.hdf5",
)


################################################################################


def dither(
    model: AnCockrellModel,
    new_epi_counts,
) -> Tuple[np.ndarray, np.ndarray]:
    # store spatial distribution of one-hot encoding for epithelial cell type
    state_vecs = np.zeros(shape=(*model.geometry, block_dim), dtype=np.float64)
    state_vecs[:, :, : len(EpiType)] = epitype_one_hot_encoding(model.epithelium)
    for idx, spatial_var in enumerate(spatial_vars, start=len(EpiType)):
        state_vecs[:, :, idx] = getattr(model, spatial_var).astype(np.float64)

    # counts of epi cell types for the incoming model
    prev_epi_count = np.sum(state_vecs[:, :, : len(EpiType)], axis=(0, 1), dtype=int)

    # alter the state vec to be compatible with the macro state encoded by new_epi_counts
    for idx, (new_count, prev_count) in enumerate(zip(new_epi_counts, prev_epi_count)):
        if prev_count == new_count:
            pass
        elif prev_count > 0:
            # if there is some of this type already existing, scale the state to match new macrostate
            state_vecs[:, :, idx] *= new_count / prev_count
        elif new_count > 0:
            # if there isn't any of this type already existing, base the state on a random field.
            rand_field = smooth_random_field(model.geometry)
            state_vecs[:, :, idx] = new_count * rand_field / np.sum(rand_field)
    # ensure that all locations sum to 1
    state_vecs[:, :, : len(EpiType)] += (
        (1 - np.sum(state_vecs[:, :, : len(EpiType)], axis=2)) / len(EpiType)
    )[:, :, np.newaxis]

    newly_set_epi_counts = np.zeros(len(EpiType), dtype=np.int64)

    for row_idx, col_idx in itertools.product(
        range(model.geometry[0]), range(model.geometry[1])
    ):
        # compute which epitypes are available for placement, where available means that we have not yet used
        # up all requested instances.
        available_epitypes = [
            epitype
            for epitype in EpiType
            if newly_set_epi_counts[epitype] < new_epi_counts[epitype]
        ]

        # find the new type
        new_state = quantizer(model, state_vecs, available_epitypes, row_idx, col_idx)
        cell_type: EpiType = EpiType(np.argmax(new_state[: len(EpiType)]))

        # update counts
        newly_set_epi_counts[cell_type] += 1

        # TODO: consistency checks for these cell types (internal virus, etc)

        error = state_vecs[row_idx, col_idx, :] - new_state
        state_vecs[row_idx, col_idx, :] = new_state

        # even weights -> ( (r,c+1), (r+1,c-1), (r+1,c), (r+1,c+1))
        if col_idx == model.geometry[1] - 1:
            if row_idx == model.geometry[0] - 1:
                # last element, do not propagate error
                weights = (0, 0, 0, 0)
            else:
                # right column, do not propagate error to right
                weights = (0, 1 / 2, 1 / 2, 0)
        else:
            if row_idx == model.geometry[0] - 1:
                # top row, do not propagate error down
                weights = (1, 0, 0, 0)
            else:
                # interior
                weights = (1 / 4, 1 / 4, 1 / 4, 1 / 4)

        num_rows = model.geometry[0]
        num_cols = model.geometry[1]
        row_idx_plus = (row_idx + 1) % num_rows
        col_idx_plus = (col_idx + 1) % num_cols
        col_idx_minus = (col_idx - 1) % num_cols

        # (r,c+1)
        # .*X
        # xxx
        state_vecs[row_idx, col_idx_plus, :] += error * weights[0]
        # ensure it sums to 1
        state_vecs[row_idx, col_idx_plus, : len(EpiType)] += (
            1 - np.sum(state_vecs[row_idx, col_idx_plus, : len(EpiType)])
        ) / len(EpiType)

        # (r+1,c-1)
        # .*x
        # Xxx
        state_vecs[
            row_idx_plus,
            col_idx_minus,
            :,
        ] += (
            error * weights[1]
        )
        # ensure it sums to 1
        state_vecs[row_idx_plus, col_idx_minus, : len(EpiType)] += (
            1 - np.sum(state_vecs[row_idx_plus, col_idx_minus, : len(EpiType)])
        ) / len(EpiType)

        # (r+1,c)
        # .*x
        # xXx
        state_vecs[row_idx_plus, col_idx, :] += error * weights[2]
        # ensure it sums to 1
        state_vecs[row_idx_plus, col_idx, : len(EpiType)] += (
            1 - np.sum(state_vecs[row_idx_plus, col_idx, : len(EpiType)])
        ) / len(EpiType)

        # (r+1,c+1)
        # .*x
        # xxX
        state_vecs[
            row_idx_plus,
            col_idx_plus,
            :,
        ] += (
            error * weights[3]
        )
        # ensure it sums to 1
        state_vecs[row_idx_plus, col_idx_plus, : len(EpiType)] += (
            1 - np.sum(state_vecs[row_idx_plus, col_idx_plus, : len(EpiType)])
        ) / len(EpiType)

    return (
        np.argmax(state_vecs[:, :, : len(EpiType)], axis=2),
        state_vecs[:, :, len(EpiType) :],
    )


################################################################################


def rescale_spatial_variables(desired_state, model, state_var_indices):
    """
    Rescale various spatial variables to their desired values

    :param desired_state:
    :param model:
    :param state_var_indices:
    :return: nothing
    """
    # it's worth pointing out that, because of the cleanup-below-a-threshold code,
    # these reset-by-scaling fields may not go to quite the right thing on the next
    # time step. I'm not seeing an easy fix/alternative here, so ¯\_(ツ)_/¯ ?
    # another problem is what to do when you have zero totals, punting here as well
    desired_t1ifn = desired_state[state_var_indices["total_T1IFN"]]
    if model.total_T1IFN > 0:
        model.T1IFN *= desired_t1ifn / model.total_T1IFN
    else:
        random_field = smooth_random_field(model.geometry)
        model.T1IFN[:] = random_field * desired_t1ifn / np.sum(random_field)

    desired_tnf = desired_state[state_var_indices["total_TNF"]]
    if model.total_TNF > 0:
        model.TNF *= desired_tnf / model.total_TNF
    else:
        random_field = smooth_random_field(model.geometry)
        model.TNF[:] = random_field * desired_tnf / np.sum(random_field)

    desired_ifn_g = desired_state[state_var_indices["total_IFNg"]]
    if model.total_IFNg > 0:
        model.IFNg *= desired_ifn_g / model.total_IFNg
    else:
        random_field = smooth_random_field(model.geometry)
        model.IFNg[:] = random_field * desired_ifn_g / np.sum(random_field)

    desired_il1 = desired_state[state_var_indices["total_IL1"]]
    if model.total_IL1 > 0:
        model.IL1 *= desired_il1 / model.total_IL1
    else:
        random_field = smooth_random_field(model.geometry)
        model.IL1[:] = random_field * desired_il1 / np.sum(random_field)

    desired_il6 = desired_state[state_var_indices["total_IL6"]]
    if model.total_IL6 > 0:
        model.IL6 *= desired_il6 / model.total_IL6
    else:
        random_field = smooth_random_field(model.geometry)
        model.IL6[:] = random_field * desired_il6 / np.sum(random_field)

    desired_il8 = desired_state[state_var_indices["total_IL8"]]
    if model.total_IL8 > 0:
        model.IL8 *= desired_il8 / model.total_IL8
    else:
        random_field = smooth_random_field(model.geometry)
        model.IL8[:] = random_field * desired_il8 / np.sum(random_field)

    desired_il10 = desired_state[state_var_indices["total_IL10"]]
    if model.total_IL10 > 0:
        model.IL10 *= desired_il10 / model.total_IL10
    else:
        random_field = smooth_random_field(model.geometry)
        model.IL10[:] = random_field * desired_il10 / np.sum(random_field)

    desired_il12 = desired_state[state_var_indices["total_IL12"]]
    if model.total_IL12 > 0:
        model.IL12 *= desired_il12 / model.total_IL12
    else:
        random_field = smooth_random_field(model.geometry)
        model.IL12[:] = random_field * desired_il12 / np.sum(random_field)

    desired_il18 = desired_state[state_var_indices["total_IL18"]]
    if model.total_IL18 > 0:
        model.IL18 *= desired_il18 / model.total_IL18
    else:
        random_field = smooth_random_field(model.geometry)
        model.IL18[:] = random_field * desired_il18 / np.sum(random_field)

    desired_total_extracellular_virus = desired_state[
        state_var_indices["total_extracellular_virus"]
    ]
    if model.total_extracellular_virus > 0:
        model.extracellular_virus *= (
            desired_total_extracellular_virus / model.total_extracellular_virus
        )
    else:
        model.infect(int(np.rint(desired_total_extracellular_virus)))


################################################################################


def update_macrophage_count(desired_state, model, state_var_indices):
    macro_delta = int(
        np.rint(desired_state[state_var_indices["macro_count"]] - model.macro_count)
    )
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
        num_to_kill = min(-macro_delta, model.num_macros)
        macros_to_kill = np.random.choice(model.num_macros, num_to_kill, replace=False)
        macro_idcs = np.where(model.macro_mask)[0]
        model.macro_mask[macro_idcs[macros_to_kill]] = False
        model.num_macros -= num_to_kill
        assert model.num_macros == np.sum(model.macro_mask)


################################################################################


def update_pmn_count(desired_state, model, state_var_indices, verbose):
    pmn_delta = int(
        np.rint(desired_state[state_var_indices["pmn_count"]] - model.pmn_count)
    )
    if pmn_delta > 0:
        # need more pmns
        # adapted from activated_endo_update
        pmn_spawn_list = list(
            zip(
                *np.where(
                    (
                        model.endothelial_adhesion_counter
                        > model.activated_endo_adhesion_threshold
                    )
                    & (model.endothelial_activation == EndoType.Activated)
                    & (
                        np.random.rand(*model.geometry)
                        < model.activated_endo_pmn_spawn_prob
                    )
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
            if verbose:
                print("Nowhere to put desired pmns")
    elif pmn_delta < 0:
        # need fewer pmns, kill them randomly
        num_to_kill = min(-pmn_delta, model.num_pmns)
        pmns_to_kill = np.random.choice(model.num_pmns, num_to_kill, replace=False)
        pmn_idcs = np.where(model.pmn_mask)[0]
        model.pmn_mask[pmn_idcs[pmns_to_kill]] = False
        model.num_pmns -= num_to_kill
        assert model.num_pmns == np.sum(model.pmn_mask)


################################################################################


def update_nk_count(desired_state, model, state_var_indices):
    nk_delta = int(
        np.rint(desired_state[state_var_indices["nk_count"]] - model.nk_count)
    )
    if nk_delta > 0:
        model.create_nk(number=int(nk_delta))
    elif nk_delta < 0:
        # need fewer nks, kill them randomly
        num_to_kill = min(-nk_delta, model.num_nks)
        nks_to_kill = np.random.choice(model.num_nks, num_to_kill, replace=False)
        nk_idcs = np.where(model.nk_mask)[0]
        model.nk_mask[nk_idcs[nks_to_kill]] = False
        model.num_nks -= num_to_kill
        assert model.num_nks == np.sum(model.nk_mask)


################################################################################


def update_dc_count(desired_state, model, state_var_indices):
    dc_delta = int(
        np.rint(desired_state[state_var_indices["dc_count"]] - model.dc_count)
    )
    if dc_delta > 0:
        for _ in range(dc_delta):
            model.create_dc()
    elif dc_delta < 0:
        # need fewer dcs, kill them randomly
        num_to_kill = min(-dc_delta, model.num_dcs)
        dcs_to_kill = np.random.choice(model.num_dcs, num_to_kill, replace=False)
        dc_idcs = np.where(model.dc_mask)[0]
        model.dc_mask[dc_idcs[dcs_to_kill]] = False
        model.num_dcs -= num_to_kill
        assert model.num_dcs == np.sum(model.dc_mask)


################################################################################


def modify_model(
    model: AnCockrellModel,
    desired_state: np.ndarray,
    *,
    ignore_state_vars: bool = False,
    variational_params,
    state_vars,
    state_var_indices,
    verbose: bool = False,
):
    """
    Modify a model's microstate to fit a given macrostate

    :param model: model instance (encodes microstate)
    :param desired_state: desired macrostate for the model
    :param ignore_state_vars: if True, only alter parameters, not state variables
    :param verbose: if true, print diagnostic messages
    :param state_var_indices:
    :param state_vars:
    :param variational_params:
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

    ################################################################################

    # rescaling both before and after the quantization in order to 1. give the
    # quantizer better incoming information and 2. to improve accuracy of spatial
    # variable estimates
    rescale_spatial_variables(desired_state, model, state_var_indices)
    model.epithelium[:, :], spatial_dither = dither(
        model, compute_desired_epi_counts(desired_state, model, state_var_indices)
    )
    # copy dithered values into model
    for idx, spatial_var in enumerate(spatial_vars):
        sv = getattr(model, spatial_var)
        if np.issubdtype(sv.dtype, np.integer):
            sv[:, :] = spatial_dither[:, :, idx].astype(np.int64)
        else:
            sv[:, :] = spatial_dither[:, :, idx]
    rescale_spatial_variables(desired_state, model, state_var_indices)

    ################################################################################

    # sanity check on updated epithelium TODO: other checks
    for epitype in EpiType:
        if epitype == EpiType.Infected:
            model.epi_intracellular_virus[model.epithelium == EpiType.Infected] = (
                np.maximum(
                    1,
                    model.epi_intracellular_virus[model.epithelium == EpiType.Infected],
                )
            )
        else:
            model.epi_intracellular_virus[model.epithelium == epitype] = 0

    ################################################################################

    desired_total_intracellular_virus = desired_state[
        state_var_indices["total_intracellular_virus"]
    ]
    if model.total_intracellular_virus > 0:
        model.epi_intracellular_virus[:] = np.rint(
            model.epi_intracellular_virus[:]
            * (desired_total_intracellular_virus / model.total_intracellular_virus),
        ).astype(int)
        # ensure that there is at least one virus in each infected cell
        model.epi_intracellular_virus[model.epithelium == EpiType.Infected] = (
            np.maximum(
                1, model.epi_intracellular_virus[model.epithelium == EpiType.Infected]
            )
        )

    model.apoptosis_eaten_counter = int(
        np.rint(desired_state[state_var_indices["apoptosis_eaten_counter"]])
    )  # no internal state here

    update_dc_count(desired_state, model, state_var_indices)

    update_nk_count(desired_state, model, state_var_indices)

    update_pmn_count(desired_state, model, state_var_indices, verbose)

    update_macrophage_count(desired_state, model, state_var_indices)


################################################################################
