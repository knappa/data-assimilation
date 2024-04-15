from typing import Final, Tuple

import numpy as np
from an_cockrell import AnCockrellModel
from matplotlib import colors
from perlin_noise import PerlinNoise

from consts import UNIFIED_STATE_SPACE_DIMENSION, state_vars, variational_params

################################################################################


def clr_hex(*args) -> str:
    return "#" + "".join(map(lambda n: hex(n)[2:].zfill(2), args))


cmap = colors.ListedColormap(
    [
        clr_hex(0, 0, 0),
        clr_hex(230, 159, 0),
        clr_hex(86, 180, 233),
        clr_hex(0, 158, 115),
        clr_hex(240, 228, 66),
        clr_hex(0, 114, 178),
        clr_hex(213, 94, 0),
        clr_hex(204, 121, 167),
    ]
)

################################################################################


def compute_desired_epi_counts(
    desired_state, model: AnCockrellModel, state_var_indices
):
    num_grid_spaces = model.GRID_WIDTH * model.GRID_HEIGHT

    # lots of cleanup here, to catch corner cases
    desired_epithelium_float = np.clip(
        np.abs(
            np.nan_to_num(
                desired_state[
                    [
                        state_var_indices["empty_epithelium_count"],
                        state_var_indices["healthy_epithelium_count"],
                        state_var_indices["infected_epithelium_count"],
                        state_var_indices["dead_epithelium_count"],
                        state_var_indices["apoptosed_epithelium_count"],
                    ]
                ],
                posinf=num_grid_spaces,
                neginf=-num_grid_spaces,
            )
        ),
        a_min=1e-6,
        a_max=num_grid_spaces,
    )
    # Since these just samples from a normal distribution, the sampling might request more or less epithelium than
    # there are grid spaces. We try to do our best to match the given distribution by scaling the quantities to the
    # grid size. Further, the counts must be rounded to ints.
    desired_epithelium = (
        desired_epithelium_float * (num_grid_spaces / np.sum(desired_epithelium_float))
    ).astype(int)

    desired_total_epithelium = np.sum(desired_epithelium)

    if desired_total_epithelium == 0:
        # leave the simulation alone
        desired_epithelium = np.array(
            [
                model.empty_epithelium_count,
                model.healthy_epithelium_count,
                model.infected_epithelium_count,
                model.dead_epithelium_count,
                model.apoptosed_epithelium_count,
            ]
        )
    elif desired_total_epithelium != num_grid_spaces:
        # if that didn't nail the value (b/c e.g. rounding) add/knock off random individuals until it's ok
        while desired_total_epithelium < num_grid_spaces:
            rand_idx = np.random.randint(len(desired_epithelium))
            desired_epithelium[rand_idx] += 1
            desired_total_epithelium += 1
        while desired_total_epithelium > num_grid_spaces:
            rand_idx = np.random.randint(len(desired_epithelium))
            if desired_epithelium[rand_idx] > 0:
                desired_epithelium[rand_idx] -= 1
                desired_total_epithelium -= 1
    # now we are certain that desired_epithelium holds attainable values

    return desired_epithelium


################################################################################


def smooth_random_field(geometry):
    noise = PerlinNoise()
    return np.abs(
        np.array(
            [
                [noise((i / geometry[0], j / geometry[1])) for j in range(geometry[1])]
                for i in range(geometry[0])
            ]
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


def cov_cleanup(cov_mat: np.ndarray) -> np.ndarray:
    # numerical cleanup: symmetrize and ensure pos def
    cov_mat = np.nan_to_num(cov_mat, copy=True)
    cov_mat[:, :] = (cov_mat + cov_mat.T) / 2.0

    epsilon: Final[float] = 1e-6
    return cov_mat - np.minimum(0.0, np.min(np.diag(cov_mat)) - epsilon)


################################################################################


def gale_shapely_matching(
    *, new_sample: np.ndarray, macro_data: np.ndarray
) -> np.ndarray:
    """
    Compute a pairing of distribution samples to existing macroscale data that minimizes the pairwise distance using
    the Gale-Shapely algorithm. (i.e. stable marriage)

    Usage::
        model_to_sample_pairing = gale_shapely_matching(
            new_sample=new_sample, macro_data=macro_data
        )
    Then the sample that pairs with the model of index `model_idx` is
     `new_sample[model_to_sample_pairing[model_idx], :]`.

    :param new_sample: (N, UNIFIED_STATE_SPACE_DIMENSION) numpy array of samples from the new distribution
    :param macro_data: (N, UNIFIED_STATE_SPACE_DIMENSION) numpy array of samples from the existing distribution
    :return: matching array of model to sample pairings
    """
    ensemble_size = new_sample.shape[0]
    # fill out preference lists for the models
    prefs = np.zeros((ensemble_size, ensemble_size), dtype=np.int64)
    for idx in range(ensemble_size):
        # noinspection PyUnboundLocalVariable
        dists = np.linalg.norm(new_sample - macro_data[idx], axis=1)
        prefs[idx, :] = np.argsort(dists)

    # arrays to record pairings
    model_to_sample_pairing = np.full(ensemble_size, -1, dtype=np.int64)
    sample_to_model_pairing = np.full(ensemble_size, -1, dtype=np.int64)

    all_paired = False
    while not all_paired:
        all_paired = True
        for model_idx in range(ensemble_size):
            if model_to_sample_pairing[model_idx] != -1:
                # skip already paired models
                continue
            # found an unpaired model, find the first thing not yet
            # checked on its preference list
            min_pref_idx = np.argmax(prefs[model_idx, :] >= 0)
            for pref_idx in range(min_pref_idx, ensemble_size):
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
                        model_to_sample_pairing[competitor_model_idx] = (
                            -1
                        )  # free the competitor
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

    return model_to_sample_pairing


################################################################################


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


################################################################################


def slogdet(m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the log of the absolute value of the determinant. Adapted to singular case.

    :param m: (..., M, M) array like
    :return: abs-logdet
    """
    evals = np.linalg.eigvals(m)
    signs = np.prod(np.sign(evals), axis=-1)

    return signs, np.nan_to_num(
        np.sum(np.nan_to_num(np.log(np.maximum(1e-10, np.abs(evals)))), axis=-1)
    )
