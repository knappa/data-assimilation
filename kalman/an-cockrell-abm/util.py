import numpy as np
import scipy
from an_cockrell import AnCockrellModel
from perlin_noise import PerlinNoise

from consts import UNIFIED_STATE_SPACE_DIMENSION, state_vars, variational_params


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


def cov_cleanup(cov_mat: np.ndarray) -> np.ndarray:
    # numerical cleanup: symmetrize and project onto pos def cone
    cov_mat = np.nan_to_num(cov_mat, copy=True)
    cov_mat = (cov_mat + cov_mat.T) / 2.0

    eigenvalues, eigenvectors = scipy.linalg.eigh(
        cov_mat, lower=True, check_finite=False
    )
    eigenvalues[:] = np.real(eigenvalues)  # just making sure
    eigenvectors[:, :] = np.real(eigenvectors)  # just making sure
    # spectrum must be positive.
    # from the scipy code, it also can't have a max/min e-val ratio bigger than 1/(1e6*double machine epsilon)
    # and that's ~4503599627.370496=1/(1e6*np.finfo('d').eps), so a ratio bounded by 1e9 is ok.
    cov_mat = (
        eigenvectors
        @ np.diag(np.minimum(1e5, np.maximum(1e-4, eigenvalues)))
        @ eigenvectors.T
    )
    return np.nan_to_num((cov_mat + cov_mat.T) / 2.0)
