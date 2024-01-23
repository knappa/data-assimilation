import numpy as np
import scipy
from an_cockrell import AnCockrellModel
from perlin_noise import PerlinNoise

from consts import UNIFIED_STATE_SPACE_DIMENSION, state_vars, variational_params


def compute_desired_epi_counts(desired_state, model, state_var_indices):
    desired_epithelium = np.rint(
        np.abs(
            desired_state[
                [
                    state_var_indices["empty_epithelium_count"],
                    state_var_indices["healthy_epithelium_count"],
                    state_var_indices["infected_epithelium_count"],
                    state_var_indices["dead_epithelium_count"],
                    state_var_indices["apoptosed_epithelium_count"],
                ]
            ]
        )
    ).astype(int)

    # Since these just samples from a normal distribution, the sampling might request more or less epithelium than
    # there are grid spaces. We try to do our best to match the given distribution
    desired_total_epithelium = np.sum(desired_epithelium)
    num_grid_spaces = model.GRID_WIDTH * model.GRID_HEIGHT
    if desired_total_epithelium != num_grid_spaces:
        # try an integer-approximation of a proportional rescale
        desired_epithelium = np.rint(
            np.abs(
                desired_epithelium
                * (model.GRID_WIDTH * model.GRID_HEIGHT / desired_total_epithelium)
            ),
        ).astype(int)

        desired_total_epithelium = np.sum(desired_epithelium)
        # if that didn't go all the way (b/c e.g. rounding) add/knock off random individuals until it's ok
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
