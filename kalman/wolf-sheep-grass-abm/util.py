from typing import Tuple

import numpy as np


def random_walk_covariance(macrostate, *, param_stoch_level: float = 0.01):
    """
    Construct a covariance matrix for the random walk in parameter space.

    :param macrostate:
    :param param_stoch_level:
    :return:
    """
    return np.diag([0.01] + list(param_stoch_level * np.abs(macrostate[1:])))


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
