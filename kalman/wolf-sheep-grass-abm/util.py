from typing import Tuple

import numpy as np


def random_walk_covariance(macrostate):
    # TODO: tuning
    return np.diag(0.01 * macrostate)


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
