import numpy as np


def random_walk_covariance(macrostate):
    # TODO: tuning
    return np.diag([0.01] + list(0.01 * np.abs(macrostate[1:])))
