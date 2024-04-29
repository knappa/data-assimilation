import numpy as np

__EPSILON__ = 1e-3

def transform_intrinsic_to_kf(macrostate_intrinsic: np.ndarray, *, index=-1) -> np.ndarray:
    """
    Transform an intrinsic macrostate to a normalized one for the KF.

    :param macrostate_intrinsic: intrinsic macrostate
    :param index: which index to transform, for arrays with single components
    :return: normalized macrostate for kf
    """
    if index == -1:
        # full state
        retval = np.zeros_like(macrostate_intrinsic)
        retval[..., 0] = np.log(__EPSILON__ + macrostate_intrinsic[..., 0])
        retval[..., 1] = macrostate_intrinsic[..., 1] / 10
        retval[..., 2] = macrostate_intrinsic[..., 2] / 100
        retval[..., 3:] = macrostate_intrinsic[..., 3:]
        return retval
    elif index == 0:
        # wolves
        return np.log(__EPSILON__ + macrostate_intrinsic)
    elif index == 1:
        # sheep
        return macrostate_intrinsic / 10
    elif index == 2:
        # grass
        return macrostate_intrinsic / 100
    else:
        # parameters
        return macrostate_intrinsic


def transform_kf_to_intrinsic(macrostate_kf: np.ndarray, *, index=-1) -> np.ndarray:
    """
    Transform a normalized macrostate to an intrinsic one.

    :param macrostate_kf: normalized macrostate for kf
    :param index: which index to transform, for arrays with single components
    :return: intrinsic macrostate
    """
    if index == -1:
        # full state
        retval = np.zeros_like(macrostate_kf)
        retval[..., 0] = np.maximum(0.0, np.exp(macrostate_kf[..., 0]) - __EPSILON__)
        retval[..., 1] = np.maximum(0.0, macrostate_kf[..., 1]) * 10
        retval[..., 2] = np.maximum(0.0, macrostate_kf[..., 2]) * 100
        retval[..., 3:] = np.maximum(0.0, macrostate_kf[..., 3:])
        return retval
    elif index == 0:
        # wolves
        return np.maximum(0.0, np.exp(macrostate_kf) - __EPSILON__)
    elif index == 1:
        # sheep
        return np.maximum(0.0, macrostate_kf) * 10
    elif index == 2:
        # grass
        return np.maximum(0.0, macrostate_kf) * 100
    else:
        # parameters
        return np.maximum(0.0, macrostate_kf)
