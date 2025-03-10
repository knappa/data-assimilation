from copy import deepcopy
from typing import Callable, Final, List, Optional

import numpy as np
from an_cockrell import AnCockrellModel
from attrs import Factory, define, field
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from ..consts import (
    UNIFIED_STATE_SPACE_DIMENSION,
    default_params,
    init_only_params,
    state_var_indices,
    variational_params,
)
from ..transform import transform_intrinsic_to_kf
from ..util import model_macro_data

OBSERVABLES = [
    "P_DAMPS",
    "T1IFN",
    "TNF",
    "IFNg",
    "IL1",
    "IL6",
    "IL8",
    "IL10",
    "IL12",
    "IL18",
    "extracellular_virus",
]


def normalize_dir_name(name: str) -> str:
    if len(name) == 0:
        name = "."
    if not name.endswith("/"):
        name = name + "/"
    return name


@define(kw_only=True)
class PhenotypeKFAnCockrell:
    """

    :param num_phenotypes:
    :param pca_matrix:
    :param phenotype_weight_means:
    :param phenotype_weight_covs:
    :param observation_uncertainty_cov:
    """

    num_phenotypes: int = field(init=True)

    transformed: bool = field(init=True, default=False)
    transform_intrinsic_to_kf: Optional[Callable] = field(init=True, default=None)
    transform_kf_to_intrinsic: Optional[Callable] = field(init=True, default=None)

    # PCA from 2017*41 dimensions to 3 by matrix C (3,2017*41)
    # then each phenotype_i from gaussian with mean m_i (3,) and cov P_i (3,3)
    # P(phi_i | x) = exp( -(1/2) (Cx-m_i)^T P_i^{-1} (Cx-m_i) ) / sum_i P(\phi_i | x)
    pca_matrix: np.ndarray = field(init=True)
    phenotype_weight_means: np.ndarray = field(init=True)
    phenotype_weight_covs: np.ndarray = field(init=True)

    # phenotype distributions
    phenotype_distribution: np.ndarray = field()

    # noinspection PyUnresolvedReferences
    @phenotype_distribution.default
    def phenotype_distribution_default(self):
        return np.full(self.num_phenotypes, 1 / self.num_phenotypes, dtype=np.float64)

    # in the transformed coordinates
    observation_uncertainty_cov: np.ndarray = field()

    # noinspection PyUnresolvedReferences
    @observation_uncertainty_cov.default
    def observation_uncertainty_default(self):
        return np.full(len(OBSERVABLES), 1.0, dtype=np.float64)

    # dimensions: (phenotype, model)
    ensemble: List[List[AnCockrellModel]] = Factory(list)

    microstate_save_directory: Optional[str] = field(default=None, converter=normalize_dir_name)

    # dimensions: (phenotype,ensemble,time,macrostate)
    ensemble_macrostate: np.ndarray = field(init=False)
    # dimensions: (phenotype,time,macrostate)
    ensemble_macrostate_mean: np.ndarray = field(init=False)
    # dimensions: (phenotype,time,macrostate,macrostate)
    ensemble_macrostate_cov: np.ndarray = field(init=False)

    end_time: Final[int] = field(default=2016)
    initial_time: int = -1
    current_time: int = -1
    kf_iteration: int = 0

    def __attrs_post_init__(self):
        assert (
            self.num_phenotypes
            == self.phenotype_weight_means.shape[0]
            == self.phenotype_weight_covs.shape[0]
            == self.phenotype_distribution.shape[0]
        ), "mismatch in number of phenotypes"
        assert (
            self.phenotype_weight_means.shape[1]
            == self.phenotype_weight_covs.shape[1]
            == self.phenotype_weight_covs.shape[2]
        ), "dimension mismatch"

        # check the R matrix (observation uncertainty)
        # if it is a vector, bump it up to a diagonal matrix
        assert 1 <= len(self.observation_uncertainty_cov.shape) <= 2
        if len(self.observation_uncertainty_cov.shape) == 1:
            self.observation_uncertainty_cov = np.diag(self.observation_uncertainty_cov)
        assert (
            self.observation_uncertainty_cov.shape[0] == self.observation_uncertainty_cov.shape[1]
        )

        # TODO: if the per_phenotype_means/covs is only for 1 phenotype, expand it to all N equally

    def initialize_ensemble(self, *, initialization_means, initialization_covs, t: int = 0) -> None:
        """
        Initialize the ensemble. Note that the initialization parameters are, while related, not the same as the state
        parameters.

        :param initialization_means:
        :param initialization_covs:
        :param t: initial time (default 0)
        :return:
        """
        # TODO: initialization uses different parameters than the running state (Ugh) what to do here?
        #  as function params?

        self.ensemble_macrostate = np.zeros(
            (
                self.num_phenotypes,
                self.ensemble_size,
                self.end_time + 1,
                UNIFIED_STATE_SPACE_DIMENSION,
            )
        )
        self.ensemble_macrostate_mean = np.zeros(
            (self.num_phenotypes, self.end_time + 1, UNIFIED_STATE_SPACE_DIMENSION)
        )
        self.ensemble_macrostate_cov = np.zeros(
            (
                self.num_phenotypes,
                self.end_time + 1,
                UNIFIED_STATE_SPACE_DIMENSION,
                UNIFIED_STATE_SPACE_DIMENSION,
            )
        )
        pass

    def project_to(
        self, *, t: int = -1, update_ensemble: bool = False, save_microstate_files: bool = True
    ) -> None:
        """
        Project the ensemble to time t.
        :param t: time (default -1, corresponding to end_time)
        :param update_ensemble: if True, move saved ensemble members to the desired time
        :param save_microstate_files: if True, save microstates to HDF5 files
        :return: None
        """
        assert (
            not save_microstate_files or self.microstate_save_directory is not None
        ), "Cannot save microstates without specifying a directory"

        if t == -1:
            t = self.end_time

        assert t >= self.current_time, "Cannot update time to the past"

        # if we aren't going to actually update the ensemble, do all of this on a copy
        if update_ensemble:
            models = self.ensemble
        else:
            models = deepcopy(self.ensemble)

        log_weights = [
            np.zeros(
                (len(models[phenotype_idx]), t - self.current_time, self.num_phenotypes),
                dtype=np.float64,
            )
            for phenotype_idx in range(self.num_phenotypes)
        ]

        for time_idx in range(self.current_time + 1, t + 1):
            time_inclusion_matrix = np.zeros((2017 * 41, 41), dtype=np.float64)
            time_inclusion_matrix[41 * time_idx : 41 * (time_idx + 1), :] = np.identity(41)
            restricted_time_pca_matrix = self.pca_matrix @ time_inclusion_matrix
            for phenotype_idx in range(self.num_phenotypes):
                for model_idx, model in enumerate(models[phenotype_idx]):
                    model.time_step()
                    self.ensemble_macrostate[phenotype_idx, model_idx, time_idx] = model_macro_data(
                        model
                    )

                    # compute the per-timestep weights
                    pca_reduction = (
                        restricted_time_pca_matrix
                        @ self.ensemble_macrostate[phenotype_idx, model_idx, time_idx]
                    )
                    for phenotype_test_idx in range(self.num_phenotypes):
                        difference_vec = (
                            pca_reduction - self.phenotype_weight_means[phenotype_test_idx, :]
                        )
                        log_weights[phenotype_idx][
                            model_idx, time_idx - self.current_time - 1, phenotype_test_idx
                        ] = -(
                            difference_vec.T
                            @ np.linalg.solve(
                                self.phenotype_weight_covs[phenotype_test_idx, :, :], difference_vec
                            )
                            / 2.0
                        )
                    # normalize it:
                    log_weights[phenotype_idx][model_idx, :] -= logsumexp(
                        log_weights[phenotype_idx][model_idx, :]
                    )
                    if save_microstate_files:
                        model.save(
                            filename=self.microstate_save_directory
                            + (
                                f"phenotype{phenotype_idx}-model{model_idx}-t{time_idx}.hdf5"
                                if update_ensemble
                                else f"phenotype{phenotype_idx}-model{model_idx}-t{time_idx}-projection.hdf5"
                            )
                        )
                # TODO: update per-phenotype means and covs (we might not be able to do bias correction for cov)

    def kf_update(
        self,
        *,
        t: int,
        observation_types: List[str],
        measurements: np.ndarray,
        save_microstate_files: bool = False,
    ) -> None:
        """

        :param t: time
        :param observation_types: list of measured quantities
        :param measurements: values of measured quantities
        :param save_microstate_files: save projected/updated microstates
        :return:
        """
        # TODO: check the time
        assert len(observation_types) == len(measurements)
        dim_observation = len(observation_types)

        if t > self.current_time:
            self.project_to(t=t, update_ensemble=True, save_microstate_files=save_microstate_files)

        # assemble the matrix for the observation model
        H = np.zeros((dim_observation, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64)
        for h_idx, obs_name in enumerate(observation_types):
            H[h_idx, state_var_indices[obs_name]] = 1.0

        # R is the restriction of the observation uncertainty matrix to the observed subset
        R = H @ self.observation_uncertainty_cov @ H.T

        observation = transform_intrinsic_to_kf(measurements)

        v = observation - (H @ self.ensemble_macrostate_mean[t, :])
        S = H @ self.ensemble_macrostate_cov[t, :, :] @ H.T + R
        K = self.ensemble_macrostate_cov[t, :, :] @ H.T @ np.linalg.pinv(S)

        self.ensemble_macrostate_mean[t, :] += K @ v
        # Joseph form update (See e.g. https://www.anuncommonlab.com/articles/how-kalman-filters-work/part2.html)
        A = np.identity(self.ensemble_macrostate_cov.shape[-1]) - K @ H
        self.ensemble_macrostate_cov[t, :, :] = np.nan_to_num(
            A @ self.ensemble_macrostate_cov[t, :, :] @ A.T + K @ R @ K.T
        )

        # some numerical belt-and-suspenders
        self.ensemble_macrostate_cov[t, :, :] = np.nan_to_num(
            (self.ensemble_macrostate_cov[t, :, :] + self.ensemble_macrostate_cov[t, :, :].T) / 2.0
        )
        min_diag = np.min(np.diag(self.ensemble_macrostate_cov[t, :, :]))
        if min_diag <= 0.0:
            self.ensemble_macrostate_cov[t, :, :] += (1e-6 - min_diag) * np.identity(
                self.ensemble_macrostate_cov.shape[-1]
            )

        # TODO: microstate updates
        # TODO: optionally save microstate update

        # update counters
        # previous_time = self.current_time
        self.current_time = t
        self.kf_iteration += 1


####################################################################################################


def model_ensemble_from(means, covariances, ensemble_size):
    """
    Create an ensemble of models from a distribution. Uses init-only
    and variational parameters

    :param means:
    :param covariances:
    :param ensemble_size:
    :return:
    """
    mdl_ensemble = []
    distribution = multivariate_normal(mean=means, cov=covariances, allow_singular=True)
    for _ in range(ensemble_size):
        model_param_dict = default_params.copy()
        sampled_params = np.abs(distribution.rvs())
        # noinspection PyShadowingNames
        for sample_component, parameter_name in zip(
            sampled_params,
            (init_only_params + variational_params),
        ):
            model_param_dict[parameter_name] = (
                round(sample_component)
                if isinstance(default_params[parameter_name], int)
                else sample_component
            )
        # create model for virtual patient
        model = AnCockrellModel(**model_param_dict)
        mdl_ensemble.append(model)

    return mdl_ensemble


####################################################################################################
