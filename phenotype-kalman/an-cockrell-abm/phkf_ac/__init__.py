from copy import deepcopy
from typing import Callable, Final, List, Optional

import numpy as np
import scipy
from an_cockrell import AnCockrellModel
from attrs import Factory, define, field
from numpy.linalg.linalg import LinAlgError
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from ..consts import (
    UNIFIED_STATE_SPACE_DIMENSION,
    default_params,
    init_only_params,
    state_var_indices,
    state_vars,
    variational_params,
)
from ..modify_epi_spatial import modify_model
from ..transform import transform_intrinsic_to_kf, transform_kf_to_intrinsic
from ..util import gale_shapely_matching, model_macro_data, slogdet

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
    log_phenotype_distribution: np.ndarray = field()

    # noinspection PyUnresolvedReferences
    @log_phenotype_distribution.default
    def phenotype_distribution_default(self):
        return np.log(np.full(self.num_phenotypes, 1 / self.num_phenotypes, dtype=np.float64))

    # in the transformed coordinates
    observation_uncertainty_cov: np.ndarray = field()

    # noinspection PyUnresolvedReferences
    @observation_uncertainty_cov.default
    def observation_uncertainty_default(self):
        return np.full(len(OBSERVABLES), 1.0, dtype=np.float64)

    # dimensions: (phenotype, model)
    ensemble: List[List[AnCockrellModel]] = Factory(list)
    ensemble_size: int = field(default=1 + 2 * UNIFIED_STATE_SPACE_DIMENSION)

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
            == self.log_phenotype_distribution.shape[0]
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

    def project_ensemble_to(
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

        for time_idx in range(self.current_time + 1, t + 1):
            for phenotype_idx in range(self.num_phenotypes):
                for model_idx, model in enumerate(models[phenotype_idx]):
                    model.time_step()
                    self.ensemble_macrostate[phenotype_idx, model_idx, time_idx, :] = (
                        model_macro_data(model)
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

    def kf_update(
        self,
        *,
        observation_time: int,
        observation_types: List[str],
        measurements: np.ndarray,
        save_microstate_files: bool = False,
    ) -> None:
        """
        Perform the KF update to the ensemble.

        :param observation_time: time of observation
        :param observation_types: list of measured quantities
        :param measurements: values of measured quantities
        :param save_microstate_files: save projected/updated microstates
        :return: None
        """
        assert len(observation_types) == len(measurements)
        dim_observation = len(observation_types)
        assert observation_time >= self.current_time, "KF update cannot work backward in time"
        if observation_time == self.current_time:
            raise NotImplementedError(
                "Multiple same-time observations split between different function calls are unsupported"
            )

        delta_time: int = observation_time - self.current_time

        self.project_ensemble_to(
            t=observation_time,
            update_ensemble=True,
            save_microstate_files=save_microstate_files,
        )

        # ########## Step 1: assemble the predictive distributions from the ensemble macrostates from the previous
        # observation time to the new observation time

        # Step 1a. Determine weights
        time_inclusion_matrix = np.zeros((2017 * 41, delta_time * 41), dtype=np.float64)
        for time_idx in range(self.current_time, observation_time + 1):
            time_inclusion_matrix[
                41 * (self.current_time + time_idx) : 41 * (self.current_time + time_idx + 1),
                41 * time_idx : 41 * (time_idx + 1),
            ] = np.identity(41)
        restricted_time_pca_matrix = self.pca_matrix @ time_inclusion_matrix
        reduced_states = np.einsum(
            "ij,pmj->pmi",
            restricted_time_pca_matrix,  # [3,41*dt]
            self.ensemble_macrostate.reshape(
                (self.num_phenotypes, self.ensemble_size, -1)
            ),  # [p,m,t,s] -> [p,m,j]
        )
        log_weights = np.zeros(
            (self.num_phenotypes, self.ensemble_size),
            dtype=np.float64,
        )
        # TODO: unscented weights?
        for phenotype_idx in range(self.num_phenotypes):
            mean_difference_vec = reduced_states - self.phenotype_weight_means[phenotype_idx]
            try:
                temp = np.linalg.lstsq(
                    self.phenotype_weight_covs[phenotype_idx],
                    mean_difference_vec,
                )[0]
            except LinAlgError:
                temp = (
                    scipy.linalg.pinvh(self.phenotype_weight_covs[phenotype_idx], return_rank=False)
                    @ mean_difference_vec
                )
            log_weights[phenotype_idx, :] = (
                -(
                    mean_difference_vec @ temp
                    + slogdet(self.phenotype_weight_covs[phenotype_idx])[1]
                )
                / 2.0
            )
        # normalize weights
        for ensemble_idx in range(log_weights.shape[1]):
            log_weights[:, ensemble_idx] -= logsumexp(log_weights[:, ensemble_idx])

        # compute weighted per-phenotype and per-time mean and covariance
        for phenotype_idx in range(self.num_phenotypes):
            weights = np.exp(log_weights[phenotype_idx, :])
            self.ensemble_macrostate_mean[phenotype_idx, :, :] = np.average(
                self.ensemble_macrostate[phenotype_idx, :, :, :], weights=weights, axis=0
            )
            for time_idx in range(self.current_time + 1, observation_time + 1):
                self.ensemble_macrostate_cov[phenotype_idx, time_idx, :, :] = np.cov(
                    self.ensemble_macrostate[phenotype_idx, :, time_idx, :],
                    aweights=weights,
                    rowvar=False,
                )

        # ######### Step 2: perform the Kalman update from the observation, per phenotype

        # Step 2a. Compute some generally useful stuff

        # assemble the matrix for the observation model
        H = np.zeros((dim_observation, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64)
        for h_idx, obs_name in enumerate(observation_types):
            H[h_idx, state_var_indices[obs_name]] = 1.0

        # R is the restriction of the observation uncertainty matrix to the observed subset
        R = H @ self.observation_uncertainty_cov @ H.T

        observation = transform_intrinsic_to_kf(measurements)

        ident_matrix = np.identity(UNIFIED_STATE_SPACE_DIMENSION)

        # Step 2b. Kalman update for each phenotype

        for phenotype_idx in range(self.num_phenotypes):
            mu = self.ensemble_macrostate_mean[phenotype_idx, observation_time, :]
            P = self.ensemble_macrostate_cov[phenotype_idx, observation_time, :, :]

            v = observation - (H @ mu)
            S = H @ P @ H.T + R

            self.log_phenotype_distribution[phenotype_idx] += (
                -(v.T @ scipy.linalg.pinvh(S) @ v - slogdet(S)[1]) / 2
            )

            K = P @ H.T @ np.linalg.pinv(S)

            # Note for future readers: mu[:] and P[:,:] access the the contents of the original arrays that mu and P
            # are derived from. If you replaced this with mu and P, you'd just get new arrays and the originals would
            # be unchanged.
            mu[:] += K @ v
            # Joseph form update (See e.g. https://www.anuncommonlab.com/articles/how-kalman-filters-work/part2.html)
            A = ident_matrix - K @ H
            P[:, :] = np.nan_to_num(A @ P @ A.T + K @ R @ K.T)

            # Make sure that the covariance matrix is on-the-nose symmetric
            P[:, :] = np.nan_to_num((P + P.T) / 2.0)
            # Make sure that the covariance matrix is positive definite
            min_diag = np.min(np.diag(P))
            if min_diag <= 1e-6:
                P[:, :] += (1e-6 - min_diag) * ident_matrix

        # Step 2c: Normalize overall phenotype probabilities
        self.log_phenotype_distribution -= logsumexp(self.log_phenotype_distribution)

        # ######### Step 3: As in the unscented kalman filter, find the + of the new Gaussian's and best matching to
        # the ensemble members

        ensemble_target_locs = np.zeros(
            (
                self.num_phenotypes,
                2 * UNIFIED_STATE_SPACE_DIMENSION + 1,
                UNIFIED_STATE_SPACE_DIMENSION,
            ),
            dtype=np.float64,
        )
        model_to_sample_pairing = np.zeros(
            (self.num_phenotypes, 2 * UNIFIED_STATE_SPACE_DIMENSION + 1), dtype=np.intp
        )
        for phenotype_idx in range(self.num_phenotypes):
            ensemble_target_locs[phenotype_idx, 0, :] = self.ensemble_macrostate_mean[
                phenotype_idx, observation_time
            ]
            U, S, Vh = np.linalg.svd(
                self.ensemble_macrostate_cov[phenotype_idx, observation_time, :, :],
                hermitian=True,
                compute_uv=True,
                full_matrices=True,
            )
            principal_axes = U * np.sqrt(S)  # TODO: proper weights
            for axis_idx, axis in enumerate(principal_axes):
                ensemble_target_locs[phenotype_idx, 2 * axis + 1, :] = axis
                ensemble_target_locs[phenotype_idx, 2 * axis + 2, :] = -axis

            model_to_sample_pairing[phenotype_idx, :] = gale_shapely_matching(
                new_sample=ensemble_target_locs[phenotype_idx],
                macro_data=self.ensemble_macrostate[phenotype_idx, observation_time],
            )

        # ######### Step 4: Microstate synthesis

        for phenotype_idx in range(self.num_phenotypes):
            for model_idx in range(2 * UNIFIED_STATE_SPACE_DIMENSION + 1):
                modify_model(
                    self.ensemble[phenotype_idx][model_idx],
                    transform_kf_to_intrinsic(
                        ensemble_target_locs[
                            phenotype_idx, model_to_sample_pairing[phenotype_idx, model_idx], :
                        ]
                    ),
                    verbose=True,  # TODO: remember to revisit
                    state_var_indices=state_var_indices,
                    state_vars=state_vars,
                    variational_params=variational_params,
                )

        # TODO: optionally save microstate update

        # ######### Step 5: Cleanup

        # update counters
        # previous_time = self.current_time
        self.current_time = observation_time
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
