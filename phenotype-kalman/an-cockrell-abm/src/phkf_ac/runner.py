from copy import deepcopy
from typing import Callable, Final, List, Optional, Tuple

import numpy as np
import scipy
from an_cockrell import AnCockrellModel
from attrs import define, field
from numpy.linalg import LinAlgError
from scipy.special import logsumexp

from phkf_ac.consts import (
    UNIFIED_STATE_SPACE_DIMENSION,
    default_params,
    init_only_params,
    state_var_indices,
    state_vars,
    variational_params,
)
from phkf_ac.util import abslogdet, gale_shapely_matching, model_macro_data

__eigenvalue_epsilon__: float = 1e-6


def normalize_dir_name(name: str) -> str:
    if name is None or len(name) == 0:
        name = "."
    if not name.endswith("/"):
        name = name + "/"
    return name


@define(kw_only=True, slots=True)
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
    transform_intrinsic_to_kf: Optional[Callable] = field(init=True)
    transform_kf_to_intrinsic: Optional[Callable] = field(init=True)

    # PCA from 2017*40 dimensions to 3 by matrix C (3,2017*40) with center m (2017*40,)
    # then each phenotype_i from gaussian with mean m_i (3,) and cov P_i (3,3)
    # P(phi_i | x) = exp( -(1/2) (C(x-m)-m_i)^T P_i^{-1} (C(x-m)-m_i) ) / sum_i P(\phi_i | x)
    pca_center: np.ndarray = field(init=True)
    pca_matrix: np.ndarray = field(init=True)
    phenotype_weight_means: np.ndarray = field(init=True)
    phenotype_weight_covs: np.ndarray = field(init=True)

    # phenotype distributions
    log_phenotype_distribution: np.ndarray = field()

    # noinspection PyUnresolvedReferences
    @log_phenotype_distribution.default
    def log_phenotype_distribution_default(self):
        return np.log(np.full(self.num_phenotypes, 1 / self.num_phenotypes, dtype=np.float64))

    # (time, phenotype)
    log_phenotype_distribution_timeseries: np.ndarray = field(init=False)

    # in transformed coordinates
    observation_uncertainty_cov: np.ndarray = field()

    # noinspection PyUnresolvedReferences
    @observation_uncertainty_cov.default
    def observation_uncertainty_default(self):
        return np.full(UNIFIED_STATE_SPACE_DIMENSION, 0.1, dtype=np.float64)

    # dimensions: (phenotype, model)
    ensemble: List[List[AnCockrellModel]] = field(init=False, factory=list)
    ensemble_size: int = field(init=False)

    microstate_save_directory: Optional[str] = field(default=None, converter=normalize_dir_name)

    # dimensions: (phenotype,ensemble,time,macrostate)
    ensemble_macrostate: np.ndarray = field(init=False)
    # dimensions: (phenotype,time,macrostate)
    ensemble_macrostate_mean: np.ndarray = field(init=False)
    # dimensions: (phenotype,time,macrostate,macrostate)
    ensemble_macrostate_cov: np.ndarray = field(init=False)

    model_modification_algorithm = field()  # one of the modify_model functions

    end_time: Final[int] = field(default=2016)
    _initial_time: int = field(default=-1, init=False)
    _current_time: int = field(default=-1, init=False)
    _kf_iteration: int = field(default=0, init=False)

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
        ), "observation uncertainty is not square!"

        assert (
            self.observation_uncertainty_cov.shape[0] == UNIFIED_STATE_SPACE_DIMENSION
        ), "observation uncertainty matrix is not the correct size"

        self.ensemble_size = 1 + 2 * UNIFIED_STATE_SPACE_DIMENSION

        # TODO: if the per_phenotype_means/covs is only for 1 phenotype, expand it to all N equally

    def initialize_ensemble(
        self, *, initialization_means: np.ndarray, initialization_covs: np.ndarray, t: int = 0
    ) -> None:
        """
        Initialize the ensemble. Note that the initialization parameters are, while related, not the same as the state
        parameters.

        :param initialization_means:
        :param initialization_covs:
        :param t: initial time (default 0)
        :return:
        """
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

        self.log_phenotype_distribution_timeseries = np.full(
            (
                self.num_phenotypes,
                self.end_time + 1,
            ),
            self.log_phenotype_distribution[:, None],
        )

        # TODO: consider per-phenotype initialization (Note that these aren't quite the same parameters, so this can't
        #  just be read from the existing data.)
        for phenotype_idx in range(self.num_phenotypes):
            self.ensemble.append(
                model_sigma_point_ensemble_from(
                    initialization_means, initialization_covs, 1 + 2 * UNIFIED_STATE_SPACE_DIMENSION
                )
            )
            for model_idx in range(self.ensemble_size):
                self.ensemble_macrostate[phenotype_idx, model_idx, 0, :] = (
                    self.transform_intrinsic_to_kf(
                        model_macro_data(self.ensemble[phenotype_idx][model_idx])
                    )
                )

            self.ensemble_macrostate_mean[phenotype_idx, 0, :] = np.mean(
                self.ensemble_macrostate[phenotype_idx, :, 0, :], axis=0
            )
            self.ensemble_macrostate_cov[phenotype_idx, 0, :, :] = np.cov(
                self.ensemble_macrostate[phenotype_idx, :, 0, :],
                rowvar=False,
            )

        self._current_time = t

    def project_ensemble_to(
        self, *, t: int = -1, update_ensemble: bool = False, save_microstate_files: bool = False
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

        assert t >= self._current_time, "Cannot update time to the past"

        # if we aren't going to actually update the ensemble, do all of this on a copy
        if update_ensemble:
            models = self.ensemble
        else:
            models = deepcopy(self.ensemble)

        for phenotype_idx in range(self.num_phenotypes):
            for model_idx, model in enumerate(models[phenotype_idx]):
                for time_idx in range(self._current_time + 1, t + 1):
                    model.time_step()
                    self.ensemble_macrostate[phenotype_idx, model_idx, time_idx, :] = (
                        self.transform_intrinsic_to_kf(model_macro_data(model))
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
            # TODO: use weighing?
            self.ensemble_macrostate_mean[phenotype_idx, :, :] = np.mean(
                self.ensemble_macrostate[phenotype_idx, :, :, :], axis=0
            )
            for time_idx in range(self._current_time + 1, t + 1):
                self.ensemble_macrostate_cov[phenotype_idx, time_idx, :, :] = np.cov(
                    self.ensemble_macrostate[phenotype_idx, :, time_idx, :],
                    rowvar=False,
                )

        if update_ensemble:
            self._current_time = t

    def kf_update(
        self,
        *,
        observation_time: int,
        observation_types: List[str],
        measurements: np.ndarray,
        previous_observation_time: int = -1,
        save_microstate_files: bool = False,
        log: bool = True,
    ) -> None:
        """
        Perform the KF update to the ensemble.

        :param observation_time: time of observation
        :param observation_types: list of measured quantities
        :param measurements: values of measured quantities
        :param previous_observation_time: if the ensemble was updated elsewhere, give the prev. obs. time.
        Otherwise, the ensemble's current time will be used.
        :param save_microstate_files: save projected/updated microstates
        :param log: print diagnostic messages
        :return: None
        """
        from phkf_ac.util import pos_def_matrix_cleanup

        assert len(observation_types) == len(measurements)
        dim_observation = len(observation_types)
        assert observation_time >= self._current_time, "KF update cannot work backward in time"

        if log:
            print(f"Projecting ensemble to t={observation_time}", flush=True)

        if previous_observation_time < 0:
            previous_observation_time = self._current_time
            self.project_ensemble_to(
                t=observation_time,
                update_ensemble=True,
                save_microstate_files=save_microstate_files,
            )

        if log:
            print(f"Projecting ensemble to t={observation_time} (Finished)", flush=True)

        # ########## Step 1: assemble the predictive distributions from the ensemble macrostates from the previous
        # ########## observation time to the new observation time

        # Step 1a. Determine weights
        if log:
            print(f"Determining weights", end="", flush=True)

        # center the trajectories on the PCA's mean
        trajectory_region = (
            self.ensemble_macrostate.reshape(self.num_phenotypes, self.ensemble_size, -1)
            - self.pca_center
        ).reshape(
            self.num_phenotypes,
            self.ensemble_size,
            self.end_time + 1,
            UNIFIED_STATE_SPACE_DIMENSION,
        )
        # zero out values before and after the relevant time region
        trajectory_region[:, :, :previous_observation_time, :] = 0
        trajectory_region[:, :, observation_time + 1 :, :] = 0

        # compute reduction to pca space (3d)
        reduced_states = np.einsum(
            "ij,pmj->pmi",
            self.pca_matrix,  # [3,UNIFIED_STATE_SPACE_DIMENSION*dt]
            trajectory_region.reshape(
                self.num_phenotypes, self.ensemble_size, -1
            ),  # [p,m,t,s] -> [p,m,j]
        )
        # print(f"{reduced_states.shape=}") # (4,81,3)

        # zero out values past the initial conditions to get initial condition
        trajectory_region[:, :, previous_observation_time + 1 :, :] = 0

        # compute initial condition's reduction to pca space (3d)
        reduced_states_init = np.einsum(
            "ij,pmj->pmi",
            self.pca_matrix,  # [3,UNIFIED_STATE_SPACE_DIMENSION*dt]
            trajectory_region.reshape(
                self.num_phenotypes, self.ensemble_size, -1
            ),  # [p,m,t,s] -> [p,m,j]
        )
        # print(f"{reduced_states.shape=}") # (4,81,3)

        log_weights = np.zeros(
            (self.num_phenotypes, self.ensemble_size),
            dtype=np.float64,
        )
        # TODO: unscented weights?
        for ensemble_phenotype_idx in range(self.num_phenotypes):
            # iterate over each member of a phenotype's ensemble
            for ensemble_idx in range(2 * UNIFIED_STATE_SPACE_DIMENSION + 1):
                per_phenotype_log_weight = np.zeros(self.num_phenotypes, dtype=np.float64)
                per_phenotype_log_weight_initial = np.zeros(self.num_phenotypes, dtype=np.float64)
                # iterate over phenotypes
                for phenotype_idx in range(self.num_phenotypes):
                    # log-probabilities for the initial (x_{k-1}) state of the trajectory
                    initial_state_vec = (
                        reduced_states_init[ensemble_phenotype_idx, ensemble_idx]
                        - self.phenotype_weight_means[phenotype_idx]
                    )
                    try:
                        temp = np.linalg.lstsq(
                            self.phenotype_weight_covs[phenotype_idx],
                            initial_state_vec,
                        )[0]
                    except LinAlgError:
                        temp = (
                            scipy.linalg.pinvh(
                                self.phenotype_weight_covs[phenotype_idx], return_rank=False
                            )
                            @ initial_state_vec
                        )
                    # noinspection PyCallingNonCallable
                    per_phenotype_log_weight_initial[phenotype_idx] = -0.5 * (
                        initial_state_vec @ temp
                        + abslogdet(self.phenotype_weight_covs[ensemble_phenotype_idx])
                        + np.log(2 * np.pi) * UNIFIED_STATE_SPACE_DIMENSION
                    )

                    # log-probabilities for the trajectory from the initial state, x_{k-1}, to x_{k+l}
                    trajectory_vec = (
                        reduced_states[ensemble_phenotype_idx, ensemble_idx]
                        - self.phenotype_weight_means[phenotype_idx]
                    )
                    try:
                        temp = np.linalg.lstsq(
                            self.phenotype_weight_covs[phenotype_idx],
                            trajectory_vec,
                        )[0]
                    except LinAlgError:
                        temp = (
                            scipy.linalg.pinvh(
                                self.phenotype_weight_covs[phenotype_idx], return_rank=False
                            )
                            @ trajectory_vec
                        )
                    # noinspection PyCallingNonCallable
                    per_phenotype_log_weight[phenotype_idx] = -0.5 * (
                        trajectory_vec @ temp
                        + abslogdet(self.phenotype_weight_covs[ensemble_phenotype_idx])
                        + np.log(2 * np.pi) * UNIFIED_STATE_SPACE_DIMENSION
                    )

                # compute weights with normalization
                log_weights[ensemble_phenotype_idx, ensemble_idx] = (
                    per_phenotype_log_weight[ensemble_phenotype_idx]
                    - per_phenotype_log_weight_initial[ensemble_phenotype_idx]
                    + logsumexp(
                        per_phenotype_log_weight_initial
                        + self.log_phenotype_distribution_timeseries[:, 0]
                    )
                    - logsumexp(
                        per_phenotype_log_weight + self.log_phenotype_distribution_timeseries[:, 0]
                    )
                )

            # normalize weights in the ensemble
            log_weights[ensemble_phenotype_idx, :] -= logsumexp(
                log_weights[ensemble_phenotype_idx, :]
            )
            # we don't want to weights to become excessively small as this can give a singular estimate
            # (can happen when one trajectory looks like a _really_ good match) so we cap and renormalize
            log_weights[ensemble_phenotype_idx, :] = np.clip(
                log_weights[ensemble_phenotype_idx, :], -2 * np.log(self.ensemble_size), 0.0
            )
            log_weights[ensemble_phenotype_idx, :] -= logsumexp(
                log_weights[ensemble_phenotype_idx, :]
            )

        # compute weighted per-phenotype and per-time mean and covariance
        for phenotype_idx in range(self.num_phenotypes):
            weights = np.exp(log_weights[phenotype_idx, :])
            self.ensemble_macrostate_mean[
                phenotype_idx, previous_observation_time : observation_time + 1, :
            ] = np.average(
                self.ensemble_macrostate[
                    phenotype_idx, :, previous_observation_time : observation_time + 1, :
                ],
                weights=weights,
                axis=0,
            )
            for time_idx in range(previous_observation_time, observation_time + 1):
                self.ensemble_macrostate_cov[phenotype_idx, time_idx, :, :] = np.cov(
                    self.ensemble_macrostate[phenotype_idx, :, time_idx, :],
                    aweights=weights,
                    rowvar=False,
                )

        if log:
            print(f" (Finished)", flush=True)

        # ######### Step 2: perform the Kalman update from the observation, per phenotype

        if log:
            print(f"Kalman update", end="", flush=True)

        # Step 2a. Compute some generally useful stuff

        # assemble the matrix for the observation model
        H = np.zeros((dim_observation, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64)
        for h_idx, obs_name in enumerate(observation_types):
            H[h_idx, state_var_indices[obs_name]] = 1.0

        # R is the restriction of the observation uncertainty matrix to the observed subset
        R = H @ self.observation_uncertainty_cov @ H.T

        observation = self.transform_intrinsic_to_kf(measurements)

        ident_matrix = np.identity(UNIFIED_STATE_SPACE_DIMENSION)

        # Step 2b. Kalman update for each phenotype

        for phenotype_idx in range(self.num_phenotypes):
            mu = self.ensemble_macrostate_mean[phenotype_idx, observation_time, :]
            P = self.ensemble_macrostate_cov[phenotype_idx, observation_time, :, :]

            v = observation - (H @ mu)
            S = H @ P @ H.T + R

            # TODO: figure out why pycharm thinks that np.log is not a function, but a boolean. Not python thinking
            #  that, pycharm.
            # noinspection PyCallingNonCallable
            self.log_phenotype_distribution[phenotype_idx] -= 0.5 * (
                v.T @ scipy.linalg.pinvh(S) @ v
                + abslogdet(S)
                + np.log(2 * np.pi) * len(observation_types)
            )

            K = P @ H.T @ np.linalg.pinv(S)

            # Note for future readers: mu[:] and P[:,:] access the the contents of the original arrays that mu and P
            # are derived from. If you replaced this with mu and P, you'd just get new arrays and the originals would
            # be unchanged.
            mu[:] += K @ v
            # Joseph form update (See e.g. https://www.anuncommonlab.com/articles/how-kalman-filters-work/part2.html)
            A = ident_matrix - K @ H
            P[:, :] = pos_def_matrix_cleanup(A @ P @ A.T + K @ R @ K.T, __eigenvalue_epsilon__)

        # Step 2c: Normalize overall phenotype probabilities and temper them so that phenotype probabilities never
        # drop below a threshold or become overly certain.
        self.log_phenotype_distribution -= logsumexp(self.log_phenotype_distribution)
        np.clip(
            self.log_phenotype_distribution,
            np.log(0.01),
            np.log(0.99),
            out=self.log_phenotype_distribution,
        )
        self.log_phenotype_distribution -= logsumexp(self.log_phenotype_distribution)

        # update timeseries
        self.log_phenotype_distribution_timeseries[:, observation_time:] = (
            self.log_phenotype_distribution[:, None]
        )

        # ########## Step 3: As in the unscented kalman filter, find the (2n+1)-point stencil of the new Gaussian's and
        # ########## best matching to the ensemble members

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
                ensemble_target_locs[phenotype_idx, 2 * axis_idx + 1, :] = (
                    axis + self.ensemble_macrostate_mean[phenotype_idx, observation_time]
                )
                ensemble_target_locs[phenotype_idx, 2 * axis_idx + 2, :] = (
                    -axis + self.ensemble_macrostate_mean[phenotype_idx, observation_time]
                )

            model_to_sample_pairing[phenotype_idx, :] = gale_shapely_matching(
                new_sample=ensemble_target_locs[phenotype_idx],
                macro_data=self.ensemble_macrostate[phenotype_idx, :, observation_time],
            )

        if log:
            print(f" (Finished)", flush=True)

        # ######### Step 4: Microstate synthesis

        if log:
            print(f"Microstate synthesis", end="", flush=True)

        for phenotype_idx in range(self.num_phenotypes):
            for model_idx in range(2 * UNIFIED_STATE_SPACE_DIMENSION + 1):
                self.model_modification_algorithm(
                    self.ensemble[phenotype_idx][model_idx],
                    self.transform_kf_to_intrinsic(
                        ensemble_target_locs[
                            phenotype_idx, model_to_sample_pairing[phenotype_idx, model_idx], :
                        ]
                    ),
                    verbose=True,  # TODO: remember to revisit
                    state_var_indices=state_var_indices,
                    state_vars=state_vars,
                    variational_params=variational_params,
                )

        if log:
            print(f" (Finished)", flush=True)

        # ######### Step 5: Cleanup

        # update counters
        # previous_time = self._current_time
        self._current_time = observation_time
        self._kf_iteration += 1

    def plot_state_vars(self, TIME_SPAN, vp_trajectory, cycle, SAMPLE_INTERVAL, FILE_PREFIX):
        """
        plot projection of state variables

        :return:
        """
        import matplotlib.pyplot as plt

        from phkf_ac.util import fix_title

        (graph_rows, graph_cols, graph_figsize) = figure_gridlayout(len(state_vars))

        fig, axs = plt.subplots(
            nrows=graph_rows,
            ncols=graph_cols,
            figsize=graph_figsize,
            sharex=True,
            sharey=False,
            layout="constrained",
        )
        for idx, state_var_name in enumerate(state_vars):
            row, col = divmod(idx, graph_cols)
            (true_value,) = axs[row, col].plot(
                range(TIME_SPAN + 1),
                vp_trajectory[:, idx],
                label="true value",
                linestyle=":",
                color="gray",
            )
            (past_estimate_center_line,) = axs[row, col].plot(
                range((cycle + 1) * SAMPLE_INTERVAL),
                self.transform_kf_to_intrinsic(
                    # TODO: indices
                    self.ensemble_macrostate_mean[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx],
                    index=idx,
                ),
                label="estimate of past",
                color="black",
            )
            past_estimate_range = axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL),
                np.clip(
                    self.transform_kf_to_intrinsic(
                        # TODO: indices
                        self.ensemble_macrostate_mean[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx]
                        - np.sqrt(
                            self.ensemble_macrostate_cov[
                                cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx, idx
                            ]
                        ),
                        index=idx,
                    ),
                    0.0,
                    np.inf,
                ),
                # np.minimum(
                #     10 * max_scales[state_var_name],
                self.transform_kf_to_intrinsic(
                    # TODO: indices
                    self.ensemble_macrostate_mean[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx]
                    + np.sqrt(
                        self.ensemble_macrostate_cov[
                            cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx, idx
                        ]
                    ),
                    index=idx,
                ),
                # ),
                color="gray",
                alpha=0.35,
            )

            # TODO: indices
            mu = self.ensemble_macrostate_mean[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx]
            sigma = np.sqrt(
                self.ensemble_macrostate_cov[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]
            )

            (prediction_center_line,) = axs[row, col].plot(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                self.transform_kf_to_intrinsic(mu, index=idx),
                label="prediction",
                color="blue",
            )
            prediction_range = axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                np.clip(self.transform_kf_to_intrinsic(mu - sigma, index=idx), 0.0, np.inf),
                self.transform_kf_to_intrinsic(
                    mu + sigma,
                    index=idx,
                ),
                color="blue",
                alpha=0.35,
            )
            axs[row, col].set_title(fix_title(state_var_name), loc="left", wrap=True)
            axs[row, col].set_ylim(bottom=max(0.0, axs[row, col].get_ylim()[0]))
        # remove axes on unused graphs
        for idx in range(
            len(state_vars),
            graph_rows * graph_cols,
        ):
            row, col = divmod(idx, graph_cols)
            axs[row, col].set_axis_off()

        # place legend
        if len(state_vars) < graph_rows * graph_cols:
            legend_placement = axs[graph_rows - 1, graph_cols - 1]
            legend_loc = "upper left"
        else:
            legend_placement = fig
            legend_loc = "outside lower center"

        # noinspection PyUnboundLocalVariable
        legend_placement.legend(
            [
                true_value,
                (past_estimate_center_line, past_estimate_range),
                (prediction_center_line, prediction_range),
            ],
            [
                true_value.get_label(),
                past_estimate_center_line.get_label(),
                prediction_center_line.get_label(),
            ],
            loc=legend_loc,
        )
        fig.suptitle("State Prediction")
        fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-state.pdf")
        plt.close(fig)

    def plot_parameters(self, TIME_SPAN, cycle, SAMPLE_INTERVAL, FILE_PREFIX, vp_init_params):
        """
        plot projection of parameters

        :return:
        """
        import matplotlib.pyplot as plt

        from phkf_ac.util import fix_title

        len_state_vars = self.ensemble_macrostate_mean.shape[-1]

        (graph_rows, graph_cols, graph_figsize) = figure_gridlayout(len(variational_params))

        fig, axs = plt.subplots(
            nrows=graph_rows,
            ncols=graph_cols,
            figsize=graph_figsize,
            sharex=True,
            sharey=False,
            layout="constrained",
        )
        (
            true_value,
            past_estimate_center_line,
            past_estimate_range,
            prediction_center_line,
            prediction_range,
        ) = [None] * 5
        for idx, param_name in enumerate(variational_params):
            row, col = divmod(idx, graph_cols)

            if param_name in vp_init_params:
                (true_value,) = axs[row, col].plot(
                    [0, TIME_SPAN + 1],
                    [vp_init_params[param_name]] * 2,
                    label="true value",
                    color="gray",
                    linestyle=":",
                )

            (past_estimate_center_line,) = axs[row, col].plot(
                range((cycle + 1) * SAMPLE_INTERVAL),
                self.transform_kf_to_intrinsic(
                    # TODO: indices
                    self.ensemble_macrostate_mean[
                        cycle, : (cycle + 1) * SAMPLE_INTERVAL, len_state_vars + idx
                    ],
                    index=len_state_vars + idx,
                ),
                color="black",
                label="estimate of past",
            )

            past_estimate_range = axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL),
                np.clip(
                    self.transform_kf_to_intrinsic(
                        # TODO: indices
                        self.ensemble_macrostate_mean[
                            cycle, : (cycle + 1) * SAMPLE_INTERVAL, len_state_vars + idx
                        ]
                        - np.sqrt(
                            self.ensemble_macrostate_cov[
                                cycle,
                                : (cycle + 1) * SAMPLE_INTERVAL,
                                len_state_vars + idx,
                                len_state_vars + idx,
                            ]
                        ),
                        index=len_state_vars + idx,
                    ),
                    0.0,
                    np.inf,
                ),
                self.transform_kf_to_intrinsic(
                    # TODO: indices
                    self.ensemble_macrostate_mean[
                        cycle, : (cycle + 1) * SAMPLE_INTERVAL, len_state_vars + idx
                    ]
                    + np.sqrt(
                        self.ensemble_macrostate_cov[
                            cycle,
                            : (cycle + 1) * SAMPLE_INTERVAL,
                            len_state_vars + idx,
                            len_state_vars + idx,
                        ]
                    ),
                    index=len_state_vars + idx,
                ),
                color="gray",
                alpha=0.35,
                label="past cone of uncertainty",
            )

            (prediction_center_line,) = axs[row, col].plot(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                self.transform_kf_to_intrinsic(
                    # TODO: indices
                    self.ensemble_macrostate_mean[
                        cycle, (cycle + 1) * SAMPLE_INTERVAL :, len_state_vars + idx
                    ],
                    index=len_state_vars + idx,
                ),
                label="predictive estimate",
                color="blue",
            )

            prediction_range = axs[row, col].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                np.clip(
                    self.transform_kf_to_intrinsic(
                        # TODO: indices
                        self.ensemble_macrostate_mean[
                            cycle, (cycle + 1) * SAMPLE_INTERVAL :, len_state_vars + idx
                        ]
                        - np.sqrt(
                            self.ensemble_macrostate_cov[
                                cycle,
                                (cycle + 1) * SAMPLE_INTERVAL :,
                                len_state_vars + idx,
                                len_state_vars + idx,
                            ]
                        ),
                        index=len_state_vars + idx,
                    ),
                    0.0,
                    np.inf,
                ),
                self.transform_kf_to_intrinsic(
                    # TODO: indices
                    self.ensemble_macrostate_mean[
                        cycle, (cycle + 1) * SAMPLE_INTERVAL :, len_state_vars + idx
                    ]
                    + np.sqrt(
                        self.ensemble_macrostate_cov[
                            cycle,
                            (cycle + 1) * SAMPLE_INTERVAL :,
                            len_state_vars + idx,
                            len_state_vars + idx,
                        ]
                    ),
                    index=len_state_vars + idx,
                ),
                color="blue",
                alpha=0.35,
            )
            axs[row, col].set_title(fix_title(param_name), loc="center", wrap=True)
            axs[row, col].set_ylim(bottom=max(0.0, axs[row, col].get_ylim()[0]))

        # remove axes on unused graphs
        for idx in range(
            len(variational_params),
            graph_rows * graph_cols,
        ):
            row, col = divmod(idx, graph_cols)
            axs[row, col].set_axis_off()

        # place legend
        if len(state_vars) < graph_rows * graph_cols:
            legend_placement = axs[graph_rows - 1, graph_cols - 1]
            legend_loc = "upper left"
        else:
            legend_placement = fig
            legend_loc = "outside lower center"

        legend_placement.legend(
            [
                true_value,
                (past_estimate_center_line, past_estimate_range),
                (prediction_center_line, prediction_range),
            ],
            [
                true_value.get_label(),
                past_estimate_center_line.get_label(),
                prediction_center_line.get_label(),
            ],
            loc=legend_loc,
        )
        fig.suptitle("Parameter Projection")
        fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-params.pdf")
        plt.close(fig)


def model_sigma_point_ensemble_from_helper(params: np.ndarray):
    model_param_dict = default_params.copy()
    for sample_component, parameter_name in zip(
        params,
        (init_only_params + variational_params),
    ):
        model_param_dict[parameter_name] = (
            round(max(0, sample_component))
            if isinstance(default_params[parameter_name], int)
            else max(0.0, sample_component)
        )
    # create model for virtual patient
    model = AnCockrellModel(**model_param_dict)
    return model


def model_sigma_point_ensemble_from(means: np.ndarray, covariances: np.ndarray, count: int):
    """
    Create an ensemble of models from a distribution. Uses init-only
    and variational parameters

    :param means:
    :param covariances:
    :param count: number of ensemble members
    :return:
    """
    # make 100% sure of the covariance matrix
    from phkf_ac.util import pos_def_matrix_cleanup

    covariances = pos_def_matrix_cleanup(covariances, __eigenvalue_epsilon__)

    # get principal axes
    U, S, Vh = np.linalg.svd(covariances, full_matrices=True, compute_uv=True, hermitian=True)
    axes = U * np.sqrt(S)

    mdl_ensemble = [model_sigma_point_ensemble_from_helper(means)]
    for axis in axes:
        mdl_ensemble.append(model_sigma_point_ensemble_from_helper(means + axis))
        mdl_ensemble.append(model_sigma_point_ensemble_from_helper(means - axis))

    if len(mdl_ensemble) >= count:
        return mdl_ensemble[:count]

    # fill out the rest of the ensemble
    while len(mdl_ensemble) < count:
        offset = np.random.multivariate_normal(
            mean=np.zeros_like(axes[0]), cov=np.identity(axes.shape[0])
        )
        # don't let it get _too_ big/weird
        norm_offset = np.linalg.norm(offset)
        if norm_offset > 1:
            offset /= norm_offset

        mdl_ensemble.append(model_sigma_point_ensemble_from_helper(means + axes @ offset))

    return mdl_ensemble


def figure_gridlayout(num_vars: int):
    # layout for graphing variables.
    # Attempts to be mostly square, with possibly more rows than columns
    graph_cols: Final[int] = int(np.floor(np.sqrt(num_vars)))
    graph_rows: Final[int] = int(np.ceil(num_vars / graph_cols))
    graph_figsize: Final[Tuple[float, float]] = (
        1.8 * graph_rows,
        1.8 * graph_cols,
    )
    return graph_rows, graph_cols, graph_figsize
