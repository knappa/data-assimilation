from phkf_ac.runner import PhenotypeKFAnCockrell

####################################################################################################


def main_cli():
    import argparse
    from typing import Final, List

    import an_cockrell
    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import multivariate_normal
    from tqdm.auto import tqdm

    from phkf_ac.consts import (
        UNIFIED_STATE_SPACE_DIMENSION,
        default_params,
        init_only_params,
        state_var_indices,
        state_vars,
        variational_params,
    )
    from phkf_ac.runner import figure_gridlayout
    from phkf_ac.transform import transform_intrinsic_to_kf, transform_kf_to_intrinsic
    from phkf_ac.util import fix_title, gale_shapely_matching, model_macro_data, slogdet

    ################################################################################

    parser = argparse.ArgumentParser()

    parser.add_argument("--prefix", type=str, default="", help="output file prefix")

    parser.add_argument(
        "--measurements",
        type=str,
        choices=[
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
        ],
        nargs="+",
        required=True,
        help="which things to measure (required)",
    )

    parser.add_argument("--graphs", help="make pdf graphs", action="store_true")

    parser.add_argument(
        "--update-algorithm",
        type=str,
        choices=["simple", "spatial", "full-spatial"],
        required=True,
    )

    # parameters for the measurement uncertainty (coeffs in the Kalman filter's R matrix)
    parser.add_argument("--uncertainty-P_DAMPS", type=float, default=0.001, required=False)
    parser.add_argument("--uncertainty-T1IFN", type=float, default=0.001, required=False)
    parser.add_argument("--uncertainty-TNF", type=float, default=0.001, required=False)
    parser.add_argument("--uncertainty-IFNg", type=float, default=0.001, required=False)
    parser.add_argument("--uncertainty-IL6", type=float, default=0.001, required=False)
    parser.add_argument("--uncertainty-IL1", type=float, default=0.001, required=False)
    parser.add_argument("--uncertainty-IL8", type=float, default=0.001, required=False)
    parser.add_argument("--uncertainty-IL10", type=float, default=0.001, required=False)
    parser.add_argument("--uncertainty-IL12", type=float, default=0.001, required=False)
    parser.add_argument("--uncertainty-IL18", type=float, default=0.001, required=False)
    parser.add_argument(
        "--uncertainty-extracellular_virus", type=float, default=0.001, required=False
    )

    # parser.add_argument(
    #     "--param_stoch_level",
    #     help="stochasticity parameter for parameter random walk",
    #     type=float,
    #     default=0.001,
    # )
    # parser.add_argument("--verbose", help="print extra diagnostic messages", action="store_true")
    parser.add_argument("--log", help="print extra diagnostic messages", action="store_true")

    parser.add_argument(
        "--predict",
        type=str,
        choices=["to-kf-update", "to-next-kf-update", "to-end"],
        help="how far to extend predictions",
        required=True,
    )

    parser.add_argument("--grid_width", help="width of simulation grid", type=int, default=51)
    parser.add_argument("--grid_height", help="height of simulation grid", type=int, default=51)

    parser.add_argument(
        "--time_span",
        help="number of abm iterations in a full run",
        type=int,
        default=2016,
    )
    parser.add_argument(
        "--sample_interval",
        help="iterations in the interval between samples",
        type=int,
        default=48,
    )

    args = parser.parse_args()

    ################################################################################
    # command line option interpretation / backfilling those options when you,
    # for example, paste the code into ipython

    if hasattr(args, "grid_width"):
        default_params["GRID_WIDTH"] = args.grid_width

    if hasattr(args, "grid_height"):
        default_params["GRID_HEIGHT"] = args.grid_height

    modification_algorithm: Final[str] = (
        "spatial" if not hasattr(args, "update_algorithm") else args.update_algorithm
    )

    if modification_algorithm == "full-spatial":
        from phkf_ac.modify_full_spatial import modify_model
    elif modification_algorithm == "spatial":
        from phkf_ac.modify_epi_spatial import modify_model
    else:
        from phkf_ac.modify_simple import modify_model

    # # rs encodes the uncertainty in the various observations. Defaults are 1.0 if
    # # not set via the command line. (See above for defaults if you are. Right now,
    # # they are also 1.0.)
    # rs: Final[Dict[str, float]] = {
    #     "total_"
    #     + name: (
    #         1.0
    #         if not hasattr(args, "uncertainty_" + name)
    #         else getattr(args, "uncertainty_" + name)
    #     )
    #     for name in [
    #         "P_DAMPS",
    #         "T1IFN",
    #         "TNF",
    #         "IFNg",
    #         "IL1",
    #         "IL6",
    #         "IL8",
    #         "IL10",
    #         "IL12",
    #         "IL18",
    #         "extracellular_virus",
    #     ]
    # }
    #
    # PARAM_STOCH_LEVEL: Final[float] = (
    #     0.001
    #     if not hasattr(args, "param_stoch_level")
    #     or args.param_stoch_level is None
    #     or args.param_stoch_level == -1
    #     else args.param_stoch_level
    # )

    # ENSEMBLE_SIZE: Final[int] = (
    #     (UNIFIED_STATE_SPACE_DIMENSION + 1) * UNIFIED_STATE_SPACE_DIMENSION // 2
    # )  # max(50, (UNIFIED_STATE_SPACE_DIMENSION + 1))
    OBSERVABLES: Final[List[str]] = (
        ["extracellular_virus"] if not hasattr(args, "measurements") else args.measurements
    )
    OBSERVABLE_VAR_NAMES: Final[List[str]] = ["total_" + name for name in OBSERVABLES]

    FILE_PREFIX: Final[str] = (
        "" if not hasattr(args, "prefix") or len(args.prefix) == 0 else args.prefix + "-"
    )

    GRAPHS: Final[bool] = True if not hasattr(args, "graphs") else bool(args.graphs)

    PREDICT: Final[str] = "to-kf-update" if not hasattr(args, "predict") else args.predict

    TIME_SPAN: Final[int] = (
        2016 if not hasattr(args, "time_span") or args.time_span <= 0 else args.time_span
    )
    SAMPLE_INTERVAL: Final[int] = (
        48
        if not hasattr(args, "sample_interval") or args.sample_interval <= 0
        else args.sample_interval
    )

    log: Final[bool] = False if not hasattr(args, "log") else args.log

    ################################################################################
    # constants

    # have the models' parameters do a random walk over time (should help
    # with covariance starvation)
    # PARAMETER_RANDOM_WALK: Final[bool] = True

    NUM_CYCLES: Final[int] = TIME_SPAN // SAMPLE_INTERVAL

    ################################################################################
    # graph layout computations

    ################################################################################
    # statistical parameters

    init_mean_vec = np.array(
        [default_params[param] for param in (init_only_params + variational_params)]
    )

    init_cov_matrix = np.diag(
        np.array(
            [
                0.75 * np.sqrt(default_params[param])
                for param in (init_only_params + variational_params)
            ]
        )
    )

    ################################################################################
    # sample a virtual patient

    # sampled virtual patient parameters
    vp_init_params = default_params.copy()
    vp_init_param_sample = np.abs(
        multivariate_normal(mean=init_mean_vec, cov=init_cov_matrix).rvs()
    )
    for sample_component, param_name in zip(
        vp_init_param_sample,
        (init_only_params + variational_params),
    ):
        vp_init_params[param_name] = (
            round(sample_component)
            if isinstance(default_params[param_name], int)
            else sample_component
        )

    # create model for virtual patient
    # rng = np.random.default_rng()
    # virtual_patient_model = an_cockrell.AnCockrellModel(rng=rng, **vp_init_params)
    virtual_patient_model = an_cockrell.AnCockrellModel(**vp_init_params)

    # evaluate the virtual patient's trajectory
    vp_trajectory = np.zeros((TIME_SPAN + 1, UNIFIED_STATE_SPACE_DIMENSION), dtype=np.float64)

    vp_trajectory[0, :] = model_macro_data(virtual_patient_model)
    # noinspection PyTypeChecker
    for t in tqdm(range(1, TIME_SPAN + 1), desc="create virtual patient"):
        virtual_patient_model.time_step()
        vp_trajectory[t, :] = model_macro_data(virtual_patient_model)

    ################################################################################
    # plot virtual patient

    if GRAPHS:
        if log:
            print("Plotting Virtual patient", end="")
        (graph_rows, graph_cols, graph_figsize) = figure_gridlayout(len(state_vars))
        fig, axs = plt.subplots(
            nrows=graph_rows,
            ncols=graph_cols,
            figsize=graph_figsize,
            sharex=True,
            sharey=False,
        )
        for idx, state_var_name in enumerate(state_vars):
            row, col = divmod(idx, graph_cols)
            axs[row, col].plot(vp_trajectory[:, idx])

            axs[row, col].set_title(
                fix_title(state_var_name),
                loc="center",
                wrap=True,
            )
            axs[row, col].set_ylim(bottom=max(0.0, axs[row, col].get_ylim()[0]))
        for idx in range(len(state_vars), graph_rows * graph_cols):
            row, col = divmod(idx, graph_cols)
            axs[row, col].set_axis_off()
        fig.tight_layout()
        fig.savefig(FILE_PREFIX + "virtual-patient-trajectory.pdf")
        plt.close(fig)
        if log:
            print(" (Finished)")

    ################################################################################

    if log:
        print("Loading phenotype data", end="")
    from importlib.resources import files

    with h5py.File(files("phkf_ac.data").joinpath("phenotype-statistics.hdf5"), "r") as h5file:
        pca_center = h5file["pca_center"][:]
        pca_matrix = h5file["pca_matrix"][:, :]
        phenotype_weight_means = h5file["phenotype_weight_means"][:]
        phenotype_weight_covs = h5file["phenotype_weight_covs"][:, :]

    ensemble = PhenotypeKFAnCockrell(
        num_phenotypes=4,
        pca_center=pca_center,
        pca_matrix=pca_matrix,
        phenotype_weight_means=phenotype_weight_means,
        phenotype_weight_covs=phenotype_weight_covs,
        model_modification_algorithm=modify_model,
    )

    if log:
        print(" (Finished)")

    ################################################################################
    # Kalman filter simulation
    ################################################################################

    if log:
        print("Initializing ensemble", end="")

    # create ensemble of models for kalman filter
    time = 0
    ensemble.initialize_ensemble(
        initialization_means=init_mean_vec, initialization_covs=init_cov_matrix, t=time
    )

    if log:
        print(" (Finished)")

    for cycle in tqdm(range(NUM_CYCLES), desc="cycle"):

        if log:
            print("Advancing ensemble", end="")

        # advance ensemble of models
        time += SAMPLE_INTERVAL
        ensemble.project_ensemble_to(t=time, update_ensemble=True)

        if PREDICT != "to-kf-update":
            if PREDICT == "to-next-kf-update":
                final_time = time + SAMPLE_INTERVAL + 1
            else:
                # PREDICT == "to-end"
                final_time = TIME_SPAN + 1
            ensemble.project_ensemble_to(t=final_time, update_ensemble=False)

        if log:
            print(" (Finished)")

        ################################################################################
        # plot projection of state variables and parameters

        if GRAPHS:
            ensemble.plot_state_vars(TIME_SPAN, vp_trajectory, cycle, SAMPLE_INTERVAL, FILE_PREFIX)
            ensemble.plot_parameters(
                TIME_SPAN, cycle, SAMPLE_INTERVAL, FILE_PREFIX, vp_init_params=vp_init_params
            )

        ################################################################################
        # Kalman filter

        ensemble.kf_update(
            observation_time=time,
            observation_types=OBSERVABLE_VAR_NAMES,
            measurements=np.array(
                [
                    transform_intrinsic_to_kf(
                        vp_trajectory[time, state_var_indices[obs_name]],
                        index=state_var_indices[obs_name],
                    )
                    for obs_name in OBSERVABLE_VAR_NAMES
                ],
                dtype=np.float64,
            ),
            save_microstate_files=False,
        )

        ################################################################################
        # plot projection of state variables and parameters, post KF update

        if GRAPHS:
            ensemble.plot_state_vars(TIME_SPAN, vp_trajectory, cycle, SAMPLE_INTERVAL, FILE_PREFIX)
            ensemble.plot_parameters(
                TIME_SPAN, cycle, SAMPLE_INTERVAL, FILE_PREFIX, vp_init_params=vp_init_params
            )

        # ################################################################################
        # # plot kalman update of state variables
        #
        # if GRAPHS:
        #     for cycle in range(NUM_CYCLES - 1):
        #         fig, axs = plt.subplots(
        #             nrows=state_var_graphs_rows,
        #             ncols=state_var_graphs_cols,
        #             figsize=state_var_graphs_figsize,
        #             sharex=True,
        #             sharey=False,
        #             layout="constrained",
        #         )
        #         for idx, state_var_name in enumerate(state_vars):
        #             row, col = divmod(idx, state_var_graphs_cols)
        #
        #             (true_value,) = axs[row, col].plot(
        #                 range(TIME_SPAN + 1),
        #                 vp_trajectory[:, idx],
        #                 label="true value",
        #                 color="black",
        #             )
        #
        #             (past_est_center_line,) = axs[row, col].plot(
        #                 range((cycle + 1) * SAMPLE_INTERVAL),
        #                 transform_kf_to_intrinsic(
        #                     mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx], index=idx
        #                 ),
        #                 color="green",
        #                 label="estimate of past",
        #             )
        #
        #             past_est_range = axs[row, col].fill_between(
        #                 range((cycle + 1) * SAMPLE_INTERVAL),
        #                 np.maximum(
        #                     0.0,
        #                     transform_kf_to_intrinsic(
        #                         mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx]
        #                         - np.sqrt(cov_matrix[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx, idx]),
        #                         index=idx,
        #                     ),
        #                 ),
        #                 transform_kf_to_intrinsic(
        #                     mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx]
        #                     + np.sqrt(cov_matrix[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx, idx]),
        #                     index=idx,
        #                 ),
        #                 color="green",
        #                 alpha=0.35,
        #             )
        #
        #             (future_est_before_update_center_line,) = axs[row, col].plot(
        #                 range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
        #                 transform_kf_to_intrinsic(
        #                     mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx], index=idx
        #                 ),
        #                 color="#d5b60a",
        #                 label="previous future estimate",
        #             )
        #             future_est_before_update_range = axs[row, col].fill_between(
        #                 range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
        #                 np.maximum(
        #                     0.0,
        #                     transform_kf_to_intrinsic(
        #                         mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx]
        #                         - np.sqrt(cov_matrix[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]),
        #                         index=idx,
        #                     ),
        #                 ),
        #                 transform_kf_to_intrinsic(
        #                     mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx]
        #                     + np.sqrt(cov_matrix[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]),
        #                     index=idx,
        #                 ),
        #                 color="#d5b60a",
        #                 alpha=0.35,
        #             )
        #
        #             (future_est_after_update_center_line,) = axs[row, col].plot(
        #                 range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
        #                 transform_kf_to_intrinsic(
        #                     mean_vec[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, idx], index=idx
        #                 ),
        #                 label="updated future estimate",
        #                 color="blue",
        #             )
        #
        #             future_est_after_update_range = axs[row, col].fill_between(
        #                 range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
        #                 np.maximum(
        #                     0.0,
        #                     transform_kf_to_intrinsic(
        #                         mean_vec[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, idx]
        #                         - np.sqrt(
        #                             cov_matrix[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]
        #                         ),
        #                         index=idx,
        #                     ),
        #                 ),
        #                 transform_kf_to_intrinsic(
        #                     mean_vec[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, idx]
        #                     + np.sqrt(cov_matrix[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]),
        #                     index=idx,
        #                 ),
        #                 color="blue",
        #                 alpha=0.35,
        #             )
        #             axs[row, col].set_title(fix_title(state_var_name), loc="left", wrap=True)
        #             axs[row, col].set_ylim(bottom=max(0.0, axs[row, col].get_ylim()[0]))
        #         # remove axes on unused graphs
        #         for idx in range(
        #             len(state_vars),
        #             state_var_graphs_rows * state_var_graphs_cols,
        #         ):
        #             row, col = divmod(idx, state_var_graphs_cols)
        #             axs[row, col].set_axis_off()
        #
        #         # place legend
        #         if len(state_vars) < state_var_graphs_rows * state_var_graphs_cols:
        #             legend_placement = axs[state_var_graphs_rows - 1, state_var_graphs_cols - 1]
        #             legend_loc = "upper left"
        #         else:
        #             legend_placement = fig
        #             legend_loc = "outside lower center"
        #
        #         # noinspection PyUnboundLocalVariable
        #         legend_placement.legend(
        #             [
        #                 true_value,
        #                 (past_est_center_line, past_est_range),
        #                 (future_est_before_update_center_line, future_est_before_update_range),
        #                 (future_est_after_update_center_line, future_est_after_update_range),
        #             ],
        #             [
        #                 true_value.get_label(),
        #                 past_est_center_line.get_label(),
        #                 future_est_before_update_center_line.get_label(),
        #                 future_est_after_update_center_line.get_label(),
        #             ],
        #             loc=legend_loc,
        #         )
        #         fig.suptitle("State Projection", ha="left")
        #         fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-state-kfupd.pdf")
        #         plt.close(fig)
        #
        # ################################################################################
        # # plot kalman update of parameters
        #
        # if GRAPHS:
        #     len_state_vars = len(state_vars)
        #     for cycle in range(NUM_CYCLES - 1):
        #         fig, axs = plt.subplots(
        #             nrows=variational_params_graphs_rows,
        #             ncols=variational_params_graphs_cols,
        #             figsize=variational_params_graphs_figsize,
        #             sharex=True,
        #             sharey=False,
        #             layout="constrained",
        #         )
        #         for idx, param_name in enumerate(variational_params):
        #             row, col = divmod(idx, variational_params_graphs_cols)
        #
        #             if param_name in vp_init_params:
        #                 (true_value,) = axs[row, col].plot(
        #                     [0, TIME_SPAN + 1],
        #                     [vp_init_params[param_name]] * 2,
        #                     label="true value",
        #                     color="black",
        #                 )
        #
        #             (past_est_center_line,) = axs[row, col].plot(
        #                 range((cycle + 1) * SAMPLE_INTERVAL),
        #                 transform_kf_to_intrinsic(
        #                     mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, len_state_vars + idx],
        #                     index=len_state_vars + idx,
        #                 ),
        #                 label="estimate of past",
        #                 color="green",
        #             )
        #
        #             past_est_range = axs[row, col].fill_between(
        #                 range((cycle + 1) * SAMPLE_INTERVAL),
        #                 np.maximum(
        #                     0.0,
        #                     transform_kf_to_intrinsic(
        #                         mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, len_state_vars + idx]
        #                         - np.sqrt(
        #                             cov_matrix[
        #                                 cycle,
        #                                 : (cycle + 1) * SAMPLE_INTERVAL,
        #                                 len_state_vars + idx,
        #                                 len_state_vars + idx,
        #                             ]
        #                         ),
        #                         index=len_state_vars + idx,
        #                     ),
        #                 ),
        #                 transform_kf_to_intrinsic(
        #                     mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, len_state_vars + idx]
        #                     + np.sqrt(
        #                         cov_matrix[
        #                             cycle,
        #                             : (cycle + 1) * SAMPLE_INTERVAL,
        #                             len_state_vars + idx,
        #                             len_state_vars + idx,
        #                         ]
        #                     ),
        #                     index=len_state_vars + idx,
        #                 ),
        #                 color="green",
        #                 alpha=0.35,
        #             )
        #
        #             (future_est_before_update_center_line,) = axs[row, col].plot(
        #                 range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
        #                 transform_kf_to_intrinsic(
        #                     mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, len_state_vars + idx],
        #                     index=len_state_vars + idx,
        #                 ),
        #                 label="old estimate",
        #                 color="#d5b60a",
        #             )
        #
        #             future_est_before_update_range = axs[row, col].fill_between(
        #                 range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
        #                 np.maximum(
        #                     0.0,
        #                     transform_kf_to_intrinsic(
        #                         mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, len_state_vars + idx]
        #                         - np.sqrt(
        #                             cov_matrix[
        #                                 cycle,
        #                                 (cycle + 1) * SAMPLE_INTERVAL :,
        #                                 len_state_vars + idx,
        #                                 len_state_vars + idx,
        #                             ]
        #                         ),
        #                         index=len_state_vars + idx,
        #                     ),
        #                 ),
        #                 transform_kf_to_intrinsic(
        #                     mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, len_state_vars + idx]
        #                     + np.sqrt(
        #                         cov_matrix[
        #                             cycle,
        #                             (cycle + 1) * SAMPLE_INTERVAL :,
        #                             len_state_vars + idx,
        #                             len_state_vars + idx,
        #                         ]
        #                     ),
        #                     index=len_state_vars + idx,
        #                 ),
        #                 color="#d5b60a",
        #                 alpha=0.35,
        #             )
        #
        #             (future_est_after_update_center_line,) = axs[row, col].plot(
        #                 range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
        #                 transform_kf_to_intrinsic(
        #                     mean_vec[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, len_state_vars + idx],
        #                     index=len_state_vars + idx,
        #                 ),
        #                 label="updated estimate",
        #                 color="blue",
        #             )
        #
        #             future_est_after_update_range = axs[row, col].fill_between(
        #                 range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
        #                 np.maximum(
        #                     0.0,
        #                     transform_kf_to_intrinsic(
        #                         mean_vec[
        #                             cycle + 1,
        #                             (cycle + 1) * SAMPLE_INTERVAL :,
        #                             len_state_vars + idx,
        #                         ]
        #                         - np.sqrt(
        #                             cov_matrix[
        #                                 cycle + 1,
        #                                 (cycle + 1) * SAMPLE_INTERVAL :,
        #                                 len_state_vars + idx,
        #                                 len_state_vars + idx,
        #                             ]
        #                         ),
        #                         index=len_state_vars + idx,
        #                     ),
        #                 ),
        #                 transform_kf_to_intrinsic(
        #                     mean_vec[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, len_state_vars + idx]
        #                     + np.sqrt(
        #                         cov_matrix[
        #                             cycle + 1,
        #                             (cycle + 1) * SAMPLE_INTERVAL :,
        #                             len_state_vars + idx,
        #                             len_state_vars + idx,
        #                         ]
        #                     ),
        #                     index=len_state_vars + idx,
        #                 ),
        #                 color="blue",
        #                 alpha=0.35,
        #                 label="new future cone of uncertainty",
        #             )
        #             axs[row, col].set_title(fix_title(param_name), loc="center", wrap=True)
        #             axs[row, col].set_ylim(bottom=max(0.0, axs[row, col].get_ylim()[0]))
        #
        #         # remove axes on unused graphs
        #         for idx in range(
        #             len(variational_params),
        #             variational_params_graphs_rows * variational_params_graphs_cols,
        #         ):
        #             row, col = divmod(idx, variational_params_graphs_cols)
        #             axs[row, col].set_axis_off()
        #
        #         # place the legend
        #         if (
        #             len(variational_params)
        #             < variational_params_graphs_rows * variational_params_graphs_cols
        #         ):
        #             legend_placement = axs[
        #                 variational_params_graphs_rows - 1, variational_params_graphs_cols - 1
        #             ]
        #             legend_loc = "upper left"
        #         else:
        #             legend_placement = fig
        #             legend_loc = "outside lower center"
        #
        #         legend_placement.legend(
        #             [
        #                 true_value,
        #                 (past_est_center_line, past_est_range),
        #                 (future_est_before_update_center_line, future_est_before_update_range),
        #                 (future_est_after_update_center_line, future_est_after_update_range),
        #             ],
        #             [
        #                 true_value.get_label(),
        #                 past_est_center_line.get_label(),
        #                 future_est_before_update_center_line.get_label(),
        #                 future_est_after_update_center_line.get_label(),
        #             ],
        #             loc=legend_loc,
        #         )
        #         fig.suptitle("Parameter Projection", x=0, ha="left")
        #         fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-params-kfupd.pdf")
        #         plt.close(fig)

    ################################################################################
    # calculate surprisal information

    #####
    # full surprisal: all state vars and params
    # TODO: indices, refactor into ensemble class
    delta_full = ensemble.ensemble_macrostate_mean - transform_intrinsic_to_kf(vp_trajectory)
    _, logdet = slogdet(ensemble.ensemble_macrostate_cov)
    sigma_inv_delta = np.array(
        [
            [
                np.linalg.lstsq(
                    ensemble.ensemble_macrostate_cov[cycle, t_idx, :, :],
                    delta_full[cycle, t_idx, :],
                    rcond=None,
                )[0]
                for t_idx in range(ensemble.ensemble_macrostate_cov.shape[1])
            ]
            for cycle in range(NUM_CYCLES + 1)
        ]
    )
    surprisal_quadratic_part = np.einsum("cij,cij->ci", delta_full, sigma_inv_delta)
    surprisal_full = (
        surprisal_quadratic_part + logdet + UNIFIED_STATE_SPACE_DIMENSION * np.log(2 * np.pi)
    ) / 2.0

    #####
    # state surprisal: restrict to just the state vars
    delta_state = (
        ensemble.ensemble_macrostate_mean[:, :, : len(state_vars)]
        - transform_intrinsic_to_kf(vp_trajectory)[:, : len(state_vars)]
    )
    _, logdet = slogdet(
        ensemble.ensemble_macrostate_cov[:, :, : len(state_vars), : len(state_vars)]
    )
    sigma_inv_delta = np.array(
        [
            [
                np.linalg.lstsq(
                    ensemble.ensemble_macrostate_cov[
                        cycle, t_idx, : len(state_vars), : len(state_vars)
                    ],
                    delta_state[cycle, t_idx, :],
                    rcond=None,
                )[0]
                for t_idx in range(ensemble.ensemble_macrostate_cov.shape[1])
            ]
            for cycle in range(NUM_CYCLES + 1)
        ]
    )
    surprisal_quadratic_part = np.einsum("cij,cij->ci", delta_state, sigma_inv_delta)
    surprisal_state = (
        surprisal_quadratic_part + logdet + len(state_vars) * np.log(2 * np.pi)
    ) / 2.0

    #####
    # param surprisal: restrict to just the params
    vp_param_trajectory = transform_intrinsic_to_kf(vp_trajectory)[:, len(state_vars) :]

    delta_param = ensemble.ensemble_macrostate_mean[:, :, len(state_vars) :] - vp_param_trajectory
    _, logdet = slogdet(
        ensemble.ensemble_macrostate_cov[:, :, len(state_vars) :, len(state_vars) :]
    )
    sigma_inv_delta = np.array(
        [
            [
                np.linalg.lstsq(
                    ensemble.ensemble_macrostate_cov[
                        cycle, t_idx, len(state_vars) :, len(state_vars) :
                    ],
                    delta_param[cycle, t_idx, :],
                    rcond=None,
                )[0]
                for t_idx in range(ensemble.ensemble_macrostate_cov.shape[1])
            ]
            for cycle in range(NUM_CYCLES + 1)
        ]
    )
    surprisal_quadratic_part = np.einsum("cij,cij->ci", delta_param, sigma_inv_delta)
    surprisal_param = (
        surprisal_quadratic_part + logdet + len(variational_params) * np.log(2 * np.pi)
    ) / 2.0

    ################################################################################

    # see the dimension label information here:
    # https://docs.h5py.org/en/latest/high/dims.html

    with h5py.File(FILE_PREFIX + "data.hdf5", "w") as f:
        f["virtual_patient_trajectory"] = vp_trajectory
        f["virtual_patient_trajectory"].dims[0].label = "time"
        f["virtual_patient_trajectory"].dims[1].label = "state component"

        f["means"] = ensemble.ensemble_macrostate_mean
        f["means"].dims[0].label = "kalman update number"
        f["means"].dims[1].label = "time"
        f["means"].dims[2].label = "state component"

        f["covs"] = ensemble.ensemble_macrostate_cov
        f["covs"].dims[0].label = "kalman update number"
        f["covs"].dims[1].label = "time"
        f["covs"].dims[2].label = "state component"
        f["covs"].dims[3].label = "state component"

        f["surprisal_full"] = surprisal_full
        f["surprisal_full"].dims[0].label = "kalman update number"
        f["surprisal_full"].dims[1].label = "time"

        f["surprisal_state"] = surprisal_state
        f["surprisal_state"].dims[0].label = "kalman update number"
        f["surprisal_state"].dims[1].label = "time"

        f["surprisal_param"] = surprisal_param
        f["surprisal_param"].dims[0].label = "kalman update number"
        f["surprisal_param"].dims[1].label = "time"


# def model_ensemble_from(means, covariances, ensemble_size):
#     """
#     Create an ensemble of models from a distribution. Uses init-only
#     and variational parameters
#
#     :param means:
#     :param covariances:
#     :param ensemble_size:
#     :return:
#     """
#     mdl_ensemble = []
#     distribution = multivariate_normal(mean=means, cov=covariances, allow_singular=True)
#     for _ in range(ensemble_size):
#         model_param_dict = default_params.copy()
#         sampled_params = np.abs(distribution.rvs())
#         # noinspection PyShadowingNames
#         for sample_component, parameter_name in zip(
#             sampled_params,
#             (init_only_params + variational_params),
#         ):
#             model_param_dict[parameter_name] = (
#                 round(sample_component)
#                 if isinstance(default_params[parameter_name], int)
#                 else sample_component
#             )
#         # create model for virtual patient
#         model = AnCockrellModel(**model_param_dict)
#         mdl_ensemble.append(model)
#
#     return mdl_ensemble


####################################################################################################

# TODO: consider log-normal distribution for initial parameters


####################################################################################################
