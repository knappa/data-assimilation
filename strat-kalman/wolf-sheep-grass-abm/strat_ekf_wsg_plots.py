import matplotlib.pyplot as plt
import numpy as np

from transform import transform_kf_to_intrinsic


def parameter_plots(
    true_value,
    vp_wolf_gain_from_food,
    vp_sheep_gain_from_food,
    vp_wolf_reproduce,
    vp_sheep_reproduce,
    vp_grass_regrowth_time,
    TIME_SPAN,
    SAMPLE_INTERVAL,
    cycle,
    mean_vec,
    cov_matrix,
    FILE_PREFIX,
):
    params = [
        "wolf gain from food",
        "sheep gain from food",
        "wolf reproduce",
        "sheep reproduce",
        "grass regrowth time",
    ]
    vp_param_values = dict(
        zip(
            params,
            [
                vp_wolf_gain_from_food,
                vp_sheep_gain_from_food,
                vp_wolf_reproduce,
                vp_sheep_reproduce,
                vp_grass_regrowth_time,
            ],
        )
    )
    fig, axs = plt.subplots(3, 2, figsize=(8, 8), sharex=True, sharey=False, layout="constrained")
    for idx, param_name in enumerate(params):
        row, col = divmod(idx, 3)
        (true_value,) = axs[row, col].plot(
            [0, TIME_SPAN + 1],
            [vp_param_values[param_name]] * 2,
            label="true value",
            color="gray",
            linestyle=":",
        )

        (past_estimate_center_line,) = axs[row, col].plot(
            range((cycle + 1) * SAMPLE_INTERVAL),
            transform_kf_to_intrinsic(
                mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, 3 + idx], index=3 + idx
            ),
            color="black",
            label="estimate of past",
        )
        past_estimate_range = axs[row, col].fill_between(
            range((cycle + 1) * SAMPLE_INTERVAL),
            transform_kf_to_intrinsic(
                mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, 3 + idx]
                - np.sqrt(cov_matrix[cycle, : (cycle + 1) * SAMPLE_INTERVAL, 3 + idx, 3 + idx]),
                index=3 + idx,
            ),
            np.minimum(
                np.maximum(
                    10 * vp_param_values[param_name],
                    np.max(
                        1.1 * transform_kf_to_intrinsic(mean_vec[cycle, :, 3 + idx], index=3 + idx)
                    ),
                ),
                transform_kf_to_intrinsic(
                    mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, 3 + idx]
                    + np.sqrt(cov_matrix[cycle, : (cycle + 1) * SAMPLE_INTERVAL, 3 + idx, 3 + idx]),
                    index=3 + idx,
                ),
            ),
            color="gray",
            alpha=0.35,
            label="past cone of uncertainty",
        )

        (prediction_center_line,) = axs[row, col].plot(
            range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
            transform_kf_to_intrinsic(
                mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, 3 + idx], index=3 + idx
            ),
            label="predictive estimate",
            color="blue",
        )
        prediction_range = axs[row, col].fill_between(
            range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
            transform_kf_to_intrinsic(
                mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, 3 + idx]
                - np.sqrt(cov_matrix[cycle, (cycle + 1) * SAMPLE_INTERVAL :, 3 + idx, 3 + idx]),
                index=3 + idx,
            ),
            np.minimum(
                10 * vp_param_values[param_name],
                transform_kf_to_intrinsic(
                    mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, 3 + idx]
                    + np.sqrt(cov_matrix[cycle, (cycle + 1) * SAMPLE_INTERVAL :, 3 + idx, 3 + idx]),
                    index=3 + idx,
                ),
            ),
            color="blue",
            alpha=0.35,
        )
        axs[row, col].set_title(param_name)
    axs[2, 1].axis("off")
    # noinspection PyUnboundLocalVariable
    fig.legend(
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
        loc="outside lower right",
    )
    fig.suptitle("Parameter Projection")
    fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-params.pdf")
    plt.close(fig)


def state_plots(
    true_value,
    vp_wolf_counts,
    vp_sheep_counts,
    vp_grass_counts,
    mean_init_wolves,
    mean_init_sheep,
    mean_init_grass_proportion,
    TIME_SPAN,
    SAMPLE_INTERVAL,
    NUM_CYCLES,
    mean_vec,
    cov_matrix,
    FILE_PREFIX,
    GRID_HEIGHT,
    GRID_WIDTH,
):
    vp_data = {
        "wolf": vp_wolf_counts,
        "sheep": vp_sheep_counts,
        "grass": vp_grass_counts,
    }
    max_scales = {
        "wolf": mean_init_wolves,
        "sheep": mean_init_sheep,
        "grass": mean_init_grass_proportion * GRID_HEIGHT * GRID_WIDTH,
    }
    for cycle in range(NUM_CYCLES - 1):
        fig, axs = plt.subplots(3, figsize=(6, 6), sharex=True, layout="constrained")
        for idx, state_var_name in enumerate(["wolf", "sheep", "grass"]):
            (true_value,) = axs[idx].plot(
                range(TIME_SPAN + 1),
                vp_data[state_var_name],
                label="true value",
                color="black",
            )

            (past_est_center_line,) = axs[idx].plot(
                range((cycle + 1) * SAMPLE_INTERVAL),
                transform_kf_to_intrinsic(
                    mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx], index=idx
                ),
                color="green",
                label="estimate of past",
            )
            past_est_range = axs[idx].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL),
                np.maximum(
                    0.0,
                    transform_kf_to_intrinsic(
                        mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx]
                        - np.sqrt(cov_matrix[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx, idx]),
                        index=idx,
                    ),
                ),
                np.minimum(
                    10 * max_scales[state_var_name],
                    transform_kf_to_intrinsic(
                        mean_vec[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx]
                        + np.sqrt(cov_matrix[cycle, : (cycle + 1) * SAMPLE_INTERVAL, idx, idx]),
                        index=idx,
                    ),
                ),
                color="green",
                alpha=0.35,
            )

            (future_est_before_update_center_line,) = axs[idx].plot(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                transform_kf_to_intrinsic(
                    mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx], index=idx
                ),
                color="#d5b60a",
                label="previous future estimate",
            )
            future_est_before_update_range = axs[idx].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                np.maximum(
                    0.0,
                    transform_kf_to_intrinsic(
                        mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx]
                        - np.sqrt(cov_matrix[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]),
                        index=idx,
                    ),
                ),
                np.minimum(
                    10 * max_scales[state_var_name],
                    transform_kf_to_intrinsic(
                        mean_vec[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx]
                        + np.sqrt(cov_matrix[cycle, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]),
                        index=idx,
                    ),
                ),
                color="#d5b60a",
                alpha=0.35,
            )

            (future_est_after_update_center_line,) = axs[idx].plot(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                transform_kf_to_intrinsic(
                    mean_vec[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, idx], index=idx
                ),
                label="updated future estimate",
                color="blue",
            )
            future_est_after_update_range = axs[idx].fill_between(
                range((cycle + 1) * SAMPLE_INTERVAL, TIME_SPAN + 1),
                np.maximum(
                    0.0,
                    transform_kf_to_intrinsic(
                        mean_vec[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, idx]
                        - np.sqrt(cov_matrix[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]),
                        index=idx,
                    ),
                ),
                np.minimum(
                    10 * max_scales[state_var_name],
                    transform_kf_to_intrinsic(
                        mean_vec[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, idx]
                        + np.sqrt(cov_matrix[cycle + 1, (cycle + 1) * SAMPLE_INTERVAL :, idx, idx]),
                        index=idx,
                    ),
                ),
                color="blue",
                alpha=0.35,
            )
            axs[idx].set_title(state_var_name, loc="left")
        # noinspection PyUnboundLocalVariable
        fig.legend(
            [
                true_value,
                (past_est_center_line, past_est_range),
                (future_est_before_update_center_line, future_est_before_update_range),
                (future_est_after_update_center_line, future_est_after_update_range),
            ],
            [
                true_value.get_label(),
                past_est_center_line.get_label(),
                future_est_before_update_center_line.get_label(),
                future_est_after_update_center_line.get_label(),
            ],
            loc="outside lower center",
        )
        fig.suptitle("State Projection", ha="left")
        fig.savefig(FILE_PREFIX + f"cycle-{cycle:03}-state-kfupd.pdf")
        plt.close(fig)
