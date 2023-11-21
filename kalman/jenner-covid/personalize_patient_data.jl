using LinearAlgebra
using Plots
using DifferentialEquations
using StochasticDelayDiffEq
using LaTeXStrings
using JLD2
# using StatsPlots
using Random
using NaNMath
import OnlineStats
import Distributions
import BasicInterpolators
import Interpolations

# Random.seed!(14);

################################################################################

include("model_full.jl")
include("data.jl")
include("dist.jl")

################################################################################

const patient_idx = parse(Int64, ARGS[1])
const num_samples = 10_000
const dt = 0.5

# these values need to be explored
const sird_noise = 0.0
const state_var_noise = 0.01
const param_noise = 0.005
const observation_noise = 0.1

################################################################################
# load virtual population posteriors as our parameter priors

virtual_pop_prior_log_mean, virtual_pop_prior_Σ =
    JLD2.load("dyn_param_virtual_pop.jld2", "posterior_log_mean", "posterior_Σ")

# cleanup    
virtual_pop_prior_log_mean = vec(virtual_pop_prior_log_mean)
print(max((virtual_pop_prior_Σ - virtual_pop_prior_Σ')...), " should be < about 1e-16")
virtual_pop_prior_Σ = (virtual_pop_prior_Σ + virtual_pop_prior_Σ') / 2.0

################################################################################
# create prior distributions (initial and historical)

# note that we are doing these for each time historical point, but not between 
# time points. the covariance matrix is already pretty big.

# initial condition of the state variables
initial_condition_state = [
    V0,
    S0,
    I0,
    R0,
    D0,
    MPhi_R_0,
    MPhi_I_0,
    M0,
    N0,
    T0,
    L_U_0,
    L_B_0,
    G_U_0,
    G_B_0,
    C_U_0,
    C_B_0,
    F_U_0,
    F_B_0,
]

# we need to keep a substantial history due to the delay 
const history_size = 1 + ceil(Int, max_tau_T / dt)
const unified_state_space_dimension = 18 - 1 + length(virtual_pop_prior_log_mean)

# create the historical means 
log_means = zeros(unified_state_space_dimension, history_size)
# apparently exp(-1e3) == 0.0 on the nose, so we can get around the -Inf problem this way
log_means[1, :] .= virtual_pop_prior_log_mean[1]
log_means[2:18, :] .= max.(-1e3, log.(initial_condition_state[2:end]))
log_means[19:end, :] .= virtual_pop_prior_log_mean[2:end]

# create the historical covariance matrices
prior_Σs = zeros(unified_state_space_dimension, unified_state_space_dimension, history_size)
prior_Σs[2:18, 2:18, :] .= 0.1 * I(18 - 1)
# the prior starts with V0, which messes with the block structure a little bit,
# so we have to put it in the right places
prior_Σs[1, 1, :] .= virtual_pop_prior_Σ[1, 1]
prior_Σs[1, 19:end, :] .= virtual_pop_prior_Σ[1, 2:end]
prior_Σs[19:end, 1, :] .= virtual_pop_prior_Σ[2:end, 1]
prior_Σs[19:end, 19:end, :] .= virtual_pop_prior_Σ[2:end, 2:end]

################################################################################

function compute_numerical_jacobian_log(initial_condition, history_interpolated, t_0, t_1)
    h = 0.1
    const_params = (sird_noise, state_var_noise, param_noise)

    J = zeros(unified_state_space_dimension, unified_state_space_dimension)

    for component_idx = 1:length(initial_condition)

        ic_plus = copy(initial_condition)
        ic_plus[component_idx] += h
        dde_prob_ic = remake(
            dde_prob;
            tspan = (t_0, t_1),
            u0 = exp.(ic_plus),
            h = history_interpolated,
            p = const_params,
        )
        plus_prediction =
            solve(dde_prob_ic, dde_alg, alg_hints = [:stiff], saveat = [t_0, t_1])(t_1)

        plus_prediction = max.(exp(-25), plus_prediction) # see explanation below about -25

        ic_minus = copy(initial_condition)
        ic_minus[component_idx] -= h
        dde_prob_ic = remake(
            dde_prob;
            tspan = (t_0, t_1),
            u0 = exp.(ic_minus),
            h = history_interpolated,
            p = const_params,
        )
        minus_prediction =
            solve(dde_prob_ic, dde_alg, alg_hints = [:stiff], saveat = [t_0, t_1])(t_1)
        minus_prediction = max.(exp(-25), minus_prediction) # see explanation below about -25

        J[component_idx, :] = (log.(plus_prediction) - log.(minus_prediction)) / (2 * h)
    end

    return J
end

################################################################################
# learn


function get_prediction(begin_time, end_time, log_prior)
    # create history: first index is final time
    ts = [begin_time + dt * (idx - history_size) for idx = 1:history_size]

    while true

        log_history_sample = hcat([rand(log_prior[idx]) for idx = 1:history_size]...)

        log_history_interpolator = BasicInterpolators.LinearInterpolator(
            ts,
            eachcol(log_history_sample),
            BasicInterpolators.NoBoundaries(),
        )

        function history_sample(p, t; idxs = nothing)
            history_eval = exp.(log_history_interpolator(t))
            if typeof(idxs) <: Number
                return history_eval[idxs]
            else
                return history_eval
            end
        end

        initial_condition = exp.(log_history_sample[:, end])
        # tau_T_samp <= max_tau_T
        initial_condition[end] = min(max_tau_T, initial_condition[end])

        const_params = (sird_noise, state_var_noise, param_noise)

        stochastic_solutions = true
        if stochastic_solutions
            sdde_prob_ic = remake(
                sdde_prob;
                tspan = (begin_time, end_time),
                u0 = initial_condition,
                h = history_sample,
                p = const_params,
            )
            prediction = solve(
                sdde_prob_ic,
                sdde_alg,
                alg_hints = [:stiff],
                saveat = dt,
                abstol = 1e-1, # 1e-9,
                reltol = 1e-1, # standard
                isoutofdomain = (u, p, t) -> (any(u .< 0.0)),
            )
        else
            dde_prob_ic = remake(
                dde_prob;
                tspan = (begin_time, end_time),
                u0 = initial_condition,
                h = history_sample,
                p = const_params,
            )
            prediction = solve(
                dde_prob_ic,
                dde_alg,
                alg_hints = [:stiff],
                saveat = dt,
                isoutofdomain = (u, p, t) -> (any(u .< 0.0)),
            )
        end

        # if prediction.retcode != :success
        #     continue
        # end

        if any(isnan.(hcat(prediction.u...))) | any(isinf.(hcat(prediction.u...)))
            continue
        end

        if minimum(hcat(prediction.u...)) >= 0.0

            virtual_population_loss = covid_minimizing_fun_severe(prediction)
            accept = rand() <= exp(-virtual_population_loss)
            if accept
                return prediction, t -> history_sample(p, t; idxs = nothing)
            else
                println(exp(-virtual_population_loss))
            end
        end
        # otherwise go again
    end
end # get_prediction


plts = []
for idx = 1:18
    local subplt
    subplt = Plots.plot()

    title!(subplt, var_meaning[idx])
    xlabel!(subplt, L"t")
    ylabel!(subplt, "")
    push!(plts, subplt)
end

for sequence_idx = 1:length(data_times[patient_idx])

    # determine the current time interval to test
    begin_time = Float64(sequence_idx <= 1 ? 0 : data_times[patient_idx][sequence_idx-1])
    end_time = Float64(data_times[patient_idx][sequence_idx])

    # param_samples = zeros(num_samples, length(prior_log_means))

    log_prior =
        [CustomNormal(log_means[:, idx], prior_Σs[:, :, idx]) for idx = 1:history_size]

    # we need to compute mean/covariance for both a history (used in the Kalman update)
    # and to plot. These will often be quite different.

    history_sample_means = zeros(unified_state_space_dimension, history_size)
    history_sample_covs_unscaled =
        zeros(unified_state_space_dimension, unified_state_space_dimension, history_size)
    history_times = [end_time + dt * (idx - history_size) for idx = 1:history_size]

    num_plot_points = ceil(Int, (end_time - begin_time) / dt)
    plot_sample_means = zeros(unified_state_space_dimension, num_plot_points)
    plot_sample_covs_unscaled =
        zeros(unified_state_space_dimension, unified_state_space_dimension, num_plot_points)
    plot_times = LinRange(begin_time, end_time, num_plot_points)


    for sample_idx = 1:num_samples

        if sample_idx % 10 == 0
            println(sequence_idx, " : ", sample_idx)
        end

        prediction, history_samp = get_prediction(begin_time, end_time, log_prior)

        # Welford's online algorithm for mean and covariance calculation. See Knuth Vol 2, pg 232
        for (time_idx, t) in enumerate(history_times)
            # One can use exp(-1e3) == 0.0, to get around the -Inf problem (leads to NaNs)
            # However, -1e3 skews the mean pretty badly, so we use exp(-25) ~ 1.4e-11, which
            # is close enough 
            # -25 ~ minimum(log.(y0)[log.(y0) .> -Inf])  - log(100)
            if t <= begin_time
                sample = max.(-25, log.(history_samp(t)))
            else
                sample = max.(-25, log.(prediction(t)))
            end

            old_mean = copy(history_sample_means[:, time_idx])
            history_sample_means[:, time_idx] +=
                (sample - history_sample_means[:, time_idx]) / sample_idx
            # use variant formula (mean of two of the standard updates) to 
            # increase symmetry in the fp error (1e-18) range
            history_sample_covs_unscaled[:, :, time_idx] +=
                (
                    (sample - history_sample_means[:, time_idx]) * (sample - old_mean)' +
                    (sample - old_mean) * (sample - history_sample_means[:, time_idx])'
                ) / 2.0
        end

        # Welford's online algorithm for mean and variance calculation. See Knuth Vol 2, pg 232
        for (time_idx, t) in enumerate(plot_times)
            # One can use exp(-1e3) == 0.0, to get around the -Inf problem (leads to NaNs)
            # However, -1e3 skews the mean pretty badly, so we use exp(-25) ~ 1.4e-11, which
            # is close enough
            # -25 ~ minimum(log.(y0)[log.(y0) .> -Inf])  - log(100)
            if t <= begin_time
                sample = max.(-25, log.(history_samp(t)))
            else
                sample = max.(-25, log.(prediction(t)))
            end

            old_mean = copy(plot_sample_means[:, time_idx])
            plot_sample_means[:, time_idx] +=
                (sample - plot_sample_means[:, time_idx]) / sample_idx
            # use variant formula (mean of two of the standard updates) to 
            # increase symmetry in the fp error (1e-18) range
            plot_sample_covs_unscaled[:, :, time_idx] +=
                (
                    (sample - plot_sample_means[:, time_idx]) * (sample - old_mean)' +
                    (sample - old_mean) * (sample - plot_sample_means[:, time_idx])'
                ) / 2.0
        end
    end

    ################################################################################
    # plots

    plot_sample_covs = plot_sample_covs_unscaled / (num_samples - 1)

    for idx = 1:18
        plot!(
            plts[idx],
            plot_times,
            exp.(plot_sample_means[idx, :]),
            ribbon = (
                exp.(plot_sample_means[idx, :] - sqrt.(plot_sample_covs[idx, idx, :])),
                exp.(plot_sample_means[idx, :] + sqrt.(plot_sample_covs[idx, idx, :])),
            ),
            fillalpha = 0.35,
            lc = :black,
            label = "",
        )

        if idx == 1
            plot!(
                plts[idx],
                [data_times[patient_idx][sequence_idx]],
                [viral_loads[patient_idx][sequence_idx]],
                seriestype = :scatter,
                mc = :red,
                label = "",
            )
        end

        if xlims(plts[idx]) == (0.0, 1.0) && ylims(plts[idx]) == (0.0, 1.0)
            # initially empty
            xlims!(plts[idx], (0.0, data_times[patient_idx][sequence_idx] + 0.25))
            ymax = maximum(
                exp.(plot_sample_means[idx, :] + sqrt.(plot_sample_covs[idx, idx, :])),
            )
            if idx == 1
                ymax = max(ymax, viral_loads[patient_idx][sequence_idx])
            end
            ylims!(plts[idx], (0.0, ymax))
        else
            xlims!(plts[idx], (0.0, data_times[patient_idx][sequence_idx] + 0.25))
            ymax = max(
                ylims(plts[idx])[2],
                maximum(
                    exp.(plot_sample_means[idx, :] + sqrt.(plot_sample_covs[idx, idx, :])),
                ),
            )
            if idx == 1
                ymax = max(ymax, viral_loads[patient_idx][sequence_idx])
            end
            ylims!(plts[idx], (0.0, ymax))
        end
    end
    plt = plot(plts..., layout = (6, 3), size = (1000, 1500))
    println("plot $sequence_idx created")
    Plots.savefig(plt, "personalization_p$patient_idx-s$sequence_idx.pdf")
    println("plot $sequence_idx saved")

    ################################################################################
    # Kalman updates

    history_sample_covs = history_sample_covs_unscaled / (num_samples - 1)

    # observation matrix
    H = zeros(unified_state_space_dimension)'
    H[1] = 1.0

    # observation noise covariance matrix
    R = hcat([observation_noise])

    # state noise covariance matrix
    Q = log_noise_matrix(sird_noise, state_var_noise, param_noise)

    smoothed_means = zeros(size(history_sample_means))
    smoothed_covs = zeros(size(history_sample_covs))

    # kalman update
    # for ref: m^- = history_sample_means[:, end], P^- = history_sample_covs[:, :, end]
    # S = (H * history_sample_covs[:, :, end] * H') .+ R
    # K = history_sample_covs[:, :, end] * H' * inv(S) # pinv?
    # v = log.(viral_loads[patient_idx][sequence_idx]) - H * history_sample_means[:, end]
    # smoothed_means[:, end] = history_sample_means[:, end] + K * v
    # smoothed_covs[:, :, end] = history_sample_covs[:, :, end] - K * S * K'

    # rewrite to avoid inv/pinv calculations, prefer the \ operator
    # the Kalman gain matrix K is possibly useful to calculate, but 
    # not nesc. at this point
    S = (H * history_sample_covs[:, :, end] * H') .+ R
    v = log.(viral_loads[patient_idx][sequence_idx]) - H * history_sample_means[:, end]

    smoothed_means[:, end] =
        history_sample_means[:, end] + history_sample_covs[:, :, end] * H' * (S \ hcat(v))
    smoothed_covs[:, :, end] =
        history_sample_covs[:, :, end] * (
            I(unified_state_space_dimension) -
            H' * (S' \ (H * history_sample_covs[:, :, end]'))
        )

    # create an interpolator for history, used in the compuation of the numerical 
    # jacobian. This just uses the means, avoiding stochasticity. (which the numerical
    # jacobian also does)
    function create_interpolator(hist)
        hist_interpolator =
            BasicInterpolators.LinearInterpolator(history_times, eachcol(hist))
        min_time = minimum(history_times)
        max_time = maximum(history_times)
        function hist_interpolated(p, t; idxs = nothing)
            t = max(t, min_time)
            t = min(t, max_time)
            history_eval = hist_interpolator(t)
            if typeof(idxs) <: Number
                return history_eval[idxs]
            else
                return history_eval
            end
        end
        return hist_interpolated
    end

    history_sample_means_interpolated = create_interpolator(history_sample_means)

    # Rauch-Tung-Striebel smoothing
    for hist_idx = (history_size-1):-1:1

        A = compute_numerical_jacobian_log(
            history_sample_means[:, hist_idx],
            history_sample_means_interpolated,
            history_times[hist_idx],
            history_times[hist_idx+1],
        )

        # A = hcat(1.0 * I(unified_state_space_dimension))
        # if history_times[hist_idx] > 0.0
        #     # model only does updates on forward time
        #     A +=
        #         dt *
        #         diagm(exp.(-history_sample_means[:, hist_idx])) *
        #         covid_model_jacobian(
        #             exp.(history_sample_means[:, hist_idx]),
        #             history_sample_means_interpolated,
        #             (sird_noise, state_var_noise, param_noise),
        #             end_time - dt * hist_idx,
        #         )
        # end

        G =
            history_sample_covs[:, :, hist_idx] *
            A' *
            pinv(A * history_sample_covs[:, :, hist_idx] * A' + Q) # inv/pinv?

        smoothed_means[:, hist_idx] =
            history_sample_means[:, hist_idx] +
            G * (smoothed_means[:, hist_idx+1] - A * history_sample_means[:, hist_idx])

        # smoothed_means[:, hist_idx] =
        #     history_sample_means[:, hist_idx] +
        #     history_sample_covs[:, :, hist_idx] *
        #     A' *
        #     (
        #         (A * history_sample_covs[:, :, hist_idx] * A' + Q) \
        #         (smoothed_means[:, hist_idx+1] - A * history_sample_means[:, hist_idx])
        #     )


        smoothed_covs[:, :, hist_idx] =
            history_sample_covs[:, :, hist_idx] +
            G *
            (
                smoothed_covs[:, :, hist_idx+1] -
                A * history_sample_covs[:, :, hist_idx] * A' - Q
            ) *
            G'

        # smoothed_covs[:, :, hist_idx] =
        #     history_sample_covs[:, :, hist_idx] * (
        #         I +
        #         A' * (A * history_sample_covs[:, :, hist_idx] * A' + Q) \ (
        #             (
        #                 smoothed_covs[:, :, hist_idx+1] -
        #                 A * history_sample_covs[:, :, hist_idx] * A' - Q
        #             ) * (
        #                 (A * history_sample_covs[:, :, hist_idx] * A' + Q)' \
        #                 (A * history_sample_covs[:, :, hist_idx]')
        #             )
        #         )
        #     )

    end

    # numerical cleanup b/c julia is persnickety
    for hist_idx = 1:history_size
        smoothed_covs[:, :, hist_idx] .=
            (smoothed_covs[:, :, hist_idx] + smoothed_covs[:, :, hist_idx]') / 2.0
        smallest_eig = LinearAlgebra.eigmin(smoothed_covs[:, :, hist_idx])
        if smallest_eig <= 0.0
            smoothed_covs[:, :, hist_idx] += (0.001 - smallest_eig) * I
        end
    end

    # for idx = 1:18
    #     plot!(
    #         plts[idx],
    #         history_times,
    #         exp.(smoothed_means[idx, :]),
    #         ribbon = (
    #             exp.(smoothed_means[idx, :] - sqrt.(smoothed_covs[idx, idx, :])),
    #             exp.(smoothed_means[idx, :] + sqrt.(smoothed_covs[idx, idx, :])),
    #         ),
    #         fillalpha = 0.35,
    #         lc = :red,
    #         label = "",
    #     )

    # end
    # plt = plot(plts..., layout = (6, 3), size = (1000, 1500))






    # copy for next round
    log_means .= smoothed_means
    prior_Σs .= smoothed_covs

    JLD2.save(
        "kalman-update-$sequence_idx.jld2",
        Dict("log_means" => log_means, "prior_Σs" => prior_Σs),
    )

    ################################################################################

end

# log_param_samples = log.(param_samples)
# posterior_log_mean = mean(log_param_samples, dims = 1)
# posterior_Σ =
#     (log_param_samples .- posterior_log_mean)' * (log_param_samples .- posterior_log_mean) /
#     num_samples


# JLD2.@save "dyn_param_virtual_pop.jld2" posterior_log_mean posterior_Σ
