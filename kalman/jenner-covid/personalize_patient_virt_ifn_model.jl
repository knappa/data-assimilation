using LinearAlgebra
using Plots
using DifferentialEquations
using StochasticDelayDiffEq
using LaTeXStrings
using JLD2
using Random
using NaNMath
import BasicInterpolators
import SciMLBase

# Random.seed!(14);

################################################################################

include("model_ifn.jl")
include("data.jl")
include("dist.jl")
include("util.jl")

################################################################################

const virt_patient_tspan = (0.0, 10.0)
const num_samples = 1_000
const dt = 0.1
const sample_dt = 1 # probably should be multiple of dt
const sample_idx = 1

# these values need to be explored
const sird_noise = 0.01
const state_var_noise = 0.01
const param_noise = 0.01
const observation_noise = 0.01

################################################################################
# load virtual population posteriors as our parameter priors

virtual_pop_prior_log_mean, virtual_pop_prior_logΣ =
    JLD2.load("dyn_param_virtual_pop.jld2", "posterior_log_mean", "posterior_Σ")

# cleanup    
virtual_pop_prior_log_mean = vec(virtual_pop_prior_log_mean)
# print(max((virtual_pop_prior_Σ - virtual_pop_prior_Σ')...), " should be < about 1e-16")
virtual_pop_prior_logΣ = Symmetric(virtual_pop_prior_logΣ)

# select parts relevant to sub-model
virtual_pop_prior_log_mean = virtual_pop_prior_log_mean[[1, 2, 5, 6, 10]]
virtual_pop_prior_logΣ = virtual_pop_prior_logΣ[[1, 2, 5, 6, 10], [1, 2, 5, 6, 10]]

# de-log it
virtual_pop_prior_mean = exp.(virtual_pop_prior_log_mean + diag(virtual_pop_prior_logΣ) / 2)
virtual_pop_prior_Σ =
    (virtual_pop_prior_log_mean * virtual_pop_prior_log_mean') .*
    exp.(virtual_pop_prior_logΣ .- 1)

################################################################################
# create prior distributions (initial and historical)

# note that we are doing these for each time historical point, but not between 
# time points. the covariance matrix is already pretty big.

# initial condition of the state variables
initial_condition_state = [V0, S0, I0, R0, D0, F_U_0, F_B_0]

# we need to keep a substantial history due to the delay 
const history_size = 1 + ceil(Int, max_tau_T / dt)
const state_space_dim = length(initial_condition_state)
const unified_state_space_dimension =
    state_space_dim + length(virtual_pop_prior_log_mean) - 1

# create the historical means 
means = zeros(unified_state_space_dimension, history_size)
means[1, end] = virtual_pop_prior_mean[1] # only last mean gets the virus
means[2:state_space_dim, :] .= initial_condition_state[2:end]
means[state_space_dim+1:end, :] .= virtual_pop_prior_mean[2:end]

# create the historical covariance matrices
prior_Σs = zeros(unified_state_space_dimension, unified_state_space_dimension, history_size)
prior_Σs[2:state_space_dim, 2:state_space_dim, :] .= 0.1 * I(state_space_dim - 1)
# the prior starts with V0, which messes with the block structure a little bit,
# so we have to put it in the right places
prior_Σs[1, 1, :] .= virtual_pop_prior_Σ[1, 1]
prior_Σs[1, state_space_dim+1:end, :] .= virtual_pop_prior_Σ[1, 2:end]
prior_Σs[state_space_dim+1:end, 1, :] .= virtual_pop_prior_Σ[2:end, 1]
prior_Σs[state_space_dim+1:end, state_space_dim+1:end, :] .=
    virtual_pop_prior_Σ[2:end, 2:end]

################################################################################

function compute_numerical_jacobian(
    initial_condition,
    sampled_history_interpolated,
    t_0,
    t_1,
)
    const_params = (sird_noise, state_var_noise, param_noise)
    history_interpolated(p, t; idxs = nothing) =
        abs.(sampled_history_interpolated(p, t; idxs = idxs))
    J = zeros(unified_state_space_dimension, unified_state_space_dimension)

    for component_idx in eachindex(initial_condition)

        # ensure that +/- h never sends us negative
        h = min(0.1, initial_condition[component_idx] / 2.0)
        if (h == 0.0) | issubnormal(h)
            continue
        end

        ic_plus = copy(initial_condition)
        ic_plus[component_idx] += h
        dde_prob_ic = remake(
            dde_prob;
            tspan = (t_0, t_1),
            u0 = abs.(ic_plus),
            h = history_interpolated,
            p = const_params,
        )
        plus_prediction = solve(
            dde_prob_ic,
            dde_alg,
            alg_hints = [:stiff],
            saveat = [t_0, t_1],
            isoutofdomain = (u, p, t) -> (any(u .< 0.0)),
        )(
            t_1,
        )

        ic_minus = copy(initial_condition)
        ic_minus[component_idx] -= h
        dde_prob_ic = remake(
            dde_prob;
            tspan = (t_0, t_1),
            u0 = abs.(ic_minus),
            h = history_interpolated,
            p = const_params,
        )
        minus_prediction = solve(
            dde_prob_ic,
            dde_alg,
            alg_hints = [:stiff],
            saveat = [t_0, t_1],
            isoutofdomain = (u, p, t) -> (any(u .< 0.0)),
        )(
            t_1,
        )

        J[component_idx, :] = (plus_prediction - minus_prediction) / (2 * h)
    end

    return J
end

################################################################################

function get_prediction(begin_time, end_time, prior)
    # create history: first index is final time
    ts = [begin_time + dt * (idx - history_size) for idx = 1:history_size]

    while true

        history_sample_arr = hcat([abs.(rand(prior[idx])) for idx = 1:history_size]...)

        history_interpolator = BasicInterpolators.LinearInterpolator(
            ts,
            eachcol(history_sample_arr),
            BasicInterpolators.NoBoundaries(),
        )

        function history_sample(p, t; idxs = nothing)
            history_eval = history_interpolator(t)
            if typeof(idxs) <: Number
                return history_eval[idxs]
            else
                return history_eval
            end
        end

        initial_condition = history_sample_arr[:, end]
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
                # abstol = 1e-1, # 1e-9,
                # reltol = 1e-1, # standard
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

        if !SciMLBase.successful_retcode(prediction)
            continue
        end

        if any(isnan.(hcat(prediction.u...))) | any(isinf.(hcat(prediction.u...)))
            continue
        end

        if minimum(hcat(prediction.u...)) >= 0.0

            # virtual_population_loss = covid_minimizing_fun_severe(prediction)
            # accept = rand() <= exp(-virtual_population_loss)
            accept = true
            if accept
                return prediction, t -> history_sample(p, t; idxs = nothing)
            else
                println(exp(-virtual_population_loss))
            end
        end
        # otherwise go again
    end
end # get_prediction

################################################################################

prior = [CustomNormal(means[:, idx], prior_Σs[:, :, idx]) for idx = 1:history_size]

virtual_patient_trajectory, virtual_patient_history =
    get_prediction(virt_patient_tspan[1], virt_patient_tspan[2], prior)

JLD2.save(
    "kalman-update-virt-traj.jld2",
    Dict(
        "virtual_patient_trajectory" => virtual_patient_trajectory,
        "virtual_patient_history" => virtual_patient_history,
    ),
)

################################################################################

state_plts = []
for idx = 1:state_space_dim
    local subplt
    subplt = plot(virtual_patient_trajectory, idxs = idx, lc = :black, label = "")

    title!(subplt, var_meaning[idx])
    xlabel!(subplt, L"t")
    ylabel!(subplt, "")
    push!(state_plts, subplt)
end
plt = plot(state_plts..., layout = (3, 3), size = (1000, 500))
Plots.savefig(plt, "personalization-virt-s0-state.pdf")
println("initial state plot saved")

param_plts = []
for idx = state_space_dim+1:unified_state_space_dimension
    local subplt
    subplt = plot(virtual_patient_trajectory, idxs = idx, lc = :black, label = "")

    title!(subplt, var_meaning[idx])
    xlabel!(subplt, L"t")
    ylabel!(subplt, "")
    push!(param_plts, subplt)
end
plt = plot(param_plts..., layout = (2, 2), size = (600, 500))
Plots.savefig(plt, "personalization-virt-s0-param.pdf")
println("initial param plot saved")


################################################################################

time_intervals = Vector(virt_patient_tspan[1]:sample_dt:virt_patient_tspan[2])

for interval_idx = 1:length(time_intervals)-1

    # determine the current time interval to test
    begin_time = time_intervals[interval_idx]
    end_time = time_intervals[interval_idx+1]

    local prior
    prior = [CustomNormal(means[:, idx], prior_Σs[:, :, idx]) for idx = 1:history_size]

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

        if sample_idx % 100 == 0
            println(interval_idx, " ::: ", sample_idx)
        end

        prediction, history_samp = get_prediction(begin_time, end_time, prior)

        # Welford's online algorithm for mean and covariance calculation. See Knuth Vol 2, pg 232
        for (time_idx, t) in enumerate(history_times)
            if t <= begin_time
                sample = abs.(history_samp(t))
            else
                sample = abs.(prediction(t))
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
            if t <= begin_time
                sample = abs.(history_samp(t))
            else
                sample = abs.(prediction(t))
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

    for component_idx = 1:state_space_dim
        plot!(
            state_plts[component_idx],
            plot_times,
            plot_sample_means[component_idx, :],
            ribbon = (
                plot_sample_means[component_idx, :] -
                sqrt.(max.(0.0, plot_sample_covs[component_idx, component_idx, :])),
                plot_sample_means[component_idx, :] +
                sqrt.(max.(0.0, plot_sample_covs[component_idx, component_idx, :])),
            ),
            fillalpha = 0.35,
            linecolor = :red,
            fillcolor = :grey,
            label = "",
        )

        if component_idx == sample_idx
            plot!(
                state_plts[component_idx],
                [plot_times[end]],
                [virtual_patient_trajectory(plot_times[end])[sample_idx]],
                seriestype = :scatter,
                mc = :red,
                label = "",
            )
        end

    end
    plt = plot(state_plts..., layout = (3, 3), size = (1000, 500))
    Plots.savefig(plt, "personalization-virt-s$interval_idx-state.pdf")
    println("state plot $interval_idx saved")

    for component_idx = state_space_dim+1:unified_state_space_dimension
        plot!(
            param_plts[component_idx-state_space_dim],
            plot_times,
            plot_sample_means[component_idx, :],
            ribbon = (
                plot_sample_means[component_idx, :] -
                sqrt.(max.(0.0, plot_sample_covs[component_idx, component_idx, :])),
                plot_sample_means[component_idx, :] +
                sqrt.(max.(0.0, plot_sample_covs[component_idx, component_idx, :])),
            ),
            fillalpha = 0.35,
            linecolor = :red,
            fillcolor = :grey,
            label = "",
        )
    end
    plt = plot(param_plts..., layout = (2, 2), size = (600, 500))
    Plots.savefig(plt, "personalization-virt-s$interval_idx-param.pdf")
    println("param plot $interval_idx saved")

    ################################################################################
    # Kalman updates

    history_sample_covs = history_sample_covs_unscaled / (num_samples - 1)
    # numerical cleanup
    for hist_idx = 1:history_size
        history_sample_covs[:, :, hist_idx] =
            pos_def_projection(history_sample_covs[:, :, hist_idx])
    end

    # observation matrix
    H = zeros(unified_state_space_dimension)'
    H[sample_idx] = 1.0

    # observation noise covariance matrix
    R = hcat([observation_noise])

    # state noise covariance matrix
    Q = noise_matrix(sird_noise, state_var_noise, param_noise, history_sample_means[:, end])

    smoothed_means = zeros(size(history_sample_means))
    smoothed_covs = zeros(size(history_sample_covs))

    # kalman update
    # for ref: m^- = history_sample_means[:, end], P^- = history_sample_covs[:, :, end]
    S = (H * history_sample_covs[:, :, end] * H') .+ R
    K = history_sample_covs[:, :, end] * H' * pinv(S) # pinv?
    v =
        virtual_patient_trajectory(time_intervals[interval_idx+1])[sample_idx] -
        H * history_sample_means[:, end]
    smoothed_means[:, end] = history_sample_means[:, end] + K * v
    smoothed_covs[:, :, end] = Symmetric(history_sample_covs[:, :, end] - K * S * K')

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
            history_eval = abs.(hist_interpolator(t))
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

        if history_times[hist_idx] <= 0.0
            # do not attempt learning for "pre-history"
            smoothed_means[:, hist_idx] = history_sample_means[:, hist_idx]
            smoothed_means[1, hist_idx] = 0.0 # no virus in prehistory
            smoothed_covs[:, :, hist_idx] = history_sample_covs[:, :, hist_idx]
        else

            A = compute_numerical_jacobian(
                history_sample_means[:, hist_idx],
                history_sample_means_interpolated,
                history_times[hist_idx],
                history_times[hist_idx+1],
            )

            G =
                history_sample_covs[:, :, hist_idx] *
                A' *
                pinv(A * history_sample_covs[:, :, hist_idx] * A' + Q) # inv/pinv?

            smoothed_means[:, hist_idx] =
                abs.(
                    history_sample_means[:, hist_idx] +
                    G *
                    (smoothed_means[:, hist_idx+1] - A * history_sample_means[:, hist_idx])
                )

            smoothed_covs[:, :, hist_idx] =
                history_sample_covs[:, :, hist_idx] +
                G *
                (
                    smoothed_covs[:, :, hist_idx+1] -
                    A * history_sample_covs[:, :, hist_idx] * A' - Q
                ) *
                G'

        end
    end

    # numerical cleanup b/c julia is persnickety
    for hist_idx = 1:history_size
        smoothed_covs[:, :, hist_idx] = pos_def_projection(smoothed_covs[:, :, hist_idx])
    end

    plot_history = true
    if plot_history
        history_plots = []
        for idx = 1:state_space_dim
            subplt = plot(
                history_times,
                smoothed_means[idx, :],
                ribbon = (
                    smoothed_means[idx, :] - sqrt.(max.(0.0, smoothed_covs[idx, idx, :])),
                    smoothed_means[idx, :] + sqrt.(max.(0.0, smoothed_covs[idx, idx, :])),
                ),
                fillalpha = 0.35,
                fc = :red,
                lc = :red,
                label = "",
            )
            push!(history_plots, subplt)

        end
        local plt
        plt = plot(state_plts..., layout = (3, 3), size = (1000, 500))
        Plots.savefig(plt, "personalization-virt-s$interval_idx-state-hist.pdf")
        # println("state plot $interval_idx saved")
    end

    # copy for next round
    means .= smoothed_means
    prior_Σs .= smoothed_covs

    JLD2.save(
        "kalman-update-$interval_idx.jld2",
        Dict("means" => means, "prior_Σs" => prior_Σs),
    )

    ################################################################################

end
