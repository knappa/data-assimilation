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
using HDF5
using ArgParse
using ProgressBars

################################################################################

function transform(v; idx = nothing)
    if typeof(idx) <: Number
        if idx == 2
            return log.(-1 .+ max.(1 + 1e-15, exp.(v)))
        elseif idx == 4
            return v .* 100
        else
            return v
        end
    else
        w = copy(v)
        # w[2,:] = log.(max.(1e-300,v[2,:]))
        w[2, :] = log.(-1 .+ max.(1 + 1e-15, exp.(v[2, :])))
        w[4, :] = v[4, :] .* 100
        return w
    end
end

function inv_transform(v; idx = nothing)
    if typeof(idx) <: Number
        if idx == 2
            return log.(1 .+ exp.(v))
        elseif idx == 4
            return v ./ 100
        else
            return v
        end
    else
        w = max.(0.0, v)
        # w[2,:] = exp.(v[2,:])
        w[2, :] = log.(1 .+ exp.(v[2, :]))
        w[4, :] = v[4, :] ./ 100
        return w
    end
end

################################################################################
# command line parameters

s = ArgParseSettings()

@add_arg_table s begin
    "--seed"
    help = "random seed, default is system default"
    arg_type = Int
    default = -1
    "--sird_noise"
    help = "multiplicative noise scale for SIRD state variables"
    arg_type = Float64
    default = 0.01
    "--state_var_noise"
    help = "multiplicative noise scale for other state variables"
    arg_type = Float64
    default = 0.01
    "--param_noise"
    help = "multiplicative noise scale for parameters"
    arg_type = Float64
    default = 0.01
    "--observation_noise"
    help = "linear noise scale for observations"
    arg_type = Float64
    default = 0.01
    "--prefix"
    help = "filename prefix for output"
    arg_type = String
    default = ""
end

parsed_args = parse_args(s)

if parsed_args["seed"] != -1
    Random.seed!(parsed_args["seed"])
end

const sird_noise = parsed_args["sird_noise"]
const state_var_noise = parsed_args["state_var_noise"]
const param_noise = parsed_args["param_noise"]
const observation_noise = parsed_args["observation_noise"]

const filename_prefix =
    length(parsed_args["prefix"]) > 0 ? (parsed_args["prefix"] * "-") : ""

################################################################################

include("model_simple.jl")
include("data.jl")
include("dist.jl")
include("util.jl")

################################################################################
# load virtual population posteriors as our parameter priors

virtual_pop_prior_mean, virtual_pop_prior_Σ =
    JLD2.load("virtual_population_statistics.jld2", "posterior_mean", "posterior_Σ")

# cleanup    
virtual_pop_prior_mean = vec(virtual_pop_prior_mean)
virtual_pop_prior_Σ = Symmetric(virtual_pop_prior_Σ)

# select parts relevant to sub-model
virtual_pop_prior_mean = virtual_pop_prior_mean[[1, 2]]
virtual_pop_prior_Σ = virtual_pop_prior_Σ[[1, 2], [1, 2]]

################################################################################
# create prior distributions (initial and historical)

# note that we are doing these for each time historical point, but not between 
# time points. the covariance matrix is already pretty big.

# initial condition of the state variables
initial_condition_state = [V0, S0, I0, D0]

# we need to keep a substantial history due to the delay 
const dt = 0.1
const history_size = 1 + ceil(Int, max_tau_T / dt)
const state_space_dim = length(initial_condition_state)
const unified_state_space_dimension = state_space_dim + length(virtual_pop_prior_mean) - 1

const simulation_end_time = 10.0
const virt_patient_tspan = (0.0, simulation_end_time)
const num_samples =
    Integer(100 * unified_state_space_dimension * (unified_state_space_dimension - 1) / 2.0)
const sample_dt = 1 # probably should be multiple of dt
const sample_idx = 1

println("number of samples $num_samples")

# smallest allowed eigenvalue in numerical cleanup
const eigenvalue_epsilon = 1e-6


# create the historical means 
means = zeros(unified_state_space_dimension, history_size)
means[1, end] = virtual_pop_prior_mean[1] # only last mean gets the virus
means[2:state_space_dim, :] .= initial_condition_state[2:end]
means[state_space_dim+1:end, :] .= virtual_pop_prior_mean[2:end]

means = transform(means)

# create the historical covariance matrices
prior_Σs = zeros(unified_state_space_dimension, unified_state_space_dimension, history_size)
prior_Σs[2:state_space_dim, 2:state_space_dim, :] .=
    diagm((0.1 * initial_condition_state[2:end]) .^ 2)
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
        sampled_history_interpolated(p, t; idxs = idxs)
    J = zeros(unified_state_space_dimension, unified_state_space_dimension)

    initial_condition = inv_transform(initial_condition)

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
            u0 = ic_plus,
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
            u0 = ic_minus,
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

        nan_on_plus = any(isnan.(plus_prediction))
        nan_on_minus = any(isnan.(minus_prediction))
        if nan_on_plus & nan_on_minus
            # can't compute, use default
            J[component_idx, :] .= 0.0
        elseif nan_on_plus | nan_on_minus

            # compute one sided derivative
            dde_prob_ic = remake(
                dde_prob;
                tspan = (t_0, t_1),
                u0 = initial_condition,
                h = history_interpolated,
                p = const_params,
            )
            zero_prediction = solve(
                dde_prob_ic,
                dde_alg,
                alg_hints = [:stiff],
                saveat = [t_0, t_1],
                isoutofdomain = (u, p, t) -> (any(u .< 0.0)),
            )(
                t_1,
            )

            if any(isnan.(zero_prediction))
                # can't compute, use default
                J[component_idx, :] .= 0.0
            elseif nan_on_plus
                J[component_idx, :] = (zero_prediction - minus_prediction) / h
            elseif nan_on_minus
                J[component_idx, :] = (plus_prediction - zero_prediction) / h
            end

        else
            # both plus and minus predictions are fine, compute the two sided derivative
            J[component_idx, :] = (plus_prediction - minus_prediction) / (2 * h)
        end
    end

    return J
end

################################################################################

function get_prediction(begin_time, end_time, prior)
    # create history: first index is final time
    ts = [begin_time + dt * (idx - history_size) for idx = 1:history_size]

    while true

        history_sample_arr =
            inv_transform(hcat([rand(prior[idx]) for idx = 1:history_size]...))

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
        # initial_condition[end] = min(max_tau_T, initial_condition[end])

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

h5open(filename_prefix * "data.hdf5", "w") do fid
    g = create_group(fid, "virtual_patient")

    dset = create_dataset(g, "trajectory_t", Float64, size(virtual_patient_trajectory.t))
    write(dset, virtual_patient_trajectory.t)

    virtual_patient_trajectory_u = vcat(virtual_patient_trajectory.u'...)
    dset = create_dataset(g, "trajectory_u", Float64, size(virtual_patient_trajectory_u))
    write(dset, virtual_patient_trajectory_u)

    history_t = Vector((-max_tau_T):dt:0)
    dset = create_dataset(g, "history_t", Float64, size(history_t))
    write(dset, history_t)

    history_u = vcat([virtual_patient_history(t) for t in history_t]'...)
    dset = create_dataset(g, "history_u", Float64, size(history_u))
    write(dset, history_u)
end

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
plt = plot(state_plts..., layout = (2, 2), size = (700, 500))
Plots.savefig(plt, filename_prefix * "s0-state.pdf")
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
plt = plot(param_plts..., layout = (1, 1), size = (300, 250))
Plots.savefig(plt, filename_prefix * "s0-param.pdf")
println("initial param plot saved")

################################################################################

const time_intervals = Vector(virt_patient_tspan[1]:sample_dt:virt_patient_tspan[2])

const prediction_size = 1 + ceil(Int, (virt_patient_tspan[2] - virt_patient_tspan[1]) / dt)
prediction_ts = Vector(virt_patient_tspan[1]:dt:virt_patient_tspan[2])
mean_record =
    zeros(unified_state_space_dimension, prediction_size, length(time_intervals) - 1)
Σ_record_unscaled = zeros(
    unified_state_space_dimension,
    unified_state_space_dimension,
    prediction_size,
    length(time_intervals) - 1,
)

################################################################################

for interval_idx in ProgressBar(1:length(time_intervals)-1)

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

    # plot from now to the end of the simulation
    num_plot_points = ceil(Int, (end_time - begin_time) / dt)
    plot_sample_means = zeros(unified_state_space_dimension, num_plot_points)
    plot_sample_covs_unscaled =
        zeros(unified_state_space_dimension, unified_state_space_dimension, num_plot_points)
    plot_times = LinRange(begin_time, end_time, num_plot_points)

    mean_record[:, 1+(interval_idx-1)*ceil(Int, sample_dt / dt):end, interval_idx:end] .=
        0.0

    for sample_idx in ProgressBar(1:num_samples)

        # previously, predict up to the end time of the interval, now change to the 
        # end time of the total simulation
        prediction, history_samp = get_prediction(begin_time, simulation_end_time, prior)

        # Welford's online algorithm for mean and covariance calculation. See Knuth Vol 2, pg 232
        for (time_idx, t) in enumerate(history_times)
            if t <= begin_time
                sample = transform(history_samp(t))
            else
                sample = transform(prediction(t))
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
                sample = transform(history_samp(t))
            else
                sample = transform(prediction(t))
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

        # Welford's online algorithm for mean and variance calculation. See Knuth Vol 2, pg 232
        for (offset_time_idx, t) in
            enumerate(prediction_ts[1+(interval_idx-1)*ceil(Int, sample_dt / dt):end])
            time_idx = (interval_idx - 1) * ceil(Int, sample_dt / dt) + offset_time_idx
            if t <= begin_time
                sample = transform(history_samp(t))
            else
                sample = transform(prediction(t))
            end

            for future_interval_idx = interval_idx:length(time_intervals)-1
                old_mean = copy(mean_record[:, time_idx, future_interval_idx])
                mean_record[:, time_idx, future_interval_idx] +=
                    (sample - mean_record[:, time_idx, future_interval_idx]) / sample_idx
                # use variant formula (mean of two of the standard updates) to 
                # increase symmetry in the fp error (1e-18) range
                Σ_record_unscaled[:, :, time_idx, future_interval_idx] +=
                    (
                        (sample - mean_record[:, time_idx, future_interval_idx]) *
                        (sample - old_mean)' +
                        (sample - old_mean) *
                        (sample - mean_record[:, time_idx, future_interval_idx])'
                    ) / 2.0
            end
        end

    end

    ################################################################################
    # plots

    plot_sample_covs = plot_sample_covs_unscaled / (num_samples - 1)

    for component_idx = 1:state_space_dim
        component_stdev =
            sqrt.(max.(0.0, plot_sample_covs[component_idx, component_idx, :]))

        upper = inv_transform(
            plot_sample_means[component_idx, :] + component_stdev;
            idx = component_idx,
        )
        mid = inv_transform(plot_sample_means[component_idx, :]; idx = component_idx)
        lower =
            max.(
                0.0,
                inv_transform(
                    plot_sample_means[component_idx, :] - component_stdev;
                    idx = component_idx,
                ),
            )

        plot!(
            state_plts[component_idx],
            plot_times,
            mid,
            ribbon = (mid - lower, upper - mid),
            fillalpha = 0.35,
            linecolor = :red,
            fillcolor = :grey,
            label = "",
        )
        if component_idx == 2
            ylims!(
                state_plts[component_idx],
                0,
                min(ylims(state_plts[component_idx])[2], 0.2),
            )
        elseif component_idx == 4
            ylims!(
                state_plts[component_idx],
                0,
                min(ylims(state_plts[component_idx])[2], 0.02),
            )
        end

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
    plt = plot(state_plts..., layout = (2, 2), size = (700, 500))
    Plots.savefig(plt, "personalization-virt-s$interval_idx-state.pdf")
    println("state plot $interval_idx saved")

    for component_idx = state_space_dim+1:unified_state_space_dimension
        component_stdev =
            sqrt.(max.(0.0, plot_sample_covs[component_idx, component_idx, :]))
        upper = inv_transform(
            plot_sample_means[component_idx, :] + component_stdev;
            idx = component_idx,
        )
        mid = inv_transform(plot_sample_means[component_idx, :]; idx = component_idx)
        lower =
            max.(
                0.0,
                inv_transform(
                    plot_sample_means[component_idx, :] - component_stdev;
                    idx = component_idx,
                ),
            )

        plot!(
            param_plts[component_idx-state_space_dim],
            plot_times,
            plot_sample_means[component_idx, :],
            ribbon = (mid - lower, upper - mid),
            fillalpha = 0.35,
            linecolor = :red,
            fillcolor = :grey,
            label = "",
        )
    end
    plt = plot(param_plts..., layout = (1, 1), size = (300, 250))
    Plots.savefig(plt, "personalization-virt-s$interval_idx-param.pdf")
    println("param plot $interval_idx saved")

    ####################
    # plotting the Kalman update

    if interval_idx > 1
        projection_update_plts = []
        ymaxes = []

        # plot the virtual patient trajectory
        for idx = 1:state_space_dim
            local subplt
            subplt = plot(virtual_patient_trajectory, idxs = idx, lc = :black, label = "")

            title!(subplt, var_meaning[idx])
            xlabel!(subplt, L"t")
            ylabel!(subplt, "")
            ylims!(subplt, 0, ylims(subplt)[2])

            push!(projection_update_plts, subplt)
            push!(ymaxes, ylims(subplt)[2])
        end

        Σ_record = Σ_record_unscaled / (num_samples - 1)

        # plot the previous
        for component_idx = 1:state_space_dim

            # plot previous projection
            component_stdev =
                sqrt.(max.(0.0, Σ_record[component_idx, component_idx, :, interval_idx-1]))

            upper = inv_transform(
                mean_record[component_idx, :, interval_idx-1] + component_stdev;
                idx = component_idx,
            )
            mid = inv_transform(
                mean_record[component_idx, :, interval_idx-1];
                idx = component_idx,
            )
            lower =
                max.(
                    0.0,
                    inv_transform(
                        mean_record[component_idx, :, interval_idx-1] - component_stdev;
                        idx = component_idx,
                    ),
                )


            plot!(
                projection_update_plts[component_idx],
                prediction_ts,
                mid,
                ribbon = (mid - lower, upper - mid),
                fillalpha = 0.35,
                linecolor = :red,
                fillcolor = :grey,
                label = "",
            )

            if max(mean_record[component_idx, :, interval_idx-1]...) > ymaxes[component_idx]
                ymaxes[component_idx] =
                    1.1 * max(mean_record[component_idx, :, interval_idx-1]...)
                if component_idx == 2
                    ymaxes[component_idx] = min(0.2, ymaxes[component_idx])
                elseif component_idx == 4
                    ymaxes[component_idx] = min(0.02, ymaxes[component_idx])
                end
            end
            ylims!(projection_update_plts[component_idx], 0, ymaxes[component_idx])

            # plot measurement
            if component_idx == sample_idx
                plot!(
                    projection_update_plts[component_idx],
                    [plot_times[begin]],
                    [virtual_patient_trajectory(plot_times[begin])[sample_idx]],
                    seriestype = :scatter,
                    mc = :red,
                    label = "",
                )
            end

            # plot new projection
            component_stdev =
                sqrt.(max.(0.0, Σ_record[component_idx, component_idx, :, interval_idx]))

            upper = inv_transform(
                mean_record[component_idx, :, interval_idx] + component_stdev;
                idx = component_idx,
            )
            mid = inv_transform(
                mean_record[component_idx, :, interval_idx];
                idx = component_idx,
            )
            lower =
                max.(
                    0.0,
                    inv_transform(
                        mean_record[component_idx, :, interval_idx] - component_stdev;
                        idx = component_idx,
                    ),
                )

            component_stdev =
                sqrt.(max.(0.0, Σ_record[component_idx, component_idx, :, interval_idx]))
            plot!(
                projection_update_plts[component_idx],
                prediction_ts,
                mid,
                ribbon = (mid - lower, upper - mid),
                fillalpha = 0.35,
                linecolor = :blue,
                fillcolor = :blue,
                label = "",
            )

            if max(mean_record[component_idx, :, interval_idx]...) > ymaxes[component_idx]
                ymaxes[component_idx] =
                    1.1 * max(mean_record[component_idx, :, interval_idx]...)
                if component_idx == 2
                    ymaxes[component_idx] = min(0.2, ymaxes[component_idx])
                elseif component_idx == 4
                    ymaxes[component_idx] = min(0.02, ymaxes[component_idx])
                end
            end
            ylims!(projection_update_plts[component_idx], 0, ymaxes[component_idx])

        end
        plt = plot(projection_update_plts..., layout = (2, 2), size = (700, 500))
        Plots.savefig(plt, filename_prefix * "s$interval_idx-update.pdf")

    end # if interval_idx > 1

    ################################################################################
    # record the prediciton

    h5open(filename_prefix * "data.hdf5", "r+") do fid
        g = create_group(fid, "prediction-$interval_idx")

        dset = create_dataset(g, "begin_time", Float64, 1)
        write(dset, begin_time)

        dset = create_dataset(g, "prediction_ts", Float64, size(prediction_ts))
        write(dset, prediction_ts)

        dset =
            create_dataset(g, "mean_record", Float64, size(mean_record[:, :, interval_idx]))
        write(dset, mean_record[:, :, interval_idx])

        dset = create_dataset(
            g,
            "sigma_record",
            Float64,
            size(Σ_record_unscaled[:, :, :, interval_idx]),
        )
        write(dset, Σ_record_unscaled[:, :, :, interval_idx] / (num_samples - 1))

    end

    ################################################################################
    # compute surprisals and record

    h5open(filename_prefix * "data.hdf5", "r+") do fid

        δ =
            hcat(virtual_patient_trajectory(prediction_ts).u...) -
            mean_record[:, :, interval_idx]
        Σs = Σ_record_unscaled[:, :, :, interval_idx] / (num_samples - 1)
        surprisal_series = zeros(size(mean_record)[2])
        for idx = 1:size(means)[2]
            surprisal_series[idx] =
                δ[:, idx]' * pinv(Σs[:, :, idx]) * δ[:, idx] / 2.0 + # TODO: replace pinv
                logdet(Σs[:, :, idx]) / 2.0 +
                unified_state_space_dimension * log(2 * pi) / 2.0
        end

        g = open_group(fid, "prediction-$interval_idx")

        dset = create_dataset(g, "prediction_surprisal", Float64, size(surprisal_series))
        write(dset, surprisal_series)

    end

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
    K = history_sample_covs[:, :, end] * H' * pinv(S)
    v =
        virtual_patient_trajectory(time_intervals[interval_idx+1])[sample_idx] -
        H * history_sample_means[:, end]
    smoothed_means[:, end] = history_sample_means[:, end] + K * v
    smoothed_covs[:, :, end] = Symmetric(history_sample_covs[:, :, end] - K * S * K')
    smoothed_covs[:, :, end] = pos_def_projection(smoothed_covs[:, :, end])

    # record the kalman matrices
    h5open(filename_prefix * "data.hdf5", "r+") do fid
        g = open_group(fid, "prediction-$interval_idx")

        dset = create_dataset(g, "H", Float64, size(H))
        write(dset, Matrix(H))

        dset = create_dataset(g, "R", Float64, size(R))
        write(dset, R)

        dset = create_dataset(g, "Q", Float64, size(Q))
        write(dset, Q)

        dset = create_dataset(g, "S", Float64, size(S))
        write(dset, S)

        dset = create_dataset(g, "K", Float64, size(K))
        write(dset, K)

        dset = create_dataset(g, "v", Float64, size(v))
        write(dset, v)

    end

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
                pinv(A * history_sample_covs[:, :, hist_idx] * A' + Q)

            smoothed_means[:, hist_idx] =
                history_sample_means[:, hist_idx] +
                G * (smoothed_means[:, hist_idx+1] - A * history_sample_means[:, hist_idx])


            smoothed_covs[:, :, hist_idx] =
                history_sample_covs[:, :, hist_idx] +
                G *
                (
                    smoothed_covs[:, :, hist_idx+1] -
                    A * history_sample_covs[:, :, hist_idx] * A' - Q
                ) *
                G'

            smoothed_covs[:, :, hist_idx] =
                pos_def_projection(smoothed_covs[:, :, hist_idx])

        end
    end

    # numerical cleanup b/c julia is persnickety
    for hist_idx = 1:history_size
        smoothed_covs[:, :, hist_idx] = pos_def_projection(smoothed_covs[:, :, hist_idx])
    end

    plot_history = false
    if plot_history
        history_plots = []
        for idx = 1:state_space_dim
            component_stdev = sqrt.(max.(0.0, smoothed_covs[idx, idx, :]))
            subplt = plot(
                history_times,
                smoothed_means[idx, :],
                ribbon = (min.(component_stdev, smoothed_means[idx, :]), component_stdev),
                fillalpha = 0.35,
                fc = :red,
                lc = :red,
                label = "",
            )
            push!(history_plots, subplt)

        end
        local plt
        plt = plot(history_plots..., layout = (2, 2), size = (700, 500))
        Plots.savefig(plt, filename_prefix * "virt-s$interval_idx-state-hist.pdf")
        # println("state plot $interval_idx saved")
    end

    ################################################################################
    # compute surprisals and record

    h5open(filename_prefix * "data.hdf5", "r+") do fid

        δ = hcat(virtual_patient_trajectory(prediction_ts).u...) - smoothed_means
        surprisal_series = zeros(size(mean_record)[2])
        for idx = 1:size(means)[2]
            surprisal_series[idx] =
                δ[:, idx]' * pinv(smoothed_covs[:, :, idx]) * δ[:, idx] / 2.0 + # TODO: replace pinv
                logdet(smoothed_covs[:, :, idx]) / 2.0 +
                unified_state_space_dimension * log(2 * pi) / 2.0
        end

        g = open_group(fid, "prediction-$interval_idx")

        dset = create_dataset(g, "updated_surprisal", Float64, size(surprisal_series))
        write(dset, surprisal_series)

    end

    ################################################################################
    # copy for next round
    means .= smoothed_means
    prior_Σs .= smoothed_covs

    # record the updated distributions
    h5open(filename_prefix * "data.hdf5", "r+") do fid
        g = open_group(fid, "prediction-$interval_idx")

        dset = create_dataset(g, "posterior_means", Float64, size(means))
        write(dset, means)

        dset = create_dataset(g, "posterior_covs", Float64, size(prior_Σs))
        write(dset, prior_Σs)

    end

    ################################################################################

end
