using NaNMath
using LinearAlgebra
using Plots
using DifferentialEquations
using StochasticDelayDiffEq
using LaTeXStrings
using JLD2
using Random
import BasicInterpolators
import SciMLBase
using HDF5
using ArgParse
using ProgressBars
import Logging

Logging.disable_logging(Logging.Error) # no logs of severity < error

################################################################################

function transform(v; idx = nothing)
    if typeof(idx) <: Number
        if idx == 1
            return v
        elseif idx == 2
            return log.(max.(1e-15, expm1.(v)))
        elseif idx == 3
            return v
        elseif idx == 4
            return v .* 100
        elseif idx == 5
            return v
        else
            return v
        end
    else
        w = copy(v)
        # w[2,:] = log.(max.(1e-300,v[2,:]))
        w[2, :] = log.(max.(1e-15, expm1.(v[2, :])))
        # w[3, :] = v[3, :] .* 100
        w[4, :] = v[4, :] .* 100
        # w[5, :] = log.(v[5, :])
        return w
    end
end

function inv_transform(v; idx = nothing)
    if typeof(idx) <: Number
        if idx == 1
            return v
        elseif idx == 2
            return log1p.(exp.(v))
        elseif idx == 3
            return v
        elseif idx == 4
            return v ./ 100
        elseif idx == 5
            return v
        else
            return v
        end
    else
        w = copy(v)
        # w[2,:] = exp.(v[2,:])
        w[2, :] = log1p.(exp.(v[2, :]))
        # w[3, :] = v[3, :] ./ 100
        w[4, :] = v[4, :] ./ 100
        # w[5, :] = exp.(v[5, :])
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
    "--graphs"
    help = "plot graphs"
    action = :store_true
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

const plot_graphs = parsed_args["graphs"]

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
history_means = zeros(unified_state_space_dimension, history_size)
history_means[1, end] = virtual_pop_prior_mean[1] # only last time point gets the virus
history_means[2:state_space_dim, :] .= initial_condition_state[2:end]
history_means[state_space_dim+1:end, :] .= virtual_pop_prior_mean[2:end]

history_means = transform(history_means)

# create the historical covariance matrices
history_prior_Σs = zeros(unified_state_space_dimension, unified_state_space_dimension, history_size)
history_prior_Σs[2:state_space_dim, 2:state_space_dim, :] .=
    diagm((0.1 * initial_condition_state[2:end]) .^ 2)
# the prior starts with V0, which messes with the block structure a little bit,
# so we have to put it in the right places
history_prior_Σs[1, 1, :] .= virtual_pop_prior_Σ[1, 1]
history_prior_Σs[1, state_space_dim+1:end, :] .= virtual_pop_prior_Σ[1, 2:end]
history_prior_Σs[state_space_dim+1:end, 1, :] .= virtual_pop_prior_Σ[2:end, 1]
history_prior_Σs[state_space_dim+1:end, state_space_dim+1:end, :] .=
    virtual_pop_prior_Σ[2:end, 2:end]

################################################################################

include("compute_jacobian.jl")
include("get_prediction.jl")

################################################################################

prior = [CustomNormal(history_means[:, idx], history_prior_Σs[:, :, idx]) for idx = 1:history_size]

virtual_patient_trajectory, virtual_patient_history =
    get_prediction(virt_patient_tspan[1], virt_patient_tspan[2], prior)

JLD2.save(
    filename_prefix * "virt-traj.jld2",
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

if plot_graphs
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
end

################################################################################

const time_intervals = Vector(virt_patient_tspan[1]:sample_dt:virt_patient_tspan[2])

const prediction_size = 1 + ceil(Int, (virt_patient_tspan[2] - virt_patient_tspan[1]) / dt)
prediction_ts = Vector(virt_patient_tspan[1]:dt:virt_patient_tspan[2])
mean =
    zeros(unified_state_space_dimension, prediction_size, length(time_intervals))
Σ_unscaled = zeros(
    unified_state_space_dimension,
    unified_state_space_dimension,
    prediction_size,
    length(time_intervals),
)


# record the distributions
h5open(filename_prefix * "data.hdf5", "r+") do fid
    g = create_group(fid, "prediction-all")

    dset = create_dataset(g, "mean", Float64, size(mean))
    write(dset, mean)

    dset = create_dataset(g, "cov", Float64, size(Σ_unscaled))
    write(dset, Σ_unscaled)

end

################################################################################

for interval_idx in ProgressBar(1:length(time_intervals)-1)

    # determine the current time interval to test
    begin_time = time_intervals[interval_idx]
    end_time = time_intervals[interval_idx+1]

    # build the historical means and covariance matrices into a series of distributions
    local prior
    prior = [CustomNormal(history_means[:, idx], history_prior_Σs[:, :, idx]) for idx = 1:history_size]

    # we need to compute mean/covariance for the predictions, the history (used in the next Kalman update),
    # and to plot.

    # historical means for the next round
    history_times = [end_time + dt * (idx - history_size) for idx = 1:history_size]
    history_sample_means = zeros(unified_state_space_dimension, history_size)
    history_sample_covs_unscaled =
        zeros(unified_state_space_dimension, unified_state_space_dimension, history_size)

    # plot from now to the end of the simulation
    num_plot_points = ceil(Int, (simulation_end_time - begin_time) / dt)
    plot_sample_means = zeros(unified_state_space_dimension, num_plot_points)
    plot_sample_covs_unscaled =
        zeros(unified_state_space_dimension, unified_state_space_dimension, num_plot_points)
    plot_times = LinRange(begin_time, end_time, num_plot_points)

    # mean and covariance 
    mean[:, 1+(interval_idx-1)*ceil(Int, sample_dt / dt):end, interval_idx:end] .= 0.0
    Σ_unscaled[:, :, 1+(interval_idx-1)*ceil(Int, sample_dt / dt):end, interval_idx:end] .= 0.0

    for sample_idx in ProgressBar(1:num_samples)

        # predict up to the end time of the total simulation
        prediction, history_samp = get_prediction(begin_time, simulation_end_time, prior)

        # update means/covs for the history using Welford's online algorithm for mean and 
        # covariance calculation. See Knuth Vol 2, pg 232
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

        # update means/covs for the plots using Welford's online algorithm for mean and
        # covariance calculation. See Knuth Vol 2, pg 232
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

        # update means/covs for the predictions using Welford's online algorithm for mean and
        # covariance calculation. See Knuth Vol 2, pg 232
        for (offset_time_idx, t) in
            enumerate(prediction_ts[1+(interval_idx-1)*ceil(Int, sample_dt / dt):end])
            time_idx = (interval_idx - 1) * ceil(Int, sample_dt / dt) + offset_time_idx
            if t <= begin_time
                sample = transform(history_samp(t))
            else
                sample = transform(prediction(t))
            end

            for future_interval_idx = interval_idx:length(time_intervals)
                old_mean = copy(mean[:, time_idx, future_interval_idx])
                mean[:, time_idx, future_interval_idx] +=
                    (sample - mean[:, time_idx, future_interval_idx]) / sample_idx
                # use variant formula (mean of two of the standard updates) to 
                # increase symmetry in the fp error (1e-18) range
                Σ_unscaled[:, :, time_idx, future_interval_idx] +=
                    (
                        (sample - mean[:, time_idx, future_interval_idx]) *
                        (sample - old_mean)' +
                        (sample - old_mean) *
                        (sample - mean[:, time_idx, future_interval_idx])'
                    ) / 2.0
            end
        end

    end


    # record the distributions
    h5open(filename_prefix * "data.hdf5", "r+") do fid
        g = open_group(fid, "prediction-all")

        dset = open_dataset(g, "mean")
        write(dset, mean)

        dset = open_dataset(g, "cov")
        write(dset, Σ_unscaled / (num_samples-1))

    end


    ################################################################################
    # plots

    plot_sample_covs = plot_sample_covs_unscaled / (num_samples - 1)

    if plot_graphs

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
        Plots.savefig(plt, filename_prefix * "s$interval_idx-state.pdf")
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
                mid,
                ribbon = (mid - lower, upper - mid),
                fillalpha = 0.35,
                linecolor = :red,
                fillcolor = :grey,
                label = "",
            )
        end
        plt = plot(param_plts..., layout = (1, 1), size = (300, 250))
        Plots.savefig(plt, filename_prefix * "s$interval_idx-param.pdf")
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

            # plot the previous
            for component_idx = 1:state_space_dim

                # plot previous projection
                component_stdev =
                    sqrt.(max.(0.0, Σ_unscaled[component_idx, component_idx, :, interval_idx-1]/(num_samples-1)))

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
        end # if plot_graphs

    end # if interval_idx > 1

    ################################################################################
    # compute surprisals and record

    h5open(filename_prefix * "data.hdf5", "r+") do fid

        δ =
            hcat(virtual_patient_trajectory(prediction_ts).u...) -
            mean[:, :, interval_idx]
        Σs = Σ_unscaled[:, :, :, interval_idx] / (num_samples - 1)
        surprisal_series = zeros(size(mean)[2])
        mahalanobis_series = zeros(size(mean)[2])
        for idx = 1:size(mean)[2]
            mahalanobis_series[idx] = δ[:, idx]' * pinv(pos_def_projection(Σs[:, :, idx])) * δ[:, idx]
            surprisal_series[idx] =
                (mahalanobis_series[idx] +
                logdet(pos_def_projection(Σs[:, :, idx])) +
                unified_state_space_dimension * log(2 * pi)) / 2.0
        end

        g = create_group(fid, "prediction-$interval_idx")

        dset = create_dataset(g, "prediction_surprisal", Float64, size(surprisal_series))
        write(dset, surprisal_series)

        dset = create_dataset(g, "mahalanobis", Float64, size(mahalanobis_series))
        write(dset, mahalanobis_series)
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
        transform(virtual_patient_trajectory(time_intervals[interval_idx+1])[sample_idx]; idx=sample_idx) -
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

    ################################################################################
    # copy for next round
    history_means .= smoothed_means
    history_prior_Σs .= smoothed_covs

    # record the updated distributions
    h5open(filename_prefix * "data.hdf5", "r+") do fid
        g = open_group(fid, "prediction-$interval_idx")

        dset = create_dataset(g, "posterior_means", Float64, size(history_means))
        write(dset, history_means)

        dset = create_dataset(g, "posterior_covs", Float64, size(history_prior_Σs))
        write(dset, history_prior_Σs)

    end

    ################################################################################

end
