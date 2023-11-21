using LinearAlgebra
using Plots
using DifferentialEquations
using StochasticDelayDiffEq
using LaTeXStrings
using JLD2
# using Turing
# using StatsPlots
using Random
# using MCMCChains
# using NPZ

# Random.seed!(14);

################################################################################

include("model_full.jl")
include("data.jl")

################################################################################
# solve system

num_samples = 100_000

virus_noise = 0.1
prior_log_means = append!([log(V0) - virus_noise / 2], log_mu_vec) # V0 and params
prior_Σ = diagm(append!([virus_noise], log_sigma2_vec))

log_prior = MvNormal(prior_log_means, prior_Σ)

param_samples = zeros(num_samples, length(prior_log_means))

for idx = 1:num_samples
    if idx % 10 == 0
        println(idx)
    end
    accept = false
    while !accept

        param_samples[idx, :] .= exp.(rand(log_prior))

        V0_samp,
        beta_samp,
        p_MPhi_I_L_samp,
        p_L_MPhi_samp,
        p_F_I_samp,
        eta_F_I_samp,
        eps_L_T_samp,
        p_M_I_samp,
        eta_F_MPhi_samp,
        eps_F_I_samp,
        p_F_M_samp,
        tau_T_samp = param_samples[idx, :]

        # p = (tau_T, state_var_noise, param_noise)
        const_params = (tau_T_samp, 0.1, 0.0) # using zero parameter noise here to get at the core of the params

        initial_condition = [
            V0_samp,
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
            # dynamic parameters
            beta_0,
            p_MPhi_I_L_0,
            p_L_MPhi_0,
            p_F_I_0,
            eta_F_I_0,
            eps_L_T_0,
            p_M_I_0,
            eta_F_MPhi_0,
            eps_F_I_0,
            p_F_M_0,
        ]

        sdde_prob_ic = remake(
            sdde_prob;
            u0 = initial_condition,
            h = history_maker(initial_condition),
            p = const_params,
        )

        predicted = solve(sdde_prob_ic, sdde_alg, alg_hints = [:stiff], saveat = 0.1)

        virtual_population_loss = covid_minimizing_fun_severe(predicted)
        accept = rand() <= exp(-virtual_population_loss)
    end
end

log_param_samples = log.(param_samples)
posterior_log_mean = mean(log_param_samples, dims = 1)
posterior_Σ =
    (log_param_samples .- posterior_log_mean)' * (log_param_samples .- posterior_log_mean) /
    num_samples


JLD2.@save "dyn_param_virtual_pop.jld2" posterior_log_mean posterior_Σ
