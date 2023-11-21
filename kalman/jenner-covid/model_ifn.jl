
var_meaning = [  # py/matlab index (julia index=matlab index)
    "virus",  # 0/1
    "susceptible lung cells",  # 1/2
    "infected lung cells",  # 2/3
    "lung cells resistant to virus",  # 3/4
    "dead lung cells",  # 4/5
    "type 1 IFN (U)",  # 16/17 -> 6
    "type 1 IFN (B)",  # 17/18 -> 7
    "beta", # 18/19 -> 8
    "p_F_I", # 21/22 -> 9
    "eta_F_I", # 22/23 -> 10
    "eps_F_I", # 26/27 -> 11
]

# param_names = ["tau_T"]

################################################################################

include("model_constants.jl")

# tau_T = 4.5 and this is more than twice that, which will have to be sufficient
const max_tau_T = 10.0

################################################################################

function covid_model(dy, y, history, p, t)
    V, S, I, R, D, F_U, F_B, beta, p_F_I, eta_F_I, eps_F_I = y

    # tau_T = min(tau_T, max_tau_T)

    V_lag1 = history(p, t - tau_I; idxs = 1) # y_delay[..., 0, 0]
    S_lag1 = history(p, t - tau_I; idxs = 2) # y_delay[..., 1, 0]

    # if there are any negative state variables, have the ode push them back to positivity
    # this is perhaps a little artificial, but it solves the problem of solvers running
    # into -ϵ values which _shouldn't_ occur, but which get exponentiated to fractional powers;
    # which is a problem. Why 16? Because it is a nice round, moderately large, power of two.
    negative_state_vars = y .< 0.0
    if any(negative_state_vars)
        dy .= 0
        dy[negative_state_vars] .= -16 * y[negative_state_vars]
        return
    end

    dy[1] = phat * I - d_V_spec * V
    dy[2] = lam_S * (1 - (S + I + D + R) / Smax) * S - beta * S * V
    dy[3] = beta / (1 + F_B / eps_F_I) * S_lag1 * V_lag1 - d_I * I
    dy[4] =
        lam_S * (1 - (S + I + D + R) / Smax) * R +
        beta * S_lag1 * V_lag1 / (1 + eps_F_I / F_B)
    dy[5] = d_I * I - d_D * D
    phi_F_prod = 0.25 # Jenner pg 13 after eqn 16
    dy[6] =
        phi_F_prod + p_F_I * I / (I + eta_F_I) - k_lin_F * F_U -
        k_B_F * ((T_star + I) * A_F - F_B) * F_U + k_U_F * F_B
    dy[7] = -k_int_F * F_B + k_B_F * ((T_star + I) * A_F - F_B) * F_U - k_U_F * F_B

    # dynamic parameters have no non-random dynamics
    dy[8:end] .= 0.0

    return
end

function history_maker(initial_conditions)
    V_ic,
    S_ic,
    I_ic,
    R_ic,
    D_ic,
    F_U_ic,
    F_B_ic,
    beta_ic,
    p_F_I_ic,
    eta_F_I_ic,
    eps_F_I_ic = initial_conditions

    function history(p, t; idxs = nothing)
        if typeof(idxs) <: Number
            if idxs == 1
                V_ic
            elseif idxs == 2
                S_ic
            elseif idxs == 3
                I_ic
            elseif idxs == 4
                R_ic
            elseif idxs == 5
                D_ic
            elseif idxs == 6
                F_U_ic
            elseif idxs == 7
                F_B_ic
            elseif idxs == 8
                beta_ic
            elseif idxs == 9
                p_F_I_ic
            elseif idxs == 10
                eta_F_I_ic
            elseif idxs == 11
                eps_F_I_ic
            end
        else
            [
                V_ic,
                S_ic,
                I_ic,
                R_ic,
                D_ic,
                F_U_ic,
                F_B_ic,
                beta_ic,
                p_F_I_ic,
                eta_F_I_ic,
                eps_F_I_ic,
            ]
        end
    end
    return history
end

constant_lags = [tau_I]

# parameter ics
beta_0 = beta
p_F_I_0 = p_F_I
eta_F_I_0 = eta_F_I
eps_F_I_0 = eps_F_I

history =
    history_maker([V0, S0, I0, R0, D0, F_U_0, F_B_0, beta_0, p_F_I_0, eta_F_I_0, eps_F_I_0])

################################################################################

# p = (sird_noise, state_var_noise, param_noise)
p = (0.0, 0.1, 0.001)

################################################################################

y0 = history(p, 0.0)

function noise_vec(sird_noise, state_var_noise, param_noise)
    noise_vec = [
        state_var_noise, # "virus",  # 0/1
        sird_noise, # "susceptible lung cells",  # 1/2
        sird_noise, # "infected lung cells",  # 2/3
        sird_noise, # "lung cells resistant to virus",  # 3/4
        sird_noise, # "dead lung cells",  # 4/5
        state_var_noise, # "type 1 IFN (U)",  # 16/17 -> 6
        state_var_noise, # "type 1 IFN (B)",  # 17/18 -> 7
        param_noise, # "beta", # 18/19 -> 8
        param_noise, # 21/22 -> 9
        param_noise, # "eta_F_I", # 22/23 -> 10
        param_noise, # "eps_F_I", # 26/27 -> 11
    ]
    return noise_vec
end

function log_noise_matrix(sird_noise, state_var_noise, param_noise)
    return diagm(noise_vec(sird_noise, state_var_noise, param_noise))
end

function noise_matrix(sird_noise, state_var_noise, param_noise, state_vec)
    return diagm(noise_vec(sird_noise, state_var_noise, param_noise) .* state_vec)
end

function covid_model_noise(du, u, h, p, t)
    sird_noise, state_var_noise, param_noise = p
    du .= noise_vec(sird_noise, state_var_noise, param_noise) .* u
end

################################################################################

sdde_prob = SDDEProblem(
    covid_model,
    covid_model_noise,
    y0,
    history,
    (0.0, 10.0),
    p;
    constant_lags = constant_lags,
)
sdde_alg = ImplicitRKMil()

################################################################################

dde_prob =
    DDEProblem(covid_model, y0, history, (0.0, 10.0), p; constant_lags = constant_lags)
dde_alg = MethodOfSteps(TRBDF2())

################################################################################
