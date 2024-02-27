
var_meaning = [  # py/matlab index (julia index=matlab index)
    "virus",  # 0/1
    "susceptible lung cells",  # 1/2
    "infected lung cells",  # 2/3
    "dead lung cells",  # 3/4
    "beta", # 4/5
]

# param_names = ["tau_T"]

################################################################################

include("model_constants.jl")

# tau_T = 4.5 and this is more than twice that, which will have to be sufficient
const max_tau_T = 10.0

################################################################################

function covid_model(dy, y, history, p, t)
    V, S, I, D, beta, = y

    V_lag1 = history(p, t - tau_I; idxs = 1) # y_delay[..., 0, 0]
    S_lag1 = history(p, t - tau_I; idxs = 2) # y_delay[..., 1, 0]

    # C_BF = C_B / (A_C * N)

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
    dy[2] = lam_S * (1 - (S + I + D) / Smax) * S - beta * S * V
    dy[3] = beta * S_lag1 * V_lag1 - d_I * I
    dy[4] = d_I * I - d_D * D

    # dynamic parameters have no non-random dynamics
    dy[5:end] .= 0.0

    return
end

function history_maker(initial_conditions)
    V_ic, S_ic, I_ic, D_ic, beta_ic = initial_conditions

    function history(p, t; idxs = nothing)
        if typeof(idxs) <: Number
            if idxs == 1
                V_ic
            elseif idxs == 2
                S_ic
            elseif idxs == 3
                I_ic
            elseif idxs == 4
                D_ic
            elseif idxs == 5
                beta_ic
            end
        else
            [V_ic, S_ic, I_ic, D_ic, beta_ic]
        end
    end
    return history
end

constant_lags = [tau_I]

# we aren't using dependent lags since this is for lags that vary over the course
# of the DDE, not merely for constants lags that vary over parameterizations of the 
# DDE
#
# lag(y, params, t) = params[9] # tau_T
# dependent_lags = [lag]


# parameter ics
beta_0 = beta

history = history_maker([V0, S0, I0, D0, beta_0])

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
        sird_noise, # "dead lung cells",  # 3/4
        param_noise, # "beta", # 4/5
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
