
var_meaning = [  # py/matlab index (julia index=matlab index)
    "virus",  # 0/1
    "susceptible lung cells",  # 1/2
    "infected lung cells",  # 2/3
    "lung cells resistant to virus",  # 3/4
    "dead lung cells",  # 4/5
    "Alveolar macrophages",  # 5/6
    "inflammatory macrophages",  # 6/7
    "monocytes",  # 7/8
    "neutrophils",  # 8/9
    "CD8+ T cells",  # 9/10
    "IL6 (U)",  # 10/11
    "IL6 (B)",  # 11/12
    "GM-CSF (U)",  # 12/13
    "GM-CSF (B)",  # 13/14
    "G-CSF (U)",  # 14/15
    "G-CSF (B)",  # 15/16
    "type 1 IFN (U)",  # 16/17
    "type 1 IFN (B)",  # 17/18
    "beta", # 18/19
    "p_MPhi_I_L", #19/20
    "p_L_MPhi", # 20/21
    "p_F_I", # 21/22
    "eta_F_I", # 22/23
    "eps_L_T", # 23/24
    "p_M_I", # 24/25
    "eta_F_MPhi", # 25/26
    "eps_F_I", # 26/27
    "p_F_M", # 27/28
    "tau_T", # 28/29
]

# param_names = ["tau_T"]

################################################################################

include("model_constants.jl")

# tau_T = 4.5 and this is more than twice that, which will have to be sufficient
const max_tau_T = 10.0

################################################################################

function covid_model(dy, y, history, p, t)
    V,
    S,
    I,
    R,
    D,
    MPhi_R,
    MPhi_I,
    M,
    N,
    T,
    L_U,
    L_B,
    G_U,
    G_B,
    C_U,
    C_B,
    F_U,
    F_B,
    beta,
    p_MPhi_I_L,
    p_L_MPhi,
    p_F_I,
    eta_F_I,
    eps_L_T,
    p_M_I,
    eta_F_MPhi,
    eps_F_I,
    p_F_M,
    tau_T = y

    tau_T = min(tau_T, max_tau_T)

    V_lag1 = history(p, t - tau_I; idxs = 1) # y_delay[..., 0, 0]
    S_lag1 = history(p, t - tau_I; idxs = 2) # y_delay[..., 1, 0]
    I_lag2 = history(p, t - tau_T; idxs = 3) # y_delay[..., 2, 1]

    C_BF = C_B / (A_C * N)

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

    dy[1] = phat * I - del_V_MPhi * MPhi_I * V - del_V_N * V * N - d_V_spec * V
    dy[2] =
        lam_S * (1 - (S + I + D + R) / Smax) * S - beta * S * V -
        rho * del_N / (1 + (IC_50_N / N)^h_N) * S
    dy[3] =
        beta / (1 + F_B / eps_F_I) * S_lag1 * V_lag1 - d_I * I -
        del_N / (1 + (IC_50_N / N) .^ h_N) * I - del_I_MPhi * MPhi_I * I - del_I_T * T * I
    dy[4] =
        lam_S * (1 - (S + I + D + R) / Smax) * R + beta * S * V / (1 + eps_F_I / F_B) -
        rho * del_N / (1 + (IC_50_N * 2 / N)^h_N) * R +
        lam_S * (1 - (S + I + D + R) / Smax) * R
    dy[5] =
        d_I * I +
        del_N * I / (1 + (IC_50_N / N)^h_N) +
        del_N * rho * S / (1 + (IC_50_N / N)^h_N) +
        del_I_MPhi * MPhi_I * I +
        del_I_T * T * I - d_D * D - del_D_MPhi * D * (MPhi_R + MPhi_I) +
        del_MPhi_D * D * (MPhi_I) +
        del_N * rho / (1 + (IC_50_N / N)^h_N) * R
    dy[6] =
        -a_I_MPhi * MPhi_R * (I + D) +
        (1 - MPhi_R / MPhi_max) * lam_MPhi * MPhi_I / (V + eps_V_MPhi)
    dy[7] =
        a_I_MPhi * MPhi_R * (I + D) +
        p_MPhi_I_G * G_B^h_M_MPhi / (G_B^h_M_MPhi + eps_G_MPhi^h_M_MPhi) * M +
        p_MPhi_I_L * L_B / (L_B + eps_L_MPhi) * M - d_MPhi_I * MPhi_I -
        del_MPhi_D * MPhi_I * D -
        (1 - MPhi_R / MPhi_max) * lam_MPhi * MPhi_I / (V + eps_V_MPhi)
    dy[8] =
        (M_prod_star + (psi_M_max - M_prod_star) * G_B^h_M / (G_B^h_M + eps_G_M^h_M)) * MR +
        p_M_I * I * M / (I + eps_I_M) -
        p_MPhi_I_G * G_B^h_M_MPhi * M / (G_B^h_M_MPhi + eps_G_MPhi^h_M_MPhi) -
        p_MPhi_I_L * L_B * M / (L_B + eps_L_MPhi) - d_M * M
    dy[9] =
        (
            N_prod_star +
            (psi_N_max - N_prod_star) * (C_BF - C_BF_star) / (C_BF - C_BF_star + eps_C_N)
        ) * NR + p_N_L * L_B / (L_B + eps_L_N) - d_N * N
    dy[10] =
        p_T_I * I_lag2 / (1 + L_B / eps_L_T) + p_T_F * F_B / (F_B + eps_F_T) * T - d_T * T
    dy[11] =
        p_L_I * I / (I + eta_L_I) +
        p_L_MPhi * MPhi_I / (MPhi_I + eta_L_MPhi) +
        p_L_M * M / (M + eta_L_M) - k_lin_L * L_U -
        k_B_L * ((N + T + M) * A_L - L_B) * L_U + k_U_L * L_B
    dy[12] = -k_int_L * L_B + k_B_L * ((N + T + M) * A_L - L_B) * L_U - k_U_L * L_B
    dy[13] =
        p_G_MPhi_I * MPhi_I / (MPhi_I + eta_G_MPhi) + p_G_M * M / (M + eta_G_M) -
        k_lin_G * G_U - k_B_G * (M * A_G - G_B) * G_U + k_U_G * G_B
    dy[14] = -k_int_G * G_B + k_B_G * (M * A_G - G_B) * G_U - k_U_G * G_B
    dy[15] =
        p_C_M * M / (M + eta_C_M) - k_lin_C * C_U - k_B_C * (N * A_C - C_B) * C_U^stoch_C +
        k_U_C * C_B
    dy[16] = -k_int_C * C_B + k_B_C * (N * A_C - C_B) * C_U^stoch_C - k_U_C * C_B
    dy[17] =
        p_F_I * I / (I + eta_F_I) +
        p_F_MPhi * MPhi_I / (MPhi_I + eta_F_MPhi) +
        p_F_M * M / (M + eta_F_M) - k_lin_F * F_U - k_B_F * ((T + I) * A_F - F_B) * F_U +
        k_U_F * F_B
    dy[18] = -k_int_F * F_B + k_B_F * ((T + I) * A_F - F_B) * F_U - k_U_F * F_B

    # dynamic parameters have no non-random dynamics
    dy[19:end] .= 0.0

    return
end

function covid_model_jacobian(y, history, p, t)
    V,
    S,
    I,
    R,
    D,
    MPhi_R,
    MPhi_I,
    M,
    N,
    T,
    L_U,
    L_B,
    G_U,
    G_B,
    C_U,
    C_B,
    F_U,
    F_B,
    beta,
    p_MPhi_I_L,
    p_L_MPhi,
    p_F_I,
    eta_F_I,
    eps_L_T,
    p_M_I,
    eta_F_MPhi,
    eps_F_I,
    p_F_M,
    tau_T = y

    tau_T = min(tau_T, max_tau_T)

    V_lag1 = history(p, t - tau_I; idxs = 1) # y_delay[..., 0, 0]
    S_lag1 = history(p, t - tau_I; idxs = 2) # y_delay[..., 1, 0]
    I_lag2 = history(p, t - tau_T; idxs = 3) # y_delay[..., 2, 1]

    # C_BF = C_B / (A_C * N)

    ddy = zeros(length(y), length(y))

    ddy[1, 1] = -MPhi_I * del_V_MPhi - N * del_V_N - d_V_spec
    ddy[1, 3] = phat
    ddy[1, 7] = -V * del_V_MPhi
    ddy[1, 9] = -V * del_V_N
    ddy[2, 1] = -S * beta
    ddy[2, 2] =
        -V * beta - lam_S * ((D + I + R + S) / Smax - 1) - S * lam_S / Smax -
        del_N * rho / ((IC_50_N / N)^h_N + 1)
    ddy[2, 3] = -S * lam_S / Smax
    ddy[2, 4] = -S * lam_S / Smax
    ddy[2, 5] = -S * lam_S / Smax
    ddy[2, 9] =
        -IC_50_N * S * del_N * h_N * rho * (IC_50_N / N)^(h_N - 1) /
        (N^2 * ((IC_50_N / N)^h_N + 1)^2)
    ddy[3, 3] = -MPhi_I * del_I_MPhi - T * del_I_T - d_I - del_N / ((IC_50_N / N)^h_N + 1)
    ddy[3, 7] = -I * del_I_MPhi
    ddy[3, 9] =
        -I * IC_50_N * del_N * h_N * (IC_50_N / N)^(h_N - 1) /
        (N^2 * ((IC_50_N / N)^h_N + 1)^2)
    ddy[3, 10] = -I * del_I_T
    ddy[3, 18] = -S_lag1 * V_lag1 * beta / (eps_F_I * (F_B / eps_F_I + 1)^2)
    ddy[4, 1] = S * beta / (eps_F_I / F_B + 1)
    ddy[4, 2] = -2 * R * lam_S / Smax + V * beta / (eps_F_I / F_B + 1)
    ddy[4, 3] = -2 * R * lam_S / Smax
    ddy[4, 4] =
        -2 * lam_S * ((D + I + R + S) / Smax - 1) - 2 * R * lam_S / Smax -
        del_N * rho / ((2 * IC_50_N / N)^h_N + 1)
    ddy[4, 5] = -2 * R * lam_S / Smax
    ddy[4, 9] =
        -2 * IC_50_N * R * del_N * h_N * rho * (2 * IC_50_N / N)^(h_N - 1) /
        (N^2 * ((2 * IC_50_N / N)^h_N + 1)^2)
    ddy[4, 18] = S * V * beta * eps_F_I / (F_B^2 * (eps_F_I / F_B + 1)^2)
    ddy[5, 2] = del_N * rho / ((IC_50_N / N)^h_N + 1)
    ddy[5, 3] = MPhi_I * del_I_MPhi + T * del_I_T + d_I + del_N / ((IC_50_N / N)^h_N + 1)
    ddy[5, 4] = del_N * rho / ((IC_50_N / N)^h_N + 1)
    ddy[5, 5] = -(MPhi_I + MPhi_R) * del_D_MPhi + MPhi_I * del_MPhi_D - d_D
    ddy[5, 6] = -D * del_D_MPhi
    ddy[5, 7] = -D * del_D_MPhi + I * del_I_MPhi + D * del_MPhi_D
    ddy[5, 9] =
        IC_50_N * R * del_N * h_N * rho * (IC_50_N / N)^(h_N - 1) /
        (N^2 * ((IC_50_N / N)^h_N + 1)^2) +
        IC_50_N * S * del_N * h_N * rho * (IC_50_N / N)^(h_N - 1) /
        (N^2 * ((IC_50_N / N)^h_N + 1)^2) +
        I * IC_50_N * del_N * h_N * (IC_50_N / N)^(h_N - 1) /
        (N^2 * ((IC_50_N / N)^h_N + 1)^2)
    ddy[5, 10] = I * del_I_T
    ddy[6, 1] = MPhi_I * lam_MPhi * (MPhi_R / MPhi_max - 1) / (V + eps_V_MPhi)^2
    ddy[6, 3] = -MPhi_R * a_I_MPhi
    ddy[6, 5] = -MPhi_R * a_I_MPhi
    ddy[6, 6] = -(D + I) * a_I_MPhi - MPhi_I * lam_MPhi / (MPhi_max * (V + eps_V_MPhi))
    ddy[6, 7] = -lam_MPhi * (MPhi_R / MPhi_max - 1) / (V + eps_V_MPhi)
    ddy[7, 1] = -MPhi_I * lam_MPhi * (MPhi_R / MPhi_max - 1) / (V + eps_V_MPhi)^2
    ddy[7, 3] = MPhi_R * a_I_MPhi
    ddy[7, 5] = MPhi_R * a_I_MPhi - MPhi_I * del_MPhi_D
    ddy[7, 6] = (D + I) * a_I_MPhi + MPhi_I * lam_MPhi / (MPhi_max * (V + eps_V_MPhi))
    ddy[7, 7] =
        -D * del_MPhi_D - d_MPhi_I + lam_MPhi * (MPhi_R / MPhi_max - 1) / (V + eps_V_MPhi)
    ddy[7, 8] =
        G_B^h_M_MPhi * p_MPhi_I_G / (G_B^h_M_MPhi + eps_G_MPhi^h_M_MPhi) +
        L_B * p_MPhi_I_L / (L_B + eps_L_MPhi)
    ddy[7, 12] =
        M * p_MPhi_I_L / (L_B + eps_L_MPhi) - L_B * M * p_MPhi_I_L / (L_B + eps_L_MPhi)^2
    ddy[7, 14] =
        G_B^(h_M_MPhi - 1) * M * h_M_MPhi * p_MPhi_I_G /
        (G_B^h_M_MPhi + eps_G_MPhi^h_M_MPhi) -
        G_B^(h_M_MPhi - 1) * G_B^h_M_MPhi * M * h_M_MPhi * p_MPhi_I_G /
        (G_B^h_M_MPhi + eps_G_MPhi^h_M_MPhi)^2
    ddy[8, 3] = M * p_M_I / (I + eps_I_M) - I * M * p_M_I / (I + eps_I_M)^2
    ddy[8, 8] =
        -d_M - G_B^h_M_MPhi * p_MPhi_I_G / (G_B^h_M_MPhi + eps_G_MPhi^h_M_MPhi) -
        L_B * p_MPhi_I_L / (L_B + eps_L_MPhi) + I * p_M_I / (I + eps_I_M)
    ddy[8, 12] =
        -M * p_MPhi_I_L / (L_B + eps_L_MPhi) + L_B * M * p_MPhi_I_L / (L_B + eps_L_MPhi)^2
    ddy[8, 14] =
        -G_B^(h_M_MPhi - 1) * M * h_M_MPhi * p_MPhi_I_G /
        (G_B^h_M_MPhi + eps_G_MPhi^h_M_MPhi) +
        G_B^(h_M_MPhi - 1) * G_B^h_M_MPhi * M * h_M_MPhi * p_MPhi_I_G /
        (G_B^h_M_MPhi + eps_G_MPhi^h_M_MPhi)^2 -
        (
            G_B^(h_M - 1) * (M_prod_star - psi_M_max) * h_M / (G_B^h_M + eps_G_M^h_M) -
            G_B^(h_M - 1) * G_B^h_M * (M_prod_star - psi_M_max) * h_M /
            (G_B^h_M + eps_G_M^h_M)^2
        ) * MR
    ddy[9, 9] =
        -NR * (
            C_B * (N_prod_star - psi_N_max) /
            (A_C * (C_BF_star - eps_C_N - C_B / (A_C * N)) * N^2) -
            C_B * (C_BF_star - C_B / (A_C * N)) * (N_prod_star - psi_N_max) /
            (A_C * (C_BF_star - eps_C_N - C_B / (A_C * N))^2 * N^2)
        ) - d_N
    ddy[9, 12] = p_N_L / (L_B + eps_L_N) - L_B * p_N_L / (L_B + eps_L_N)^2
    ddy[9, 16] =
        NR * (
            (N_prod_star - psi_N_max) /
            (A_C * (C_BF_star - eps_C_N - C_B / (A_C * N)) * N) -
            (C_BF_star - C_B / (A_C * N)) * (N_prod_star - psi_N_max) /
            (A_C * (C_BF_star - eps_C_N - C_B / (A_C * N))^2 * N)
        )
    ddy[10, 10] = -d_T + F_B * p_T_F / (F_B + eps_F_T)
    ddy[10, 12] = -I_lag2 * p_T_I / (eps_L_T * (L_B / eps_L_T + 1)^2)
    ddy[10, 18] = T * p_T_F / (F_B + eps_F_T) - F_B * T * p_T_F / (F_B + eps_F_T)^2
    ddy[11, 3] = p_L_I / (I + eta_L_I) - I * p_L_I / (I + eta_L_I)^2
    ddy[11, 7] =
        p_L_MPhi / (MPhi_I + eta_L_MPhi) - MPhi_I * p_L_MPhi / (MPhi_I + eta_L_MPhi)^2
    ddy[11, 8] = -A_L * L_U * k_B_L + p_L_M / (M + eta_L_M) - M * p_L_M / (M + eta_L_M)^2
    ddy[11, 9] = -A_L * L_U * k_B_L
    ddy[11, 10] = -A_L * L_U * k_B_L
    ddy[11, 11] = -(A_L * (M + N + T) - L_B) * k_B_L - k_lin_L
    ddy[11, 12] = L_U * k_B_L + k_U_L
    ddy[12, 8] = A_L * L_U * k_B_L
    ddy[12, 9] = A_L * L_U * k_B_L
    ddy[12, 10] = A_L * L_U * k_B_L
    ddy[12, 11] = (A_L * (M + N + T) - L_B) * k_B_L
    ddy[12, 12] = -L_U * k_B_L - k_U_L - k_int_L
    ddy[13, 7] =
        p_G_MPhi_I / (MPhi_I + eta_G_MPhi) - MPhi_I * p_G_MPhi_I / (MPhi_I + eta_G_MPhi)^2
    ddy[13, 8] = -A_G * G_U * k_B_G + p_G_M / (M + eta_G_M) - M * p_G_M / (M + eta_G_M)^2
    ddy[13, 13] = -(A_G * M - G_B) * k_B_G - k_lin_G
    ddy[13, 14] = G_U * k_B_G + k_U_G
    ddy[14, 8] = A_G * G_U * k_B_G
    ddy[14, 13] = (A_G * M - G_B) * k_B_G
    ddy[14, 14] = -G_U * k_B_G - k_U_G - k_int_G
    ddy[15, 8] = p_C_M / (M + eta_C_M) - M * p_C_M / (M + eta_C_M)^2
    ddy[15, 9] = -A_C * C_U^stoch_C * k_B_C
    ddy[15, 15] = -(A_C * N - C_B) * C_U^(stoch_C - 1) * k_B_C * stoch_C - k_lin_C
    ddy[15, 16] = C_U^stoch_C * k_B_C + k_U_C
    ddy[16, 9] = A_C * C_U^stoch_C * k_B_C
    ddy[16, 15] = (A_C * N - C_B) * C_U^(stoch_C - 1) * k_B_C * stoch_C
    ddy[16, 16] = -C_U^stoch_C * k_B_C - k_U_C - k_int_C
    ddy[17, 3] = -A_F * F_U * k_B_F + p_F_I / (I + eta_F_I) - I * p_F_I / (I + eta_F_I)^2
    ddy[17, 7] =
        p_F_MPhi / (MPhi_I + eta_F_MPhi) - MPhi_I * p_F_MPhi / (MPhi_I + eta_F_MPhi)^2
    ddy[17, 8] = p_F_M / (M + eta_F_M) - M * p_F_M / (M + eta_F_M)^2
    ddy[17, 10] = -A_F * F_U * k_B_F
    ddy[17, 17] = -(A_F * (I + T) - F_B) * k_B_F - k_lin_F
    ddy[17, 18] = F_U * k_B_F + k_U_F
    ddy[18, 3] = A_F * F_U * k_B_F
    ddy[18, 10] = A_F * F_U * k_B_F
    ddy[18, 17] = (A_F * (I + T) - F_B) * k_B_F
    ddy[18, 18] = -F_U * k_B_F - k_U_F - k_int_F

    return ddy
end


function history_maker(initial_conditions)
    V_ic,
    S_ic,
    I_ic,
    R_ic,
    D_ic,
    MPhi_R_ic,
    MPhi_I_ic,
    M_ic,
    N_ic,
    T_ic,
    L_U_ic,
    L_B_ic,
    G_U_ic,
    G_B_ic,
    C_U_ic,
    C_B_ic,
    F_U_ic,
    F_B_ic,
    beta_ic,
    p_MPhi_I_L_ic,
    p_L_MPhi_ic,
    p_F_I_ic,
    eta_F_I_ic,
    eps_L_T_ic,
    p_M_I_ic,
    eta_F_MPhi_ic,
    eps_F_I_ic,
    p_F_M_ic,
    tau_T_ic = initial_conditions

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
                MPhi_R_ic
            elseif idxs == 7
                MPhi_I_ic
            elseif idxs == 8
                M_ic
            elseif idxs == 9
                N_ic
            elseif idxs == 10
                T_ic
            elseif idxs == 11
                L_U_ic
            elseif idxs == 12
                L_B_ic
            elseif idxs == 13
                G_U_ic
            elseif idxs == 14
                G_B_ic
            elseif idxs == 15
                C_U_ic
            elseif idxs == 16
                C_B_ic
            elseif idxs == 17
                F_U_ic
            elseif idxs == 18
                F_B_ic
            elseif idxs == 19
                beta_ic
            elseif idxs == 20
                p_MPhi_I_L_ic
            elseif idxs == 21
                p_L_MPhi_ic
            elseif idxs == 22
                p_F_I_ic
            elseif idxs == 23
                eta_F_I_ic
            elseif idxs == 24
                eps_L_T_ic
            elseif idxs == 25
                p_M_I_ic
            elseif idxs == 26
                eta_F_MPhi_ic
            elseif idxs == 27
                eps_F_I_ic
            elseif idxs == 28
                p_F_M_ic
            elseif idxs == 29
                tau_T_ic
            end
        else
            [
                V_ic,
                S_ic,
                I_ic,
                R_ic,
                D_ic,
                MPhi_R_ic,
                MPhi_I_ic,
                M_ic,
                N_ic,
                T_ic,
                L_U_ic,
                L_B_ic,
                G_U_ic,
                G_B_ic,
                C_U_ic,
                C_B_ic,
                F_U_ic,
                F_B_ic,
                beta_ic,
                p_MPhi_I_L_ic,
                p_L_MPhi_ic,
                p_F_I_ic,
                eta_F_I_ic,
                eps_L_T_ic,
                p_M_I_ic,
                eta_F_MPhi_ic,
                eps_F_I_ic,
                p_F_M_ic,
                tau_T_ic,
            ]
        end
    end
    return history
end

constant_lags = [tau_I, max_tau_T] # tau_T is actually set in the ODE function

# we aren't using dependent lags since this is for lags that vary over the course
# of the DDE, not merely for constants lags that vary over parameterizations of the 
# DDE
#
# lag(y, params, t) = params[9] # tau_T
# dependent_lags = [lag]


# parameter ics
beta_0 = beta
p_MPhi_I_L_0 = p_MPhi_I_L
p_L_MPhi_0 = p_L_MPhi
p_F_I_0 = p_F_I
eta_F_I_0 = eta_F_I
eps_L_T_0 = eps_L_T
p_M_I_0 = p_M_I
eta_F_MPhi_0 = eta_F_MPhi
eps_F_I_0 = eps_F_I
p_F_M_0 = p_F_M
tau_T_0 = tau_T

history = history_maker([
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
    tau_T_0,
])

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
        state_var_noise, # "Alveolar macrophages",  # 5/6
        state_var_noise, # "inflammatory macrophages",  # 6/7
        state_var_noise, # "monocytes",  # 7/8
        state_var_noise, # "neutrophils",  # 8/9
        state_var_noise, # "CD8+ T cells",  # 9/10
        state_var_noise, # "IL6 (U)",  # 10/11
        state_var_noise, # "IL6 (B)",  # 11/12
        state_var_noise, # "GM-CSF (U)",  # 12/13
        state_var_noise, # "GM-CSF (B)",  # 13/14
        state_var_noise, # "G-CSF (U)",  # 14/15
        state_var_noise, # "G-CSF (B)",  # 15/16
        state_var_noise, # "type 1 IFN (U)",  # 16/17
        state_var_noise, # "type 1 IFN (B)",  # 17/18
        param_noise, # "beta", # 18/19
        param_noise, # "p_MPhi_I_L", # 19/20
        param_noise, # "p_L_MPhi", # 20/21
        param_noise, # "p_F_I", # 21/22
        param_noise, # "eta_F_I", # 22/23
        param_noise, # "eps_L_T", # 23/24
        param_noise, # "p_M_I", # 24/25
        param_noise, # "eta_F_MPhi", # 25/26
        param_noise, # "eps_F_I", # 26/27
        param_noise, # "p_F_M", # 27/28
        param_noise, # "tau_T", # 28/29
    ]
    return noise_vec
end

# function log_noise_matrix(sird_noise, state_var_noise, param_noise)
#     return diagm(noise_vec(sird_noise, state_var_noise, param_noise))
# end

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
