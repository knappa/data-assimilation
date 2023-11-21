# initial conditions (for state variables)
const V0 = 4.5
const S0 = 0.16
const I0 = 0.0
const R0 = 0.0
const D0 = 0.0
const MPhi_R_0 = 23 * 1e-3 / 843
# MPhi_I_0 computed below
const M0 = 0.0004
const N0 = 0.00526
const T0 = 1.104 * 1e-4
const L_U_0 = 1.1
#L_B_0 computed below
const G_U_0 = 2.43
#G_B_0 computed below
const C_U_0 = 0.025
#C_B_0 computed below
const F_U_0 = 1.5 * 1e-2
#F_B_0 computed below

# other constants
const rho = 0.5
const d_V = 18.94
const lam_S = 0.7397
const Smax = S0
const h_N = 3.0198
const del_N = 1.6786
const tau_I = 0.1667
const del_I_MPhi = 121.195
const del_D_MPhi = 8.0256
const d_D = 8.0
const lam_MPhi = 5.9432 * 1e3
const MPhi_max = MPhi_R_0
const del_MPhi_D = 6.0607
const h_M_MPhi = 2.0347
const d_MPhi_I = 0.3
const eps_G_M = 57.197
const d_M = 0.7562
const h_M = 1.6711
const p_M_I = 2.2 * 1e-1
const MR = 0.1619 * 70 / 5000
const psi_M_max = 11.5451
const NR = 0.03156
const eps_C_N = 1.8924 * 1e-4
const d_N = 1.2773
const psi_N_max = 4.1335
const eps_L_N = eps_G_M
const p_T_L = 4.0
const p_T_F = 4.0
const d_T = 0.4
const eps_T_I = 1e-6
const tau_T = 4.5
const stoch = 1.0
const stoch_C = 1.4608
const avo = 6.02214e23
const k_lin_L = 16.636
const k_int_L = 61.8
const k_B_L = 0.0018
const k_U_L = 22.29
const R_L_N = 720.0
const R_L_T = 300.0
const R_L_M = 509.0
const MM_L = 21000.0
const eta_G_M = 0.14931
const k_lin_G = 11.7427
const k_int_G = 73.44
const k_B_G = 0.0021
const k_U_G = 522.72
const R_G_M = 1058.0
const MM_G = 14e3
const k_lin_C = 0.16139
const k_int_C = 462.42
const k_B_C = 2.243
const k_U_C = 184.87
const R_C_N = 600.0
const MM_C = 19.6e3
const k_lin_F = 16.635
const k_int_F = 16.968
const k_B_F = 0.0107
const k_U_F = 6.072
const R_F_T = 1000.0
const R_F_I = 1300.0
const MM_F = 19000.0
const A_L = stoch * MM_L / avo * (R_L_M + R_L_N + R_L_T) * (1 / 5000) * 10^9 * 1e12
const A_G = stoch * MM_G / avo * R_G_M * (1 / 5000) * 10^9 * 1e12
const A_C = 2 * MM_C / avo * R_C_N * (1 / 5000) * 10^9 * 1e9
const A_F = stoch * MM_F / avo * (R_F_T + R_F_I) * (1 / 5000) * 10^9 * 1e12
const eta_F_M = 5.4 * 1e-1
const eta_L_M = 0.4498 * 1e-2
const eps_F_I = 2 * 1e-4
const p_F_I = 2.8235 * 1e4 * 1e-4
const eta_F_I = 0.0011164 * 1e1
const p_F_M = 3.5600 * 1e4 * 1e-4
const p_L_M = 72560 / 1e2 * 0.05
const p_L_MPhi = 1872.0
const eps_L_MPhi = 1102.9 / 1e5
const eps_G_MPhi = 2664.5 / 1e5
const eta_L_I = 0.7
const p_L_I = 1188.7 / 1e2
const eps_V_MPhi = 905.22 / 1e3
const IC_50_N = 4.7054 * 1e-2
const p_F_MPhi = 1.3
const eps_F_T = 1e-3 * 1.5
const a_I_MPhi = 1100.0
const eps_I_M = 0.11
const p_MPhi_I_L = 0.42 * 4
const p_MPhi_I_G = 0.42 * 4
const phat = 394.0
const beta = 0.3
const d_I = 0.1
const d_V_spec = 0.0
const del_V_MPhi = 76800 / 200
const del_V_N = 2304 / 2.5
const eps_L_T = 1.5 * 1e-5
const p_T_I = 0.008 * 2
const del_I_T = 238 * 0.5
const d_MPhi_R = 0.0
const M_star = M0
const MPhi_R_star = MPhi_R_0
const N_star = N0
const T_star = T0
const F_U_star = F_U_0
const F_B_star = k_B_F * T_star * A_F * F_U_star / (k_int_F + k_B_F * F_U_star + k_U_F)
const F_B_0 = F_B_star
const G_U_star = G_U_0
const G_B_star = (k_B_G * M_star * A_G * G_U_star) / (k_int_G + k_U_G + k_B_G * G_U_star)
const G_B_0 = G_B_star
const C_U_star = C_U_0
const C_B_star =
    (k_B_C * C_U_star^stoch_C * A_C * N_star) / (k_int_C + k_U_C + C_U_star^stoch_C * k_B_C)
const C_B_0 = C_B_star
const C_BF_star = C_B_star / (A_C * N_star)
const L_U_star = L_U_0
const L_B_star = (
    k_B_L * (T_star + N_star + M_star) * A_L * L_U_star /
    (k_int_L + k_B_L * L_U_star + k_U_L)
)
const L_B_0 = L_B_star
const MPhi_I_star =
    (
        p_MPhi_I_G * G_B_star^h_M_MPhi * M_star /
        (G_B_star^h_M_MPhi + eps_G_MPhi^h_M_MPhi) +
        p_MPhi_I_L * L_B_star * M_star / (L_B_star + eps_L_MPhi)
    ) / ((1 - MPhi_R_star / MPhi_max) * lam_MPhi / eps_V_MPhi + d_MPhi_I)
const MPhi_I_0 = MPhi_I_star
const KK = (
    1 / p_L_MPhi * (
        -p_L_M * M_star / (M_star + eta_L_M) +
        k_lin_L * L_U_star +
        k_B_L * ((N_star + T_star + M_star) * A_L - L_B_star) * L_U_star -
        k_U_L * L_B_star
    )
)
const eta_L_MPhi = (MPhi_I_star - KK * MPhi_I_star) / KK
const eta_G_MPhi = eta_L_MPhi
const p_G_MPhi_I =
    -(
        -k_lin_G * G_U_star - k_B_G * (M_star * A_G - G_B_star) * G_U_star +
        k_U_G * G_B_star
    ) / (MPhi_I_star / (MPhi_I_star + eta_G_MPhi) + M_star / (M_star + eta_G_M))
const p_G_M = p_G_MPhi_I
const p_C_M = p_G_M / 100
const BB = (
    k_lin_C * C_U_star + k_B_C * (N_star * A_C - C_B_star) * C_U_star^stoch_C -
    k_U_C * C_B_star
)
const eta_C_M = (p_C_M * M_star - M_star * BB) / BB
const T_prod_star = d_T * T_star
const T_M_prod_star = d_T * T_star - p_T_F * F_B_star * T_star / (F_B_star + eps_F_T)
const DD = (
    1 / MR * (
        p_MPhi_I_G * G_B_star^h_M_MPhi * M_star /
        (G_B_star^h_M_MPhi + eps_G_MPhi^h_M_MPhi) +
        p_MPhi_I_L * L_B_star * M_star / (L_B_star + eps_L_MPhi) +
        d_M * M_star
    )
)
const EE = G_B_star^h_M / (G_B_star^h_M + eps_G_M^h_M)
const M_prod_star = (DD - psi_M_max * EE) / (1 - EE)
const N_prod_star = d_N * N_star / (NR + L_B_star / (L_B_star + eps_L_N))
const p_N_L = N_prod_star
const CC = (
    p_F_M * M_star / (M_star + eta_F_M) - k_lin_F * F_U_star -
    k_B_F * (T_star * A_F - F_B_star) * F_U_star + k_U_F * F_B_star
)
const eta_F_MPhi = (p_F_MPhi * MPhi_I_star + CC * MPhi_I_star) / (-CC)

################################################################################

# number of patients
const N = 100
# calculating beta std
const std_day3_Munster = 1.1958
const mean_day3_Munster = 5.9966
const relative_std_day3_Munster = std_day3_Munster / mean_day3_Munster
const sigma_beta = relative_std_day3_Munster
# calculating eps_F_I std
const sigma_eps_F_I = 208.9 / (3125000)  # standard deviation sigma informed by confidence intervals for data fit
# calculating p_F_I,eta_F_MPhi and eta_F_I std
const CI_day0_TrouilletAssant = [2 * 1e2, 3 * 1e3] / 1000  # pg/ml
const mean_day0_TrouilletAssant = 3.5 * 1e2 / 1000
const number_patients_TrouilletAssant = 26
const t_score_26df_95CI = 2.060
const std_TrouilletAssant = (
    (CI_day0_TrouilletAssant[2] - mean_day0_TrouilletAssant) / t_score_26df_95CI *
    sqrt(number_patients_TrouilletAssant)
)
const relative_std_day0_TrouilletAssant = std_TrouilletAssant / mean_day0_TrouilletAssant
const sigma_p_F_I = relative_std_day0_TrouilletAssant / 10
const sigma_eta_F_MPhi = relative_std_day0_TrouilletAssant / 1e6 * 5
const sigma_eta_F_I = relative_std_day0_TrouilletAssant / 1e3
# calculating p_L_MPhi and p_MPhi_L std
const mean_mild_Herold = 21
const std_mild_Herold = 19
const mean_severe_Herold = 195
const std_severe_Herold = 165
const mean_moderate_Lucas = 2.449560839602535e02
const std_moderate_Lucas = 5.385157964298745e02
const relative_std_moderate_Lucas = std_moderate_Lucas / mean_moderate_Lucas
const mean_severe_Lucas = 1.004274975130175e03
const std_severe_Lucas = 1.646578521236555e03
const relative_std_severe_Lucas = std_severe_Lucas / mean_severe_Lucas
const relative_std_mild_Herold = std_mild_Herold / mean_mild_Herold
const relative_std_severe_Herold = std_severe_Herold / mean_severe_Herold
const sigma_p_L_MPhi_mild = relative_std_moderate_Lucas  # relative_std_mild_Herold
const sigma_p_MPhi_I_L_mild = (
    relative_std_moderate_Lucas  # relative_std_mild_Herold*parameters.p_MPhi_I_L/10
)
const sigma_p_L_MPhi_severe = relative_std_severe_Lucas  # relative_std_severe_Herold
const sigma_p_MPhi_I_L_severe = (
    relative_std_severe_Lucas  # relative_std_severe_Herold*parameters.p_MPhi_I_L/10
)
# calculating eps_L_T and tau_T std
const mean_Liu_Tcells_day10 = 0.35
const std_Liu_Tcells_day10 = 0.1
const relative_std_Liu_Tcells_day10 = std_Liu_Tcells_day10 / mean_Liu_Tcells_day10
const sigma_eps_L_T = relative_std_Liu_Tcells_day10 * eps_L_T
const sigma_tau_T = relative_std_Liu_Tcells_day10
# calculating p_M_I std
const mean_Liu_monocytes_day4 = 1
const std_Liu_monocytes_day4 = 0.75
const relative_std_Liu_monocytes_day4 = std_Liu_monocytes_day4 / mean_Liu_monocytes_day4
const sigma_p_M_I = relative_std_Liu_monocytes_day4 * p_M_I / 2
# calculating p_F_M std
const sigma_p_F_M = (126.2 * 1e-4)  # standard deviation sigma informed by confidence intervals for data fit
const mu_vec = [
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
    tau_T,
]
const sigma_mild_vec = [
    sigma_beta,
    sigma_p_MPhi_I_L_mild,
    sigma_p_L_MPhi_mild,
    sigma_p_F_I,
    sigma_eta_F_I,
    sigma_eps_L_T,
    sigma_p_M_I,
    sigma_eta_F_MPhi,
    sigma_eps_F_I,
    sigma_p_F_M,
    sigma_tau_T,
]
const sigma_vec = [
    sigma_beta,
    sigma_p_MPhi_I_L_severe,
    sigma_p_L_MPhi_severe,
    sigma_p_F_I,
    sigma_eta_F_I,
    sigma_eps_L_T,
    sigma_p_M_I,
    sigma_eta_F_MPhi,
    sigma_eps_F_I,
    sigma_p_F_M,
    sigma_tau_T,
]

const cov_vec = sigma_vec .^ 2
const log_sigma2_vec = log1p.(cov_vec ./ (mu_vec .^ 2))
const log_mu_vec = log.(mu_vec) .- log_sigma2_vec / 2
