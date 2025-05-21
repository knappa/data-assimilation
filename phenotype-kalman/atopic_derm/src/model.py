import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def atopic_cont(t, state, R, K, G):

	# Default Parameters: Fixed
	kappa_P = 0.6
	alpha_I = 0.25
	kappa_B = 0.5
	gamma_R = 10
	delta_B = 0.1
	gamma_G = 1
	kappa_D = 4
	delta_D = 0.5
	P_neg = 40
	P_pos = 26.6
	D_pos = 85
	R_on = 16.7
	G_off = 0
	G_on = 1
	m_on = 0.45
	beta_on = 6.71

	# Default parameters: Subject to change
	P_env = 95
	gamma_B = 1
	delta_P = 1
	R_off = 0
	K_off = 0
	beta_1 = 1
	beta_2 = 1
	beta_3 = 1
	beta_4 = 1

	# Treatments
	A = 0
	E = 0
	C = 0

	# Continuous state variables

	P, B, D = state

	dP = P_env * (kappa_P / (1 + gamma_B * B)) - (((alpha_I * R) / (1 + beta_1 * C)) + delta_P) * P - A * P

	dB = ((kappa_B * (1 - B)) / ((1 + gamma_R * (R / (1 + beta_2 * C))) * (1 + gamma_G * (G / (1 + beta_3 * C))))) - delta_B * K + E

	dD = kappa_D * (R / (1 + beta_4 * C)) - delta_D * D

	return [dP, dB, dD]



# Values of time (t) at which the solution will be evaluated
t_init = 0
t_end = 10
t_steps = 101
t = np.linspace(t_init, t_end, t_steps)

# Initial conditions; placeholder values used
I_0 = [0, 0, 0, 0, 0, 0]

# WIP
# def solve_model

result = solve_ivp(atopic_derm, [0, 10],  I_0)