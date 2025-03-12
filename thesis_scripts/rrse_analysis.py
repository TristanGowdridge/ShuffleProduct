# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 08:29:35 2023

@author: trist
"""
import dill
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

from params import A_range, t, iter_depth, A_log
from params import m, c, k1, k2, k3, t_span, init_cond
from params import FONTSIZE
from scipy.integrate import solve_ivp
from rk_quad_cube import duffing_equation

y_rks = []

A_range = A_log

for A in A_range:
    # Solve Duffing's equation
    sol = solve_ivp(
        duffing_equation, t_span, init_cond, method='RK45', t_eval=t,
        args=(m, c, k1, k2, k3, A)
    )
    y_rks.append(sol.y[0])

y_rks = np.vstack(y_rks).T


def rrse(pred, true):
    """
    Relative root squared error along each column.
    """
    numer = np.square(np.subtract(        pred, true)).sum(0)
    denom = np.square(np.subtract(true.mean(0), true)).sum(0)
    
    return 100 * np.sqrt(np.divide(numer, denom))

    
volterra_gen = []
for i in range(1, iter_depth+2):
    with open(f"quad_cube_y{i}_lambdify_A_t_gen.txt", "rb") as f_read:
        volterra_gen.append(dill.load(f_read))
        

all_funcs = []
for func in volterra_gen:
    all_amps = []
    for amp in A_range:
        all_amps.append(func(amp, t).real)
    all_funcs.append(np.vstack(all_amps).T)
    
ys_summed = []
rrses = []
for i, times in enumerate(all_funcs):
    if i == 0:
        ys_summed.append(times)
    else:
        ys_summed.append(ys_summed[-1] + times)
    rrses.append(rrse(y_rks, ys_summed[-1]))
    

# =============================================================================
# Saving Data
# =============================================================================
with (
        open("all_volterra_terms.pkl", "wb") as f_write1,
        open("all_rrses.pkl", "wb") as f_write2,
        open("all_volterra_series.pkl", "wb") as f_write3
):
    pkl.dump(all_funcs, f_write1)
    pkl.dump(rrses, f_write2)
    pkl.dump(ys_summed, f_write3)

y_rks = np.save("Runge_kuttas.npy", y_rks)
