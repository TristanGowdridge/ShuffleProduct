# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 08:29:35 2023

@author: trist
"""
import dill
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


fig_error = plt.figure()
ax_error = fig_error.gca()
for i, root_rel_err in enumerate(rrses, 1):
    ax_error.semilogx(A_range, root_rel_err, label=f"up to $y^g_{i}$")
ax_error.legend()
ax_error.set_title(
    "Relative Root Squared Error with Runge Kutta", fontsize=FONTSIZE
)
ax_error.set_xlabel("Impulse Amplitude", fontsize=FONTSIZE)
ax_error.set_ylabel("Error", fontsize=FONTSIZE)


# =============================================================================
# Time domain plots
# =============================================================================
amp_index = 48
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

axs[0].plot(t, y_rks[:, amp_index], label="$y_{rk}$", c="k")
axs[1].plot(t, y_rks[:, amp_index], label="$y_{rk}$", c="k")

for i, (y_s, y_i) in enumerate(zip(ys_summed, all_funcs), 1):
    axs[0].plot(
        t, y_s[:, amp_index], label=f"up to $y_{i}^g$", linestyle="--",
        alpha=0.7
    )
    axs[1].plot(
        t, y_i[:, amp_index], label=f"$y_{i}^g$", linestyle="--", alpha=0.7
    )
for i in range(len(axs)):
    axs[i].legend(loc=1, prop={"size": 10})
    axs[i].set_xlabel("t")
    axs[i].set_ylabel("Amplitude")
    
axs[0].set_title(
    f"Volterra series at impulse ampliude {A_range[amp_index]:.2f}"
)
axs[1].set_title(
    f"Individual Volterra terms at impulse amplitude {A_range[amp_index]:.2f}"
)


        
