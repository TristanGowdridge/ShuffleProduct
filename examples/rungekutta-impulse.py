# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:46:29 2025

@author: trist
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from params import m, c, k1, k2, k3, A, t_span, dt, t, init_cond
from vci_quad_cube import y1, y2, y3, y3_k2, y3_k3


y_ci = y1+y2+y3


def duffing_equation(t, y, m, c, k1, k2, k3, A):
    """
    Duffing's equation:
        m*y'' + c*y' + k1*y + k2*y^2 + k3*y^3 = amp*dirac_del

    """
    x, vx = y[0], y[1]

    dx_dt  = vx
    
    if t <= dt:
        imp = 1/dt
    else:
        imp = 0
    
    dvx_dt = (A*imp - c*vx - k1*x - k2*x**2 - k3*x**3) / m

    return np.array([dx_dt, dvx_dt])


# Solve Duffing's equation
sol = solve_ivp(
    duffing_equation, t_span, init_cond, method='RK45', t_eval=t,
    args=(m, c, k1, k2, k3, A)
)

# Extract displacement and velocity
displacement = sol.y[0]




gs_responses = []
for i in range(1, 4):
    gs_responses.append(np.load(f"quad_cube_y{i}_gen_num.npy"))

gs_response = sum(gs_responses)

# Plot the impulse response
plt.figure(figsize=(10, 6))


# Plot displacement
fig = plt.figure()
plt.plot(sol.t, displacement, label="Displacement (x(t))")
plt.plot(sol.t, gs_response, label="gs response", linestyle="--", color="k")
plt.plot(sol.t, y_ci, label="ci response", linestyle="-.", color="r")

plt.xlim(0, 0.4)
plt.legend()
plt.xlabel('Time (t)')
plt.ylabel('Displacement')
plt.title('Impulse Response of the Duffing Oscillator')
plt.grid(True)
plt.show()


fig, axs = plt.subplots(3, 1)
ci_responses = [y1, y2, y3]
plt.suptitle("Differences")
for i in range(3):
    axs[i].plot(ci_responses[i]-gs_responses[i])
   

fig1, axs1 = plt.subplots(2, 1)
gs_y3k2 = np.load("quad_cube_y3_k2_only_gen_num.npy")
gs_y3k3 = np.load("quad_cube_y3_k3_only_gen_num.npy")
plt.suptitle("Differences individual k2 and k3")

axs1[0].plot(gs_y3k2-y3_k2)
axs1[1].plot(gs_y3k3-y3_k3)

