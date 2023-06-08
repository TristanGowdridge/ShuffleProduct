# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 08:22:04 2023

@author: trist
"""

import numpy as np
from scipy.integrate import solve_ivp
from params import m, c, k1, k3, A, t_span, t, init_cond, dt, plot
import matplotlib.pyplot as plt


def duffing_equation(t, y, m, c, k1, k3, A):
    """
    Duffing's equation:
        m*y'' + c*y' + k1*y + k3*y^3 = amp*dirac_del

    """
    x, vx = y[0], y[1]

    dx_dt  = vx
    
    if t <= dt:
        imp = 1/dt
    else:
        imp = 0
    
    dvx_dt = (A*imp - c*vx - k1*x - k3*x**3) / m

    return np.array([dx_dt, dvx_dt])


# Solve Duffing's equation
sol = solve_ivp(
    duffing_equation, t_span, init_cond, method='RK45', t_eval=t,
    args=(m, c, k1, k3, A)
)

y = sol.y[0]


if __name__ == "__main__":
    plot(y, None, "Runge")
    plt.title(
        f"Runge Kutta of Cube with Dirac delta amplitude: {A}"
    )
