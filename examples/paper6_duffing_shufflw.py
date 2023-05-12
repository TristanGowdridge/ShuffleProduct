# -*- coding: utf-8 -*-
"""
Created on Wed May 10 09:25:15 2023

@author: trist
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.getcwd()) + "\shuffleproduct")

import numpy as np
from sympy import Symbol
import matplotlib.pyplot as plt

import shuffle as shfl
import responses as rsps

x0 = 0
x1 = 1

# Keith's Fave values
m = 1
c = 20
k1 = 1e4
k2 = 5e7
k3 = 1e9
amplitude = 1 # No greater than 10.

t = np.linspace(0, 2, 1000)

fig = plt.figure()
ax = fig.gca()
v_init = 0
x_init = 1

# =============================================================================
# Generating Series
# =============================================================================
a1, a2 = shfl.sdof_roots(m, c, k1)

g0 = [shfl.GeneratingSeries(np.array([
        [ 1, x0, x1],
        [a1, a2,  0]
    ]))
]

if x_init:
    g0.append(shfl.GeneratingSeries(np.array([
            [x_init, x0],
            [    a2,  0]
        ])))

if v_init-x_init*c:
    g0.append(shfl.GeneratingSeries(np.array([
        [v_init-x_init*c, x0],
        [             a1,  0]
    ])))

multiplier = np.array([
    [k3, x0, x0],
    [a1, a2,  0]
])

scheme = shfl.iterate_gs(g0, multiplier, n_shuffles=3, iter_depth=1)
imp_gs = rsps.impulse(scheme, amplitude)
imp_pf = rsps.matlab_partfrac(imp_gs)
time_domain = rsps.inverse_lb(imp_pf)
t_func = rsps.time_function(time_domain)

y_gs = t_func(t)
_c1 = "tab:blue"
ax.plot(t, y_gs, color=_c1, label="gs impulse")
ax.tick_params(axis='y', labelcolor=_c1)
ax.legend(loc="upper right")



# =============================================================================
# Toybox solution.
# =============================================================================
import toybox as tb

# Simulate
n_points = 1000
fs = 1/500
amplitude = 10
system = tb.symetric(dofs=1, m=m, c=c, k=k1)

def cubic_stiffness_single(_, t, y, ydot):
    return np.dot(y**3, np.array([k3]))

system.N = cubic_stiffness_single
system.excitation = [np.array([amplitude] + [0]*(n_points-1))]

data = system.simulate((n_points, fs))

ax2 = ax.twinx()
_c2 = "tab:red"
ax2.plot(data['t'], data["y1"], color=_c2, label="toybox")
ax2.tick_params(axis='y', labelcolor=_c2)
ax2.legend(loc="upper left")