# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 09:38:44 2023

@author: trist
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.getcwd()) + "\shuffleproduct")
import shuffle as shfl
import responses as rsps
from generating_series import GeneratingSeries

x0 = 0
x1 = 1

# Keiths Fave values
m = 1
c = 20
k1 = 1e4
k2 = 5e7
x_init = 0
v_init = 0
amplitude = 1  # No greater than 10.


a1, a2 = shfl.sdof_roots(m, c, k1)


g0 = GeneratingSeries(np.array([
    [ 1, x0, x1],
    [a1, a2,  0]]
))


if x_init:
    g0.append(GeneratingSeries(np.array([
        [x_init, x0],
        [    a2,  0]
    ])))

if v_init-x_init*c:
    g0.append(GeneratingSeries(np.array([
        [v_init-x_init*c, x0],
        [             a1,  0]
    ])))

multiplier = np.array([
    [-k2, x0, x0],
    [ a1, a2,  0]
])

scheme = shfl.impulse_from_iter(g0, multiplier, n_shuffles=2, iter_depth=2)
imp_response = rsps.impulse(scheme, amplitude)
imp_response_partfrac = rsps.matlab_partfrac(imp_response)
time_domain = rsps.inverse_lb(imp_response_partfrac)
t_func = rsps.time_function(time_domain)

t = np.linspace(0, 1, 1000)
y = t_func(t)
fig = plt.figure()
ax = fig.gca()
ax.plot(t, y)
