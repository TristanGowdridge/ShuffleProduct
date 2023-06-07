# -*- coding: utf-8 -*-
"""
Created on Tue May 30 12:01:55 2023

@author: trist
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.getcwd()) + r"\shuffleproduct")
import shuffle as shfl
import responses as rsps
from generating_series import GeneratingSeries


FORMAT = ".pdf"

x0 = 0
x1 = 1

# Keiths Fave values
m = 1
c = 20
k1 = 1e4
k3 = 1e9
amplitude = 1  # No greater than 10.

a1, a2 = shfl.sdof_roots(m, c, k1)

g0 = shfl.GeneratingSeries(np.array([
    [ 1, x0, x1],
    [a1, a2,  0]
]))

multiplier = [np.array([
    [-k3, x0, x0],
    [ a1, a2,  0]
])]

imp_response = rsps.impulse_from_iter(
    g0, multiplier, n_shuffles=3, iter_depth=2, amplitude=amplitude
)

imp_response_partfrac = rsps.matlab_partfrac(
    imp_response, precision=5
)

time_domain = rsps.inverse_lb(imp_response_partfrac)
t_func = rsps.time_function(time_domain)

t = np.linspace(0, 1, 1000)
y = t_func(t)
fig = plt.figure()
ax = fig.gca()
ax.plot(t, y)

# plt.savefig("duffing_impulse_3iter" + FORMAT, bbox_inches="tight", dpi=500,
#             transparent=True, pad_inches=0)