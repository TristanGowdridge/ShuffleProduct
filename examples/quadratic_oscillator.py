# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 09:38:44 2023

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

m = 1
c = 20
k1 = 1e4
k2 = 5e7
k3 = 1e9

impulse_amplitude = 10 # No greater than 10.

r1, r2 = np.roots([m, c, k1])

a1 = 1/r1
a2 = 1/r2

g0 = shfl.GeneratingSeries(np.array([
    [ 1, x0, x1],
    [a1, a2,  0]
    ]))

multiplier = np.array([
    [-k2, x0, x0],
    [ a1, a2,  0]
    ])

scheme = shfl.iterate_gs(g0, multiplier, n_shuffles=2, iter_depth=1)

imp_response = rsps.impulse(scheme)
imp_response = rsps.array_to_fraction(imp_response)
imp_response_partfrac = rsps.matlab_partfrac(imp_response, delete_files=False)
# time_domain = rsps.inverse_lb(imp_response_partfrac)
# t_func = rsps.time_function(time_domain)


# t = np.linspace(0, 10, 1000)
# y = t_func(t)
# fig = plt.figure()
# ax = fig.gca()
# ax.plot(t, y)