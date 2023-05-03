# -*- coding: utf-8 -*-
"""
Created on Tue May  2 13:30:43 2023

@author: trist
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.getcwd()) + "\shuffleproduct")

import numpy as np
from sympy import Symbol, lambdify
import matplotlib.pyplot as plt

import responses as rsps
import shuffle as shfl


x0 = 0
x1 = 1
k1 = 10
k2 = 5

multiplier = np.array([
    [-k2, x0],
    [ k1,  0]
])

g0 = shfl.GeneratingSeries(np.array([
    [ 1, x1],
    [k1,  0]
]))

scheme = shfl.iterate_gs(g0, multiplier, n_shuffles=2, iter_depth=2)
imp_response = rsps.impulse(scheme)
imp_response = rsps.array_to_fraction(imp_response)

imp_response_partfrac = rsps.matlab_partfrac(imp_response)

time_domain = rsps.inverse_lb(imp_response_partfrac)

t_func = rsps.time_function(time_domain)

t = np.linspace(0, 3, 1000)
y = t_func(t)

fig = plt.figure()
ax = fig.gca()
ax.plot(t, y)

