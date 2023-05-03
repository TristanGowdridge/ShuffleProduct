# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:42:27 2023

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


scheme = shfl.iterate_gs(g0, multiplier, n_shuffles=2, iter_depth=5)
scheme = rsps.step_input(scheme)
sum_of_partials = rsps.matlab_partfrac(scheme)
t_domain = rsps.inverse_lb(sum_of_partials)

t_func = lambdify(Symbol('t'), t_domain)
times = np.linspace(0, 10, 1000)
y = t_func(times)


fig = plt.figure()
ax = fig.gca()

ax.plot(times, y)