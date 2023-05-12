# -*- coding: utf-8 -*-
"""
Created on Thu May  4 17:44:13 2023

@author: trist
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.getcwd()) + "\shuffleproduct")

import numpy as np
from sympy import Symbol
import matplotlib.pyplot as plt

import responses as rsps
import shuffle as shfl

t = Symbol("t")
time_vec = np.linspace(0, 3, 1000)

fig1 = plt.figure()
ax1 = fig1.gca()
fig1.suptitle("Impulse")

x0 = 0
x1 = 1

multipliers = [
    np.array([
        [1/3],
        [  0]
    ]),
    np.array([
        [-1/3, x0],
        [   1,  0]
    ])
]

g0 = shfl.GeneratingSeries(np.array([
    [1, x1],
    [1,  0]
]))

scheme = shfl.iterate_gs(g0, multipliers, n_shuffles=3, iter_depth=2)
imp = rsps.impulse(scheme, amplitude=0.63)
imp_pf = rsps.matlab_partfrac(imp)
imp_time_domain = rsps.inverse_lb(imp_pf)
imp_func = rsps.time_function(imp_time_domain)
y_imp = imp_func(time_vec)

ax1.plot(time_vec, y_imp)
ax1.legend()