# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:11:31 2023

@author: trist
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.getcwd()) + "\shuffleproduct")

import numpy as np

import matplotlib.pyplot as plt
from sympy import lambdify, Symbol

import responses as rsps
from shuffle import GeneratingSeries, iterate_gs

a = 3
b = 7
x0 = 0
x1 = 1


multiplier = np.array([
    [-b, x0],
    [ a,  0]
])

g0 = GeneratingSeries(np.array([
    [ 1, x1],
    [ a,  0]
]))


scheme = iterate_gs(g0, multiplier, 2, 3)

gwn = rsps.gwn_response(scheme)

sum_of_partials = rsps.matlab_partfrac(gwn, delete_files=False)
t_domain = rsps.inverse_lb(sum_of_partials)

t_func = lambdify(Symbol('t'), t_domain)
times = np.linspace(0, 10, 1000)
y = t_func(times)


fig = plt.figure()
ax = fig.gca()

ax.plot(times, y)
