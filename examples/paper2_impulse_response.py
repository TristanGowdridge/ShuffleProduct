# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:42:27 2023

@author: trist

This shows the results for the impulse response in the paper An Algebraic
Approach to Nonliner Functional Expansions - Fleiss.
"""
import numpy as np
from sympy import Symbol, lambdify
import matplotlib.pyplot as plt

import shuffleproduct.responses as rsps
import shuffleproduct.shuffle as shfl
from shuffleproduct.generating_series import GeneratingSeries

# x0 and x1 are the generating series terms used in the expansion. These are
# required to be integers rather than symbolic, because symbolic objects
# convert the numpy arrays to object arrays with dtype="O", which essentially
# an array of pointers, then when calculating the hash (which is required in
# BinaryShuffle and GeneratingSeries) the result is non deterministic.
x0 = 0
x1 = 1

# k1 and k2 are system parameters nonlinear differential equation being solved.
k1 = 7
k2 = 5

# The term that prepended after each iteration of the shuffle product.
multiplier = GeneratingSeries([
    [-k2, x0],
    [ k1,  0]
])

# The initial term in the iterative scheme.
g0 = GeneratingSeries([
    [ 1, x1],
    [k1,  0]
])

# Calculate the iterative scheme of to depth 5.
scheme = shfl.iterate_gs(g0, multiplier, n_shuffles=2, iter_depth=5)

# Apply the conditions for an impulse input to the generating series.
scheme = rsps.impulse(scheme, amplitude=5)

# Decomposes the generating series terms into partial fractions. This
# decompostion is required for the inverse Laplace-Borel Transform.
gs_decomposed = rsps.sympy_partfrac_here(scheme)

# inverse_lb determines the inverse Laplace-Borel transform of the decomposed
# generating series, meaning they are now in the time domain.
time_domain = rsps.inverse_lb(gs_decomposed)

# Vectorises the equation to now take time as an input, which then can be
# plotted.
t_func = lambdify(Symbol('t'), time_domain)


fig = plt.figure()
ax = fig.gca()
time_vec = np.linspace(0, 10, 1000)
y = t_func(time_vec)
ax.plot(time_vec, y)