# -*- coding: utf-8 -*-
"""
Created on Mon Jun 5 11:38:48 2023

@author: trist
"""
import numpy as np



import shuffle as shfl


# System params
m = 1
c = 20
k1 = 1e4
k2 = 1e7
k3 = 5e9

A = 0.07

a1, a2 = shfl.sdof_roots(m, c, k1)

dr = c / (2 * np.sqrt(m * k1))
wn = np.sqrt(k1 / m)
wd = wn * np.sqrt(1 - dr ** 2)


# Generating series run params.
iter_depth = 5  # Gives iter_depth + 1 terms as y_1 is the linear term.


# Time span
t_span = (0.0, 1.0)
t_window = (0.0, 0.2)
dt = 1e-4
t = np.arange(t_span[0], t_span[1], dt)


# Initial conditions
init_cond = np.array([0.0, 0.0])

# For MSE analysis
A_min = 0.00
A_max = 0.15
A_step = 0.01
A_range = np.arange(A_min, A_max + A_step, A_step)
A_log = np.logspace(-4, 0, 200)


# Plotting Params
FONTSIZE = 18
