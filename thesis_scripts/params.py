# -*- coding: utf-8 -*-
"""
Created on Mon Jun 5 11:38:48 2023

@author: trist
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.getcwd()) + r"/shuffleproduct")
import responses as rsps
import shuffle as shfl
from sympy.core.add import Add as SympyAdd


def plot(y, figax=None, legend_label="y", **kwargs):
    # Plot the results
    if not figax:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.gca()
        ax.set_xlim(t_window)
        ax.set_xlabel('t')
        ax.set_ylabel('y')
        ax.grid(True)
    else:
        fig, ax = figax
    
    ax.plot(t, y, label=legend_label, **kwargs)
    ax.legend()

    return (fig, ax)


def to_bmatrix(term):
    """
    
    """
    coeff = str(term[0])
    term = term[1]
    
    row1 = [coeff.replace('**', '^').replace('*', '').replace('k', 'k_')]
    row2 = []
    for i, j in zip(term[0], term[1]):
        i, j = str(i), str(j)
        row1.append(i.replace('x0', 'x_0').replace('x1', 'x_1'))
        j = j.replace('a1', 'a_1').replace('a2', 'a_2').replace("*", '')
        row2.append(j)
    else:
        row2.append("0")
    
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(line) + r' \\' for line in (row1, row2)]
    rv += [r'\end{bmatrix}']
    
    print("\n".join(rv))


def worker(term):
    """
    Worker function for the conversion.
    """
    if isinstance(term, SympyAdd):
        ts = []
        for term1 in term.make_args(term):
            ts.append(rsps.convert_term(term1))
        return tuple(ts)
    else:
        return rsps.convert_term(term)


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
t_span = (0.0, 0.2)
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
A_log = np.outer(np.logspace(-4, -1, 4), np.arange(1, 10)).flatten()


# Plotting Params
FONTSIZE = 18

