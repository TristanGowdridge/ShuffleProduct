# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:06:17 2023

@author: trist
"""
from collections import defaultdict
from operator import itemgetter
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

import shufflesym as shfl
import responses as rsps
from generating import GeneratingSeries

# from params import m, c, k1, k2, k3, A

from sympy import symbols

k2, k3, x0, x1, a1, a2, A = symbols("k2 k3 x0 x1 a1 a2 A")

g0 = GeneratingSeries([
    [ 1, x0, x1],
    [a1, a2,  0]
])


mult_quad = GeneratingSeries([
    [-k2, x0, x0],
    [ a1, a2,  0]
])


def convert_gs_to_time(g):
    """
    Takes the list of generating series and returns the associated time
    function.
    """
    g = rsps.matlab_partfrac(g)
    y = rsps.inverse_lb(g)
    return y


scheme = rsps.impulse_from_itersym(g0, mult_quad, 2, 2, amplitude=A)
y = convert_gs_to_time(scheme)


r1, r2 = shfl.sdof_roots(1, 20, 1e4)

vals = {
    A: 0.03,
    a1: r1,
    a2: r2,
    k2: 1e7
}
y1 = y.subs(vals)

# # t = np.linspace(0, 1, 1000)

# # fig = plt.figure()
# # ax = fig.gca()
# # ax.plot(t, y1(t), label="y1")
# # ax.plot(t, y1(t)+y2(t), label="y1 + y2")
# # ax.plot(t, y1(t)+y2(t)+y3(t), label="y1 + y2 + y3")
# # ax.legend()
