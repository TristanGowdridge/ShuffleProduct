# -*- coding: utf-8 -*-
"""
Created on Thu Jun 8 07:59:31 2023

@author: trist
"""

import os
import sys
from collections import defaultdict
from operator import itemgetter
from itertools import product

import numpy as np
from sympy import symbols, lambdify

from params import A, m, c, k1, k2, k3, plot, t

sys.path.insert(0, os.path.dirname(os.getcwd()) + r"\shuffleproduct")
import shufflesym as shfl
import responses as rsps
from generating import GeneratingSeries


_k2, _x0, _x1, _a1, _a2, _A = symbols("k2 x0 x1 a1 a2 A")

g0 = GeneratingSeries([
    [  1, _x0, _x1],
    [_a1, _a2,   0]
])


mult_quad = GeneratingSeries([
    [-_k2, _x0, _x0],
    [ _a1, _a2,   0]
])


def remove_nonimp(terms):
    """
    Have to define this as want g0, g1, g2 separate
    """
    store = []
    for term in terms:
        has_been_x1 = False
        for val in term.words:
            if val == _x1:
                has_been_x1 = True
                continue
            elif val == _x0 and (not has_been_x1):
                continue
            else:
                break
        else:
            store.append(term)
            
    return store


def convert_gs_to_time(terms, A):
    """
    Takes the list of generating series and returns the associated time
    function.
    """
    g = shfl.handle_output_type({0: terms}, tuple)
    g = rsps.impulsesym(g, A)
    
    g = rsps.matlab_partfrac(g)
    y = rsps.inverse_lb(g)
    
    return y


scheme = shfl.iterate_gs(g0, mult_quad, 2, 2, return_type=dict)

y1 = convert_gs_to_time(scheme[0], A)
y2 = convert_gs_to_time(scheme[1], A)
y3 = convert_gs_to_time(scheme[2], A)

a1, a2 = shfl.sdof_roots(m, c, k1)

vals = {
    _A: A,
    _a1: a1,
    _a2: a2,
    _k2: k2,
}

y11 = lambdify(symbols('t'), y1.subs(vals))(t)
y21 = lambdify(symbols('t'), y2.subs(vals))(t)
y31 = lambdify(symbols('t'), y3.subs(vals))(t)

np.save("y1_sym_quad.npy", y11)
np.save("y2_sym_quad.npy", y21)
np.save("y3_sym_quad.npy", y31)
    
figax = plot(y11, None, "$y^g_1$")
figax = plot(y11 + y21, figax, "$y^g_1 + y^g_2$")
figax = plot(y11 + y21 + y31, figax, "$y^g_1 + y^g_2 + y^g_3$")
