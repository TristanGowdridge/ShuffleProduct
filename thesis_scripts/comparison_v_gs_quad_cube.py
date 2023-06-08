# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 08:53:42 2023

@author: trist
"""
import os
import sys
from collections import defaultdict
from operator import itemgetter
from itertools import product
from math import prod

import numpy as np
from sympy import symbols, lambdify
from sympy.printing import latex
import matplotlib.pyplot as plt

from params import A, m, c, k1, k2, k3, plot, t
from vci_quad_cube import y2 as y2_volt
from vci_quad_cube import y3 as y3_volt

sys.path.insert(0, os.path.dirname(os.getcwd()) + r"\shuffleproduct")
import shufflesym as shfl
import responses as rsps
from generating import GeneratingSeries


_k2, _k3, _x0, _x1, _a1, _a2, _A = symbols("k2 k3 x0 x1 a1 a2 A")

_g0 = GeneratingSeries([
    [  1, _x0, _x1],
    [_a1, _a2,   0]
])


_mult_quad = GeneratingSeries([
    [-_k2, _x0, _x0],
    [ _a1, _a2,   0]
])
_mult_cube = GeneratingSeries([
    [-_k3, _x0, _x0],
    [ _a1, _a2,  0]
])


def remove_nonimp(terms):
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


def iterate_quad_cubic(iter_depth):
    """
    A very hastily written iterative expansion of a SDOF oscillator with
    quadratic and cubic nonlinearities.
    
    This function is reliant on global variables, be careful! It also isn't
    generalisable at all, but it's what we need for our specific example.
    """
    term_storage = defaultdict(list)
    term_storage[0].append(_g0)

    term_storage_quad = defaultdict(list)
    term_storage_cube = defaultdict(list)
    
    for depth in range(iter_depth):
        for part in shfl.partitions(depth, 2):
            terms = itemgetter(*part)(term_storage)
            for in_perm in product(*terms):
                term_storage_quad[depth+1].extend(shfl.nShuffles(*in_perm))
        term_storage_quad[depth+1] = shfl.collect(term_storage_quad[depth+1])
        term_storage_quad[depth+1] = remove_nonimp(term_storage_quad[depth+1])
        
        for part in shfl.partitions(depth, 3):
            terms = itemgetter(*part)(term_storage)
            for in_perm in product(*terms):
                term_storage_cube[depth+1].extend(shfl.nShuffles(*in_perm))
        term_storage_cube[depth+1] = shfl.collect(term_storage_cube[depth+1])
        term_storage_cube[depth+1] = remove_nonimp(term_storage_cube[depth+1])
    
        # After the shuffles for this iteration's depth have been caluclated,
        # prepend the multiplier to each term.
        for gs_term in term_storage_quad[depth+1]:
            gs_term.prepend_multiplier(_mult_quad)
        term_storage[depth + 1].extend(term_storage_quad[depth+1])
        
        for gs_term in term_storage_cube[depth+1]:
            gs_term.prepend_multiplier(_mult_cube)
        term_storage[depth+1].extend(term_storage_cube[depth+1])
        
        term_storage[depth+1] = shfl.collect(term_storage[depth+1])
        
    return term_storage


scheme = iterate_quad_cubic(2)

term_number = 1
iter_dep = 1
g2 = shfl.handle_output_type({0: scheme[iter_dep]}, tuple)
print(f"ONLY CONSIDERING TERM {term_number} OF EXPANSION")
g2_imp = rsps.impulsesym([g2[term_number]], _A)
g2_pf = rsps.matlab_partfrac(g2_imp)
y2 = rsps.inverse_lb(g2_pf)


eq_fig = plt.figure(figsize=(20, 3))
eq_ax = eq_fig.gca()
eq_ax.axis("off")
_fntsz = 15
_spcng = 0.5


print(scheme[iter_dep][term_number])
eq_ax.text(-0.13, 2*_spcng, "$g_2^i$:" + f"${latex(g2_imp)}$", fontsize=_fntsz)
eq_ax.text(-0.13, _spcng, "$g_2^{pf}$:" + f"${latex(g2_pf)}$", fontsize=_fntsz)
eq_ax.text(-0.13, 0, f"$y_2^g: {latex(y2)}$", fontsize=_fntsz)


a1, a2 = shfl.sdof_roots(m, c, k1)

vals = {
    _A: A,
    _a1: a1,
    _a2: a2,
    _k2: k2,
    _k3: k3,
}

y21 = lambdify(symbols('t'), y2.subs(vals))(t)
# y31 = lambdify(symbols('t'), y3.subs(vals))(t)

    
_figax = plot(y2_volt, None, "$y_2^v$")
# _figax = plot(y3_volt, _figax, "$y_3^v$")
_figax = plot(y21, _figax, "$y^g_2$")
# _figax = plot(y31, _figax, "$y^g_1 + y^g_2 + y^g_3$")

_fig, _ax = _figax
_ax.set_title(
    f"Comparison of Duffing's equation solutions with Dirac delta {A}"
)

ratio_2 = test = y21[1:]/y2_volt[1:]