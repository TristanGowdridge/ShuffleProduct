# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:16:01 2023

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


_k2, _k3, _x0, _x1, _a1, _a2, _A = symbols("k2 k3 x0 x1 a1 a2 A")

g0 = GeneratingSeries([
    [  1, _x0, _x1],
    [_a1, _a2,   0]
])


mult_quad = GeneratingSeries([
    [-_k2, _x0, _x0],
    [ _a1, _a2,   0]
])
mult_cube = GeneratingSeries([
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
    term_storage[0].append(g0)

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
            gs_term.prepend_multiplier(mult_quad)
        term_storage[depth + 1].extend(term_storage_quad[depth+1])
        
        for gs_term in term_storage_cube[depth+1]:
            gs_term.prepend_multiplier(mult_cube)
        term_storage[depth+1].extend(term_storage_cube[depth+1])
        
        term_storage[depth+1] = shfl.collect(term_storage[depth+1])
        
    return term_storage


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


scheme = iterate_quad_cubic(2)

y1 = convert_gs_to_time(scheme[0], A)
y2 = convert_gs_to_time(scheme[1], A)
y3 = convert_gs_to_time(scheme[2], A)

a1, a2 = shfl.sdof_roots(m, c, k1)

vals = {
    _A: A,
    _a1: a1,
    _a2: a2,
    _k2: k2,
    _k3: k3,
}

y11 = lambdify(symbols('t'), y1.subs(vals))(t)
y21 = lambdify(symbols('t'), y2.subs(vals))(t)
y31 = lambdify(symbols('t'), y3.subs(vals))(t)

np.save("y1_sym_quad_cube.npy", y11)
np.save("y2_sym_quad_cube.npy", y21)
np.save("y3_sym_quad_cube.npy", y31)
    
figax = plot(y11, None, "$y^g_1$")
figax = plot(y11 + y21, figax, "$y^g_1 + y^g_2$")
figax = plot(y11 + y21 + y31, figax, "$y^g_1 + y^g_2 + y^g_3$")
