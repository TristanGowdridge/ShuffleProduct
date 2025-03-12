# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:16:01 2023

@author: trist
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.getcwd()) + r"\shuffleproduct")
from collections import defaultdict
from operator import itemgetter
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

import shuffle as shfl
import responses as rsps
from generating_series import GeneratingSeries

from params import m, c, k1, k2, k3, A, t, plot

x0 = 0
x1 = 1

a1, a2 = shfl.sdof_roots(m, c, k1)

g0 = GeneratingSeries([
    [ 1, x0, x1],
    [a1, a2,  0]
])


mult_quad = np.array([
    [-k2, x0, x0],
    [ a1, a2,  0]
])
mult_cube = np.array([
    [-k3, x0, x0],
    [ a1, a2,  0]
])



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
        
        for part in shfl.partitions(depth, 3):
            terms = itemgetter(*part)(term_storage)
            for in_perm in product(*terms):
                term_storage_cube[depth+1].extend(shfl.nShuffles(*in_perm))
        term_storage_cube[depth+1] = shfl.collect(term_storage_cube[depth+1])
    
        # After the shuffles for this iteration's depth have been caluclated,
        # prepend the multiplier to each term.
        next_terms = []
        for gs_term in term_storage_quad[depth+1]:
            next_terms.append(gs_term.prepend_multiplier(mult_quad))
        term_storage[depth + 1].extend(next_terms)
        
        next_terms = []
        for gs_term in term_storage_cube[depth+1]:
            next_terms.append(gs_term.prepend_multiplier(mult_cube))
        term_storage[depth+1].extend(next_terms)
        
        term_storage[depth+1] = shfl.collect(term_storage[depth+1])
        
    return term_storage


def convert_gs_to_time(terms):
    """
    Takes the list of generating series and returns the associated time
    function.
    """
    g = shfl.handle_output_type({0: terms}, tuple)
    g = rsps.impulse(g, A)
    g = rsps.matlab_partfrac(g)
    y = rsps.inverse_lb(g)
    
    return rsps.time_function(y)


scheme = iterate_quad_cubic(2)

y1 = convert_gs_to_time(scheme[0])(t)
y2 = convert_gs_to_time(scheme[1])(t)
y3 = convert_gs_to_time(scheme[2])(t)

np.save("y1_num_quad_cube.npy", y1)
np.save("y2_num_quad_cube.npy", y2)
np.save("y3_num_quad_cube.npy", y3)
    
figax = plot(y1, None, "$y^g_1$")
figax = plot(y1 + y2, figax, "$y^g_1 + y^g_2$")
figax = plot(y1 + y2 + y3, figax, "$y^g_1 + y^g_2 + y^g_3$")
