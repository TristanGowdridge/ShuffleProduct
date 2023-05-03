# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 13:25:20 2023

@author: tristan

This script provides a quick demonstration of the shuffle module. The
generating series terms will be determined in accordance with "An algebraic
approach to nonlinear functional expansions"- Michel Fleiss. 

The first generating series term (g1) will be determined manually to showcase
the procedure. This is the most flexible method given the current state of the
module. Currently the automated function (iterate_gs) does not allow for
iterative schemes that contain multiple chained shuffle instances. I.e a scheme
with both gs_i1 Ш gs_i2 Ш gs_i3 and gs_j1 Ш gs_j2. If you wanted to perform
this analysis, it would have to be performed manually.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.getcwd()) + "\shuffleproduct")

import numpy as np
from sympy import Symbol

from shuffle import BinaryShuffle, GeneratingSeries, iterate_gs

# Defining some symbols that will be used throughout this demonstration.
x0 = Symbol("x0")
x1 = Symbol("x1")
b = Symbol('b')
a = Symbol('a')


# Convert a numpy array into a GeneratingSeries instance. The GeneratingSeries
# class provides some useful functionality when determining the shuffle
# product.
g0 = GeneratingSeries(np.array([
    [ 1, x1],
    [ a,  0]
]))


# =============================================================================
# Manual Iteration
# =============================================================================
# To calculate higher order terms, call a instance of BinaryShuffle on the 
# GeneratingSeries objects that you want to shuffle.
shuffle_object = BinaryShuffle()

g1 = shuffle_object(g0, g0)
print("g1 without multiplier\n", *g1)

# g1 contains the shuffle product of g0 and g0, note that a list type is
# returned with a single term inside, since there is only one term produced
# from this application of the shuffle product. For longer terms, this will
# not be the case.

# Since BinaryShuffle only calculates the shuffle product between the two 
# generating series, we need to prepend the "multiplier". Note that multiplier
# is a numpy array and not an instance of GeneratingSeries.
multiplier = np.array([
    [-b, x0],
    [ a,  0]
])

# The GeneratingSeries class has a method, prepend_multiplier(), which will
# return the GeneratingSeries instance with the multiplier prepended. Since 
# there is only one term, I will not used a for-loop and just reference it 
# directly.
g1 = g1[0].prepend_multiplier(multiplier)
print("\ng1 with multiplier prepended\n", g1)

# This process would then have to be repeated manually... Note that there is a 
# function called collect(), which collects the like terms. When calculating
# higher order terms, collect() will prevent repeating the shuffle product
# multiple times for the same inputs (caching may solve this, but better not
# left to chance).

# There is also a function nShuffles() which automates chains of shuffles
# greater than two i.e, gs_1 Ш gs_2 Ш gs_3 with two applications of the shuffle
# product or gs_1 Ш ... Ш gs_n with n-1 applications of the shuffle product.


# =============================================================================
# Automated Iteration
# =============================================================================
# This process can be automated for simple iterative schemes (only one chain of 
# shuffle products). As before, the g0 term refers to the first term in the
# iterative scheme, and multiplier refers the the term prepended after all the
# shuffles have been computed. iter_depth gives the maximum iterations depth.
# I think the shuffle product is O(2 ** N), as each call results in another
# two. I've attempted to optimise the code somewhat, with caching and using
# the O(1) nature of hashing look-up. If there are any suggestions for
# optimisation, I'm all ears. However, I think there are bottlenecks further
# down the analysis pipeline (converting to partial fractions), so further
# optimsations here might be somewhat pointless? I could be wrong about the
# bottlenecks further down the line. Currently, it takes ~20s on my computer 
# for iter_depth=7, for the example outlined here.

# More information regarding iterate_gs() can be found in the source code.
gs = iterate_gs(g0, multiplier, n_shuffles=2, iter_depth=4, return_type=dict)

for i, gs_i in gs.items():
    print(f"\ngs for iteration depth {i}")
    print(*gs_i, sep='\n')

# These values have been compared to the ones in the paper.