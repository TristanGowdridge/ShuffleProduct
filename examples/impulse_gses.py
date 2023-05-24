# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:00:24 2023

@author: trist

OEIS: A074664- Number of algebraically independent elements of degree n in the
algebra of symmetric polynomials in noncommuting variables.

Matches the number of terms formed for the impulse response expansion for
quadratic duffing. [1, 1, 2, 6, 22, 92, 426, ...]. Lot's of similar words being
thrown around Hopf algebra etc...
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.getcwd()) + r"\shuffleproduct")
import shuffle as shfl
import responses as rsps
from generating_series import GeneratingSeries

x0 = 0
x1 = 1


x_init = 0
v_init = 0
amplitude = 1  # No greater than 10.


a1 = 1
a2 = 100
k3 = 1
g0 = shfl.GeneratingSeries(np.array([
    [ 1, x0, x1],
    [a1, a2,  0]
]))


multiplier = [np.array([
    [-k3, x0, x0],
    [ a1, a2,  0]
])]

imp_response_dict = rsps.impulse_from_iter(
    g0, multiplier, n_shuffles=2, iter_depth=2, return_type=dict
)

imp_response_tuple = shfl.handle_output_type(imp_response_dict, tuple)

# print(*imp_response, sep="\n\n")
