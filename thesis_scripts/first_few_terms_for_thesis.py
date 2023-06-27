# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:24:05 2023

@author: trist
"""

import os
import sys

from sympy import symbols, Matrix, print_latex
sys.path.insert(0, os.path.dirname(os.getcwd()) + r"/shuffleproduct")
import shuffle as shfl
from generating_series import GeneratingSeries as GS


iter_depth = 1

_k2, _k3, _x0, _x1, _a1, _a2, _A = symbols("k2 k3 x0 x1 a1 a2 A")

g0 = GS([
    [  1, _x0, _x1],
    [_a1, _a2,   0]
])

mult_quad = GS([
    [-_k2, _x0, _x0],
    [ _a1, _a2,   0]
])
mult_cube = GS([
    [-_k3, _x0, _x0],
    [ _a1, _a2,  0]
])

g1_quad = []
for i in shfl.binary_shuffle(g0, g0):
    i.prepend_multiplier(mult_quad)
    mat_quad = Matrix([[i.coeff, *i.words], [*i.dens, 0]])
    g1_quad.append(mat_quad)
    print_latex(mat_quad)


g1_cube = []
for i in shfl.nShuffles(g0, g0, g0):
    i.prepend_multiplier(mult_cube)

    mat_cube = Matrix([[i.coeff, *i.words], [*i.dens, 0]])
    g1_cube.append(mat_cube)
    print_latex(mat_cube)
    
    