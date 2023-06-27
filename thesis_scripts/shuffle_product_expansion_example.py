# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 12:46:21 2023

@author: trist

Example shown in the generating series chapter after the explanation of how to
compute the shuffle products.
"""
import os
import sys

from sympy import symbols

sys.path.insert(0, os.path.dirname(os.getcwd()) + r"/shuffleproduct")
import shuffle as shfl
from generating_series import GeneratingSeries


l1, l2, xi1, xi2, xj1, b0, b1, d0 = symbols(
    "\lambda_1 \lambda_2 x_{i_1} x_{i_2} x_{j_1} b_0 b_1 d_0"
)


g1 = GeneratingSeries([
    [l1, xi1, xi2],
    [b0,  b1,   0]
])

g2 = GeneratingSeries([
    [l2, xj1],
    [d0,   0]
])


result = shfl.binary_shuffle(g1, g2)