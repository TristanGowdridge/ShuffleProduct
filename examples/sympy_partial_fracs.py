# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:07:28 2023

@author: trist
"""
import shuffleproduct.shuffle as shfl
from shuffleproduct.generating_series import GeneratingSeries
import shuffleproduct.responses as rsps

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

t = sym.Symbol("t")


x0 = 0
x1 = 1

k1 = 1
k2 = 2

multiplier = np.array([
    [-k2, x0],
    [ k1,  0]
])
g0 = shfl.GeneratingSeries(np.array([
    [ 1, x1],
    [k1,  0]
]))
scheme = shfl.iterate_gs(g0, multiplier, n_shuffles=2, iter_depth=5)
a = rsps.step_input(scheme)

i = 6
pf1 = a[i].apart()
pf2 = a[i].apart(full=True).doit()

to_mat = lambda x: "partfrac(" + str(x).replace("**", "^") +")"
print(to_mat(a[i]))