# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 18:02:52 2023

@author: trist
"""

import pickle as pkl
import dill
from params import k2, k3, a1, a2, iter_depth
from sympy import symbols, lambdify, Add

from sympy.core.add import Add as SympyAdd


_a1, _a2, _k2, _k3 = symbols("a1 a2 k2 k3")

vals = {
    _a1: a1,
    _a2: a2,
    _k2: k2,
    _k3: k3,
}



with (
        open(f"quad_cube_y6_amp_var_gen.txt", "rb") as f
):

    part_subbed = pkl.load(f)
args = SympyAdd.make_args(part_subbed)

store_lambdas = []
for arg in args:
    # Storing the lambdified functions with A and t as parameters.
    store_lambdas.append(lambdify(symbols("A t"), arg, "numpy"))




import numpy as np
def y6_to_dill(A, t):
    return np.sum([f(A, t) for f in store_lambdas], axis=0)

     
with (
        open("quad_cube_y6_lambdify_A_t_gen.txt", "wb") as f_write2
):
    dill.settings["recurse"] = True
    dill.dump(y6_to_dill, f_write2)
