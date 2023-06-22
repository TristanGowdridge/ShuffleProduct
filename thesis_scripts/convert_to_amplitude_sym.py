# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 08:12:13 2023

@author: trist

This loads in the pickled Volterra kernels and substitutes in some values.
The fully symbolic Volterra kernels are loaded in from quad_cube_y{i}_gen.txt,
these then have the system parameters substituted in to give the equations in a
partial symbolic form, where A and t are the only remaining parameters, these
are stored in quad_cube_y{i}_amp_var_gen.txt. The lambdified functions of these
are then stored in quad_cube_y{i}_lambdify_A_t_gen.txt, which take A as the
first parameter and t as the second.
"""
import pickle as pkl
import dill
from params import k2, k3, a1, a2, iter_depth
from sympy import symbols, lambdify, Add


_a1, _a2, _k2, _k3 = symbols("a1 a2 k2 k3")

vals = {
    _a1: a1,
    _a2: a2,
    _k2: k2,
    _k3: k3,
}


for i in range(1, iter_depth+2):
    with (
            open(f"quad_cube_y{i}_volt_sym.txt", "rb") as f_read,
            open(f"quad_cube_y{i}_amp_var_gen.txt", "wb") as f_write1,
            open(f"quad_cube_y{i}_lambdify_A_t_gen.txt", "wb") as f_write2
    ):
        print(f"Processing {i}")
        # Load the pickled terms.
        temp = Add(*pkl.load(f_read))
        
        print(f"Subbing in {i}")
        # Sub in params, leaving A and t.
        temp = temp.subs(vals)
        pkl.dump(temp, f_write1)
        
        print(f"Lambdifying {i}")
        # Storing the lambdified functions with A and t as parameters.
        lamb_temp = lambdify(symbols("A t"), temp, "numpy")
        dill.settings["recurse"] = True
        dill.dump(lamb_temp, f_write2)
        print()
