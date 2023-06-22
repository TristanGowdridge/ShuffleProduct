# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:47:25 2023

@author: trist
"""

import dill
from params import k2, k3, a1, a2, iter_depth
from sympy import symbols, lambdify
import os
from concurrent.futures import ProcessPoolExecutor
import params
import pickle as pkl


if __name__ == "__main__":
    with open("quad_cube_y6_partfrac_symbolic.txt", "rb") as f_sym:
        loaded = pkl.load(f_sym)
    
    result = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for r in executor.map(params.worker, loaded):
            if isinstance(r, (tuple, list)):
                result.extend(r)
            else:
                result.append(r)

    result = tuple(result)
        
    _a1, _a2, _k2, _k3 = symbols("a1 a2 k2 k3")
    
    vals = {
        _a1: a1,
        _a2: a2,
        _k2: k2,
        _k3: k3,
    }
    
    with (
            open("quad_cube_y6_amp_var_gen.txt", "wb") as f_write1,
            open("quad_cube_y6_lambdify_A_t_gen.txt", "wb") as f_write2
    ):
        print("Processing 6")
        # Load the pickled terms.
        temp = sum(result)
        
        print("Subbing in 6")
        # Sub in params, leaving A and t.
        temp = temp.subs(vals)
        pkl.dump(temp, f_write1)
        
        print("Lambdifying 6")
        # Storing the lambdified functions with A and t as parameters.
        lamb_temp = lambdify(symbols("A t"), temp, "numpy")
        dill.settings["recurse"] = True
        dill.dump(lamb_temp, f_write2)
        print()
