# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 06:47:13 2023

@author: trist

I needed to run for one instance as the inverse laplace borel for the
exponential wasn't working. This file loads the partial fraction form of the
generating series and converts to the time domain (y6).
"""
import os
import pickle as pkl
from concurrent.futures import ProcessPoolExecutor
import params

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
    with open("quad_cube_y6_volt_sym.txt", "wb") as f_sym:
        pkl.dump(result, f_sym)
