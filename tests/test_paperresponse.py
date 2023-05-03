# -*- coding: utf-8 -*-
"""
Created on Tue May 2 17:16:23 2023

@author: trist
"""
import unittest

import numpy as np
from sympy import Symbol
from sympy.functions.elementary.exponential import exp

import shuffle as shfl
import responses as rsps

x0 = 0
x1 = 1
b = 1
a = 1
t = Symbol('t')

class TestPaper4(unittest.TestCase):
    """
    Compares the responses obtained to paper 4 in the papers directory.
    """
    multiplier = np.array([
        [-b, x0],
        [ a,  0]
    ])
    
    g0 = shfl.GeneratingSeries(np.array([
        [ 1, x1],
        [ a,  0]
    ]))
    
    iter_args = (g0, multiplier, 2)
    
    t0 = (1/a)*(1 - exp(-a*t))
    t1 = -(b/a**3)*(1-2*a*t*exp(-a*t)-exp(-2*a*t))

    def test_step_input(self):
        all_gs = shfl.iterate_gs(*TestPaper4.iter_args, 1)
        
        imp_response = rsps.step_input(all_gs)
        imp_response_partfrac = rsps.matlab_partfrac(imp_response)
        time_domain = rsps.inverse_lb(imp_response_partfrac)
        
        
        assert (time_domain-TestPaper.t0-TestPaper.t1 == 0)
        
if __name__ == "__main__":
    unittest.main()