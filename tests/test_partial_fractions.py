# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:15:51 2023

@author: trist
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.getcwd()) + "\shuffleproduct")


import unittest

import numpy as np
from sympy import Symbol, simplify
from sympy.functions.elementary.exponential import exp

import responses as rsps
from shuffle import GeneratingSeries

a = 5
b = 7

x0 = 0
x1 = 1

x0_sym = Symbol("x0")
x1_sym = Symbol("x1")
t_sym = Symbol('t')


class TestPaper4(unittest.TestCase):
    """
    Regression tests against the terms obtained in paper 4. This deals with the
    converting to the time domain section of the problem.    
    """
    
    g0 = (1, GeneratingSeries(
        np.array([
            [x1],
            [ a]])
        ))
    g1 = (-2*b, GeneratingSeries(
        np.array([[x0,  x1, x1],
                  [ a, 2*a,  a]])
        ))
    
    g2 = (4*b**2, GeneratingSeries(
        np.array([[x0,  x1, x0,  x1, x1],
                  [ a, 2*a,  a, 2*a,  a]])
        ))
    
    def test_term1_arr2frac(self):
        test = rsps.array_to_fraction([TestPaper4.g0])[0]
        assert test.equals(x1_sym / (1 + a*x0_sym))
    
    def test_term1_step(self):
        g0 = rsps.step_input([TestPaper4.g0])
        part_fracs = rsps.matlab_partfrac(g0)
        assert part_fracs.equals(simplify(1/a - 1/(a + a**2*x0_sym)))

    def test_term1_lb(self):
        part_fracs = 1/a - 1/(a + a**2*x0_sym)
        time = rsps.inverse_lb([part_fracs])[0]
        assert time.equals((1/a)*(1 - exp(-a*t_sym)))                 
        
    
    def test_term2_arr2frac(self):
        test = rsps.array_to_fraction([TestPaper4.g1])[0]
        numerator = -2*b * x0_sym*x1_sym**2
        denominator =  (1 + a*x0_sym)**2 * (1 + 2*a*x0_sym)
        assert test.equals(numerator / denominator)
        
    def test_term3_arr2frac(self):
        test = rsps.array_to_fraction([TestPaper4.g2])[0]
        test = simplify(test)
        numerator = 4*b**2 * x0_sym**2 * x1_sym**3
        denominator =  (1 + a*x0_sym)**3 * (1 + 2*a*x0_sym)**2
        term = simplify(numerator / denominator)
        
        assert term.equals(numerator / denominator)
    
    
    
    
    
    
    
    
    
        
if __name__ == "__main__":
    unittest.main()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    