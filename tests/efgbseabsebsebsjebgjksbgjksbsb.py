# -*- coding: utf-8 -*-
"""
Created on Wed May  3 12:37:44 2023

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


part_fracs = 1/a - 1/(a + a**2*x0_sym)
print(part_fracs)

print(rsps.inverse_lb([part_fracs]))
