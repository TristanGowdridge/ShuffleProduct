# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:35:36 2023

@author: trist
"""
import sympy as sym
from sympy.functions.elementary.exponential import exp as sympyexp

import responses as rsps

x = sym.Symbol('x')
y = sym.Symbol('y')
t = sym.Symbol('t')


# Unit test cases that should only pass lb_unit()
_unit0 = 1
_unit1 = 1.1
_unit2 = sym.core.numbers.Number(2)
_unit3 = sym.core.numbers.Number(2.1)
_unit4 = sym.core.numbers.Number(1)
_unit5 = sym.core.numbers.Number(-1)
units = [_unit0, _unit1, _unit2, _unit3, _unit4, _unit5]


# Polynomial test cases that should only pass lb_polynomial()
_poly0 = x
_poly1 = 2 * x
_poly2 = 2.1 * x
_poly3 = x ** 2
_poly4 = 2 * x ** 2
_poly5 = 2.1 * x ** 2
polys = [
    _poly0, _poly1, _poly2, _poly3, _poly4, _poly5
    ]

_poly_t0 = t
_poly_t1 = 2 * t
_poly_t2 = 2.1 * t
_poly_t3 = t ** 2 / 2
_poly_t4 = t ** 2
_poly_t5 = (2.1 / 2)* t ** 2
polys_t = [
    _poly_t0, _poly_t1, _poly_t2, _poly_t3, _poly_t4, _poly_t5
    ]
assert (len(polys) == len(polys_t))


# Unit test cases that should only pass lb_exponential()
_exp0 = (1 - x) ** -2
_exp1 = 2 * (1 - x) ** -2
_exp2 = 2.1 * (1 - x) ** -2
_exp3 = (1 - 2 * x) ** -2
_exp4 = 2 * (1 - 2 * x) ** -2
_exp5 = 2.1 * (1 - 2 * x) ** -2
_exp6 = (1 - 2.1 * x) ** -2
_exp7 = 2 * (1 - 2.1 * x) ** -2
_exp8 = 2.1 * (1 - 2.1 * x) ** -2
_exp9 = (1 - 2.1 * x) ** -1
_exp10 = 2 * (1 - 2.1 * x) ** -1
_exp11 = 2.1 * (1 - 2.1 * x) ** -1
_exp12 = (3.2 - 2.1 * x) ** -1
_exp13 = (x - 3.2) ** -1
_exp14 = (x - 2) ** -4
exps = [
    _exp0, _exp1, _exp2, _exp3, _exp4, _exp5, _exp6, _exp7, _exp8, _exp9,
    _exp10, _exp11, _exp12, _exp13, _exp14
    ]

_exp_t0 = (1 + t) * sympyexp(t)
_exp_t1 = 2 * (1 + t) * sympyexp(t)
_exp_t2 = 2.1 * (1 + t) * sympyexp(t)
_exp_t3 = (1 + 2 * t) * sympyexp(2 * t)
_exp_t4 = 2 * (1 + 2 * t) * sympyexp(2 * t)
_exp_t5 = 2.1 * (1 + 2 * t) * sympyexp(2 * t)
_exp_t6 = (1 + 2.1 * t) * sympyexp(2.1 * t)
_exp_t7 = 2 * (1 + 2.1 * t) * sympyexp(2.1 * t)
_exp_t8 = 2.1 * (1 + 2.1 * t) * sympyexp(2.1 * t)
_exp_t9 = sympyexp(2.1 * t)
_exp_t10 = 2 * sympyexp(2.1 * t)
_exp_t11 = 2.1 * sympyexp(2.1 * t)
_exp_t12 = (1 / 3.2) * sympyexp((2.1/3.2) * t)
_exp_t13 = (1 / -3.2) * sympyexp((1 / 3.2) * t)
_exp_t14 = (-2)**-4 * (1 + 3*t/2 + 3*t**2/8 + t**3/48) * sympyexp(t/2)
exps_t = [
    _exp_t0, _exp_t1, _exp_t2, _exp_t3, _exp_t4, _exp_t5, _exp_t6, _exp_t7,
    _exp_t8, _exp_t9, _exp_t10, _exp_t11, _exp_t12, _exp_t13, _exp_t14
    ]
assert (len(exps) == len(exps_t))


# Unit test cases that should only pass lb_cosine()
_cos0 = (1 + x**2) ** -1
_cos1 = (1 + 2 * x**2) ** -1
_cos2 = (1 + 2.1 * x**2) ** -1
_cos3 = 2 * (1 + x**2) ** -1
_cos4 = 2 * (1 + 2 * x**2) ** -1
_cos5 = 2 * (1 + 2.1 * x**2) ** -1
_cos6 = 2.1 * (1 + x**2) ** -1
_cos7 = 2.1 * (1 + 2 * x**2) ** -1
_cos8 = 2.1 * (1 + 2.1 * x**2) ** -1
_cos9 = 2.1 * (3 + x**2) ** -1
_cos10 = 2.1 * (3 + 2 * x**2) ** -1
_cos11 = 2.1 * (3 + 2.1 * x**2) ** -1
coses = [
    _cos0, _cos1, _cos2, _cos3, _cos4, _cos5, _cos6, _cos7, _cos8, _cos9,
    _cos10, _cos11
    ]

_cos_t0 = sym.cos(t)
_cos_t1 = sym.cos(sym.sqrt(2) * t)
_cos_t2 = sym.cos(sym.sqrt(2.1) * t)
_cos_t3 = 2 * sym.cos(t)
_cos_t4 = 2 * sym.cos(sym.sqrt(2) * t)
_cos_t5 = 2 * sym.cos(sym.sqrt(2.1) * t)
_cos_t6 = 2.1 * sym.cos(t)
_cos_t7 = 2.1 * sym.cos(sym.sqrt(2) * t)
_cos_t8 = 2.1 * sym.cos(sym.sqrt(2.1) * t)
_cos_t9 = 2.1/3 * sym.cos(sym.sqrt(1/3) * t)
_cos_t10 = 2.1/3 * sym.cos(sym.sqrt(2/3) * t)
_cos_t11 = 2.1/3 * sym.cos(sym.sqrt(2.1/3) * t)
coses_t = [
    _cos_t0, _cos_t1, _cos_t2, _cos_t3, _cos_t4, _cos_t5, _cos_t6, _cos_t7,
    _cos_t8, _cos_t9, _cos_t10, _cos_t11
    ]


# Test cases that should fail every check.
_faulty0 = x * y
_faulty1 = x + y
_faulty2 = 3.2 * x + 7 * y
_faulty3 = 1 + x + x**2
_faulty4 = (1 + x + x**2) ** -2
_faulty5 = (3 + 2.1*x + 2.1*x**2) ** -2
_faulty6 = x ** 2.1

faulties = [
    _faulty0, _faulty1, _faulty2, _faulty3, _faulty4, _faulty5, _faulty6
    ]


# =============================================================================
# These checks ensure the accuracy of the form being analysed, and ensure that
# no cases are misidentified.
# =============================================================================
def TestUnitForm():
    try:
        for test_var in units:
            assert (test_var == rsps.lb_unit(test_var))
            
        for test_var in polys:
            assert not rsps.lb_unit(test_var)    
        
        for test_var in exps:
            assert not rsps.lb_unit(test_var)
        
        for test_var in coses:
            assert not rsps.lb_unit(test_var)
            
        for test_var in faulties:
            assert not rsps.lb_unit(test_var)
    
    except AssertionError:
        print('\n' + f"{test_var}" + '\n' )
        raise AssertionError
        
    
def TestPolynomialForm():
    try:
        for test_var in units:
            assert not rsps.lb_polynomial(test_var)
            
        for test_var in polys:
            assert rsps.lb_polynomial(test_var)
        
        for test_var in exps:
            assert not rsps.lb_polynomial(test_var)        
        
        for test_var in coses:
            assert not rsps.lb_polynomial(test_var)
            
        for test_var in faulties:
            assert not rsps.lb_polynomial(test_var)
        
    except AssertionError:
        print('\n' + f"{test_var}" + '\n' )
        raise AssertionError


def TestExponentialForm():
    try:
        for test_var in units:
            assert not rsps.lb_exponential(test_var)
            
        for test_var in polys:
            assert not rsps.lb_exponential(test_var)
            
        for test_var in exps:
            assert rsps.lb_exponential(test_var)        
        
        for test_var in coses:
            assert not rsps.lb_exponential(test_var)
            
        for test_var in faulties:
            assert not rsps.lb_exponential(test_var)
        
    except AssertionError:
        print('\n' + f"{test_var}" + '\n' )
        raise AssertionError    


def TestCosineForm():
    try:
        for test_var in units:
            assert not rsps.lb_cosine(test_var)
            
        for test_var in polys:
            assert not rsps.lb_cosine(test_var)
            
        for test_var in exps:
            assert not rsps.lb_cosine(test_var)
        
        for test_var in coses:
            assert rsps.lb_cosine(test_var)
        
        for test_var in faulties:
            assert not rsps.lb_cosine(test_var)
            
    except AssertionError:
        print('\n' + f"{test_var}" + '\n' )
        raise AssertionError


# =============================================================================
# These checks ensure that we are getting the correct answers for the Laplace-
# Borel transforms, by comparing against manually determined cases.
# =============================================================================
def TestUnit():
    try:
        for index, test_var in enumerate(units):
            assert(test_var ==  rsps.lb_unit(test_var))

    except AssertionError:
        print(f"Testcase : {index}")
        print(f"Input    : {test_var}")
        print(f"Answer   : {test_var}")
        print(f"Result   : {rsps.lb_unit(test_var)}")
        raise AssertionError


def TestPolynomial():
    try:
        for index, (ans, test_var) in enumerate(zip(polys_t, polys)):
            assert (ans.equals(rsps.lb_polynomial(test_var)))

    except AssertionError:
        print(f"Testcase : {index}")
        print(f"Input    : {test_var}")
        print(f"Answer   : {ans}")
        print(f"Result   : {rsps.lb_polynomial(test_var)}")
        raise AssertionError


def TestExponential():
    try:
        for index, (ans, test_var) in enumerate(zip(exps_t, exps)):
            assert (ans.equals(rsps.lb_exponential(test_var)))

    except AssertionError:
        print(f"Testcase : {index}")
        print(f"Input    : {test_var}")
        print(f"Answer   : {ans}")
        print(f"Result   : {rsps.lb_exponential(test_var)}")
        raise AssertionError


def TestCosine():
    try:
        for index, (ans, test_var) in enumerate(zip(coses_t, coses)):
            assert (ans.equals(rsps.lb_cosine(test_var)))

    except AssertionError:
        print(f"Testcase : {index}")
        print(f"Input    : {test_var}")
        print(f"Answer   : {ans}")
        print(f"Result   : {rsps.lb_cosine(test_var)}")
        raise AssertionError


if __name__ == "__main__":
    # Tests for checking the form of the terms.
    TestUnitForm()
    TestPolynomialForm()
    TestExponentialForm()
    TestCosineForm()
    
    # # Tests for converting the GS to Time domain (Laplace-Borel transform).
    TestUnit()
    TestPolynomial()
    TestExponential()
    print("Cosine has been commented out")
    # TestCosine()