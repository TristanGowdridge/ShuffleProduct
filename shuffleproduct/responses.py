# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:57:45 2023

@author: trist
"""
import os
from math import factorial, prod, comb
import subprocess

import numpy as np
import sympy as sym
from sympy import Symbol, Wild, symbols
from sympy.core.numbers import Number as SympyNumber
from sympy.core.mul import Mul as SympyMul
from sympy.core.power import Pow as SympyPow
from sympy.core.symbol import Symbol as SympySymbol
from sympy.core.add import Add as SympyAdd
from sympy.functions.elementary.exponential import exp as sympyexp

import shuffle as shfl


def array_to_fraction(terms):
    """
    This converts from the array form of the generating series to the fraction
    form, so they can be sovled by sympy's partial fraction calculator.
    """
    output_list = []
    x0 = Symbol("x0")
    for term in terms:
        top_row = term[1][0].astype(dtype="O")
        bottom_row = term[1][1]
        top_row[top_row == 0] = x0
        top_row[top_row == 1] = Symbol("x1")
        
        # top_row[top_row == 2] = 1 # Used in imp response for term of len 1.
    
        numerator = prod(top_row)
        denominator = prod([(1 + i*x0) for i in bottom_row])
        
        output_list.append(term[0] * numerator / denominator)
    
    return output_list


def calculate_response(gs, input_type, *args, **kwargs):
    """
    This calculates the response for some generating series, given the type
    of input.
    """
    
    if input_type == "step":
        raise NotImplementedError
    
    elif input_type == "sine":
        raise NotImplementedError
    
    elif input_type == "exponential":
        raise NotImplementedError
            
    elif input_type == "polynomial":
        raise NotImplementedError
        
    elif input_type in ("GWN", "gwn"):
        gs_response = gwn_response(gs, *args, **kwargs)
    
    elif input_type == "impulse":
        gs_response = impulse(gs, *args, **kwargs)
        
    else:
        raise NotImplementedError(f"Unknown response f{input_type}.")
        
    partfrac_sum = matlab_partfrac(gs_response)

    ts = inverse_lb(partfrac_sum)
    
    return ts
 
    
def step_input(scheme, amplitude=1):
    """
    Converts the generating series to have a step input.
    """
    step_gs = []
    for term in array_to_fraction(scheme):
        step_gs.append(term.subs({Symbol("x1") : amplitude * Symbol("x0")}))
        
    return step_gs      


def gwn_response(scheme, sigma=1):
    """
    Calculates the GWN response given an input generating series. 
    """
    gwn = []
    
    # This first section of the for-loop is getting the terms of the desired
    # form, as many will be reduced to zero.
    for coeff, term in scheme:
        numerator = term[0]
        count_1s = 0
        for val in numerator:
            if val == 0:
                if (count_1s % 2 == 0):
                    continue
                else:
                    count_1s = 0
                    break
            if val == 1:
                count_1s += 1
            
        else:
            if (count_1s % 2 != 0):
                continue
            
            count_1s = 0
            new_term = np.zeros((2, 1))
            for i, val in enumerate(numerator):
                if val == 0:
                    temp = term[:, i].reshape(2, 1)
                    new_term = np.hstack([new_term, temp])
                    continue
                
                elif val == 1 and count_1s == 1:
                    count_1s = 0
                    coeff *= ((sigma ** 2) / 2)
                    temp = term[:, i-1].reshape(2, 1)
                    temp[0, 0] = 0
                    new_term = np.hstack([new_term, temp])
                    continue
                
                elif val == 1 and count_1s == 0:
                    count_1s = 1
                    continue
                
                else:
                    raise ValueError("Code should not be here.")
            gwn.append((coeff, new_term))
    
    return array_to_fraction(gwn)
            

    
def impulse(scheme, amplitude=1):
    """
    Defined in "Functional Analysis of Nonlinear Circuits- a Generating Power
    Series Approach".
    """
    imp = []
    for coeff, term in scheme:
        x0_storage = []
        for i, x_i in enumerate(term[0, :]):
            if x_i == 1:
                if all(term[0, i:] == 1):
                    n = int(np.real(np.sum(term[0, :])))
                    frac = (coeff/factorial(int(n))) / (1 + term[1, i]*Symbol("x0"))
                    if x0_storage:
                        for x0_term in x0_storage:
                            frac *= x0_term
                    imp.append(amplitude * frac)
                break
            elif x_i == 0:
                x0_storage.append(Symbol("x0") / (1 + term[1, i]*Symbol("x0")))
            else:
                raise ValueError("Unknown term in 0th row.")

    return imp
 
    
def deterministic_response(scheme, excitation):
    """
    
    """
    raise NotImplementedError


         
def matlab_partfrac(scheme, filename="terms", precision=None, delete_files=True):
    """
                        ****** VERY HACKY ******
    
    Writes to the fractional forms into a .txt file to be loaded into matlab
    because it's partial fractions function actually works. Then does the
    calculation in MATLAB, writes to another .txt file which is then loaded
    into Python.
    
    Because of how this adds the folder to the path, it needs to be ran inside
    a sub folder.
    """
    with open(f"{filename}_python.txt", 'w') as f:
        for term in scheme:
            if precision:
                term = term.evalf(precision)
            str_term = str(term).replace("**", '^').replace('I', 'i')+ "\n"
            f.write(str_term)
    run_str = "matlab -nosplash -nodesktop -wait -r"
    run_str += " \"addpath('../shuffleproduct/');" 
    run_str += f"partial_fractions('{filename}'); exit\""
    subprocess.run(run_str)
    
    x0 = Symbol("x0") # noqa. This will be eval'd.
    sum_of_partials = []
    with open(f"{filename}_MATLAB.txt") as file:
        while line := file.readline():
            # print(line)
            sum_of_partials.append(eval(line.rstrip()))
    
    if delete_files:
        os.remove(filename+"_MATLAB.txt")
        os.remove(filename+"_python.txt")
    
    return sum_of_partials


    

    
    
# =============================================================================
# Converting to the time domain.
# =============================================================================
def get_symbol(term):
    symbol = list(term.free_symbols)
    
    if (len(symbol) != 1):
        raise ValueError("The should only be one symbol in the term.")

    return symbol[0]


def convert_term(term):
    """
    Checks against each of the required forms, if the correct form is
    identified the inverse laplace borel transform of it is returned. If the 
    term fits no form, an error it raised.
    """
    func_pairs = (
        (is_unit_form,        lb_unit),
        (is_polynomial_form,  lb_polynomial),
        (is_exponential_form, lb_exponential),
        (is_cosine_form,      lb_cos),
    )
    
    for (form_test, converter) in func_pairs:
        if form_test(term):
            return converter(term)
    else:
        raise TypeError(f"Term is of an unknown form.\n\n{term}")
    
    
def inverse_lb(sum_of_fractions):
    """
    Converts the partial fractions into the time domain.
    
    There are two cases that need to be analysed here. If the term is type
    SympyAdd, we need to get each term in the addition and analyse these
    separately. Otherwise, the term is passed in as usual.
    """
    ts = 0
    for term in sum_of_fractions:
        if isinstance(term, SympyAdd):
            for term1 in term.make_args(term):
                ts += convert_term(term1)
        else:
            term_ts = convert_term(term)
            ts += term_ts
        
        # value = ts.subs({Symbol('t'): 5})
        # if abs(value) > 10:
        #     print(f"term is greater with a value of {value}\n\n", term, end="\n\n\n")
    return ts


def is_unit_form(term):
    """
    Determines whether the term is of unit form, both sympy Float and 
    sympy Integer inherit from sympy Number.
    """
    return isinstance(term, (SympyNumber, int, float))


def lb_unit(term):
    """
    
    """
    return term


def is_polynomial_form(term):
    """
    Tests whether the term is of polynomial form.
    """
    if is_unit_form(term):
        return False
    
    x0 = get_symbol(term)
    a = Wild("a", exclude=[x0, 0])
    n = Wild("n")
        
    polynomial_form = a * x0 ** n
    
    match = term.match(polynomial_form)
    if match:
        if not match[n].is_integer:
            return False
    
    return match


def lb_polynomial(term):
    """
    a * x ** n
    """
    x0 = get_symbol(term)
    a = Wild("a", exclude=[x0, 0])
    n = Wild("n", exclude=[0])
        
    polynomial_form = a * x0 ** n
    
    match = term.match(polynomial_form)
    n = match[n]
    a = match[a]
    
    t = Symbol("t")
    
    return (a / sym.factorial(n)) * t ** n


def is_exponential_form(term):
    """
    Tests whether the term is of exponential form.
    """
    if is_unit_form(term):
        return False
    
    x0 = get_symbol(term)

    b = Wild("b", exclude=[x0])
    c = Wild("c", exclude=[x0])
    d = Wild("d", exclude=[x0])
    n = Wild("n")
        
    a, denom = term.as_numer_denom()
    
    if a.free_symbols:
        return False
    
    denom_form = b * (c + d*x0) ** n
    match = denom.match(denom_form)
    
    if match:
        if not match[n].is_integer:
            return False       
    
    return match


def lb_exponential(term):
    """
    a / b * (c + d*x0) ** -n
    """
    x0 = get_symbol(term)

    b = Wild("b", exclude=[x0])
    c = Wild("c", exclude=[x0])
    d = Wild("d", exclude=[x0])
    n = Wild("n")
    
    a, denom = term.as_numer_denom()
    # print(f"a is {a}")  
    denom_form = b * (c + d*x0) ** n
    match = denom.match(denom_form)

    b = match[b]
    c = match[c]
    d = -match[d]
    n = match[n]
    
    t = Symbol("t")
    
    coeff1 = a / (b * c **n)
    coeff2 = d / c
    # print(coeff2)
    # if coeff2 > 0:
    #     print(f"coeff2 is positive {term}")

    ts = 0
    for i in range(n):
        ts += (comb(n-1, i) / sym.factorial(i)) * (coeff2 * t) ** i
    ts *= (coeff1 * sympyexp(coeff2 * t))

    return ts


def is_cosine_form(term):
    """
    Tests whether the term is of cosine form.
    """
    if is_unit_form(term):
        return False
    
    x0 = get_symbol(term)
    a = Wild("a", exclude=[x0, 0])
    b = Wild("b", exclude=[x0, 0])
    c = Wild("c", exclude=[x0, 0])
    n = Wild("n", exclude=[x0, 0])
    
    cosine_form = a * (b + c*x0**2) ** -n
    
    match = term.match(cosine_form)
    
    if match:
        if not match[n].is_integer:
            return False
    
    return match


def lb_cosine(term):
    """
    a * (b + c*x**2) ** -1
    """
    x0 = get_symbol(term)
    a = Wild("a", exclude=[x0, 0])
    b = Wild("b", exclude=[x0, 0])
    c = Wild("c", exclude=[x0, 0])
    n = Wild("n", exclude=[x0, 0])
    
    cosine_form = a * (b + c*x0**2) ** -n
    
    match = term.match(cosine_form)

    a = match[a]
    b = match[b]
    c = match[c]
    n = match[n]
    
    t = Symbol("t")
    
    coeff1 = a / b**n
    coeff2 = sym.sqrt(c / b)
    
    return coeff1 * sym.cos(coeff2 * t)
    

def lb_sin(term):
    """
    Given in Unal's paper in the papers repo.
    
    LB(sin(wt)) = iwx0 / (1 + w**2 x0**2)
    """
    pass


def lb_cos(term):
    """
    Given in Unal's paper in the papers repo.
    
    LB(cos(wt)) = 1 / (1 + w**2 x0**2)
    """
    pass


def lb_sinh(term):
    """
    Given in Unal's paper in the papers repo.
    
    LB(sinh(wt)) = wx0 / (1 - w**2 x0**2)
    """
    pass


def lb_cosh(term):
    """
    Given in Unal's paper in the papers repo.
    
    LB(cosh(wt)) = 1 / (1 - w**2 x0**2)
    """
    pass

def time_function(time_domain):
    """
    Get a function of the response wth respect to time.
    
    """
    return sym.lambdify(Symbol('t'), time_domain)