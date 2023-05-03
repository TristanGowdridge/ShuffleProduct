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
from sympy import Symbol
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
        top_row[top_row == 2] = 1 # Used in imp response for term of len 1.
    
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
            

    
def impulse(scheme):
    """
    Defined in "Functional Analysis of Nonlinear Circuits- a Generating Power
    Series Approach".
    """
    imp = []
    for coeff, term in scheme:
        numerator = term[0]
        count = 0
        for val in numerator:
            count += np.real(val)
            if val == 0 and count != 0:
                break
        else:
            if len(numerator) != 1:
                new_term = shfl.GeneratingSeries(term[:, :(-int(count))])
                imp.append((coeff * (1 / factorial(count)), new_term))
            else:
                term[0, 0] = 2
                imp.append((coeff, term))
    return imp

          
def matlab_partfrac(scheme, filename="terms", delete_files=True):
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
    only_one_symbol = (len(symbol) == 1)

    return symbol[0], only_one_symbol


def get_exponent(term):
    if isinstance(term, (SympySymbol, SympyPow)):
        return sym.log(term).expand(force=True).as_coeff_Mul()
    else:
        raise TypeError("Needs to be a sympy.core.power.Pow object.")
 
 
def _check_form(term):
    """
    Checks against each of the required forms, if the correct form is
    identified the inverse laplace borel transform of it is returned. If the 
    term fits no form, an error it raised.
    """
    for form in [lb_unit, lb_polynomial, lb_exponential, lb_cosine]:
        term_ts = form(term)
        if term_ts:
            return term_ts
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
                term_ts = _check_form(term1)
                ts += term_ts  
        else:
            term_ts = _check_form(term)
            ts += term_ts    

    return ts


def is_unit_form(term):
    """
    Determines whether the term is of unit form, both sympy Float and 
    sympy Integer inherit from sympy Number.
    """
    return isinstance(term, (SympyNumber, int, float))


def lb_polynomial(term):
    """
    Determines whether the term is of polynomial form. There are two potential
    cases for the polynomial form, a * x ** b (type Mul) or x ** b (type Pow),
    so we need to check for both of these cases. If the term is of the first
    case, we reduce it to the second case and then checks are the same after.
    We check the log of the term, if it as a number coeffcient and that the
    other part is the log of the symbol, if so, this can only be of polynomial
    form.
    
    
    There is similar functionality across these function, these definitely 
    can be refactored.
    """
    t = Symbol('t')

    was_mul = False # Check for the type was mul or pow.
    cond1 = False   # Check if 'a' is number in the form a * x ** b.
    cond2 = False   # Check if the exponent is an integer.
    cond3 = False   # Check if the remaining term is log(x).
    
    if is_unit_form(term): # Fail-fast for unit form.
        return False
    
    symbol, only_one_symbol = get_symbol(term)
    if not only_one_symbol: # Test to see if there is only one variable.
        return False
        
    if isinstance(term, SympySymbol): # If term is just a symbol.
        return t    
    
    if isinstance(term, SympyMul):
        was_mul = True
        coeff1, term = term.as_coeff_Mul()
        cond1 = is_unit_form(coeff1)
        
    if isinstance(term, (SympyPow, SympySymbol)):
        if not was_mul:
            coeff1 = 1
            cond1 = True
        exponent, log_term = get_exponent(term)
        cond2 = (float(exponent) == int(exponent) and is_unit_form(exponent))
        cond3 = (log_term == sym.log(symbol)) 
        
    if (cond1 and cond2 and cond3):
        a = coeff1
        b = exponent
        
        to_return = (a * t ** b / sym.factorial(b))
        to_return = sym.simplify(to_return)
        
        return to_return


def lb_exponential(term):
    """
    Determines whether the term is of exponential form.
    
    There is similar functionality across these function, these definitely 
    can be refactored.
    """
    t = Symbol('t')

    if is_unit_form(term): # Fail-fast for unit form.
        return False

    symbol, only_one_symbol = get_symbol(term)
    if not only_one_symbol: # Test to see if there is only one variable.
        return False

    was_mul1 = False
    
    # At this point term can be either SympyMul or SympyPow. If SympyPow,
    # reduce to SympyMul.
    if isinstance(term, SympyMul):
        was_mul1 = True
        coeff1, term = term.as_coeff_Mul()
    
    if isinstance(term, SympyPow):
        if not was_mul1:
            coeff1 = 1
        exponent, log_term = get_exponent(term)
        # Check if the exponent is a negative integer. 
        good_exponent = (float(exponent) == int(exponent)) and exponent < 0
        if not (good_exponent and is_unit_form(exponent)):
            return False
    else:
        return False

    term = sympyexp(log_term)
    
    # Check if the denominator has type SympyAdd.
    if not isinstance(term, SympyAdd):
        return False
    
    unit, term = term.as_coeff_Add()
    if not is_unit_form(unit): # Check if the unit is unit form.
        return False

    # Check if the symbol side is SympyMul, if so reduce to a SympySymbol.
    if isinstance(term, SympyMul):
        coeff2, dummy_term = term.as_coeff_Mul()
    elif term == symbol:
        dummy_term = term
    else:
        return False
    if not dummy_term == symbol:
        return False
       
    coeff2 = term.subs(symbol, 1)
    
    # Handling the conversion of the laplace-borel transform.
    a = -coeff2
    n = -exponent

    if unit != 1:
        a /= unit
        coeff1 /= unit ** n
        unit = 1
    ts = 0
    
    # print(f"a = {a}, n = {n}, coeff1 = {coeff1}")
    for i in range(n):
        ts += (comb(n-1, i) / sym.factorial(i)) * (a * t) ** i
    ts *= (coeff1 * sympyexp(a * t))

    return ts
    

def lb_cosine(term):
    """
    Tests whether the term is of cosine form.
    
    There is similar functionality across these function, these definitely 
    can be refactored.
    """
    t = Symbol('t')

    if is_unit_form(term): # Fail-fast for unit form.
        return False
    
    symbol, only_one_symbol = get_symbol(term)
    if not only_one_symbol: # Test to see if there is only one variable.
        return False
    
    was_mul = False
    # At this point term can be either SympyMul or SympyPow. If SympyPow,
    # reduce to SympyMul.
    if isinstance(term, SympyMul):
        was_mul = True
        coeff1, term = term.as_coeff_Mul()
        
    if isinstance(term, SympyPow):
        if not was_mul:
            coeff1 = 1
        exponent, log_term = get_exponent(term)
        # Check if the exponent is a negative integer. 
        good_exponent = (float(exponent) == int(exponent)) and exponent == -1
        if not (good_exponent and is_unit_form(exponent)):
            return False
    else:
        return False
    
    term = sympyexp(log_term)
    
    # Check if the denominator is of type SympyAdd.
    if isinstance(term, SympyAdd):
        unit, term = term.as_coeff_Add()
        if not is_unit_form(unit):
            return False
    else: 
        return False
    
    # Determine the coefficient of the Symbol.
    if isinstance(term, SympyMul):
        coeff2, term = term.as_coeff_Mul()
    elif isinstance(term, SympyPow):
        coeff2 = term.subs(symbol, 1)
    else:
        return False
    
    # Ensure that the signs are the same.
    if np.sign(unit) != np.sign(coeff2):
        return False
    
    # Check for an exponent of 2 on the symbol.
    if isinstance(term, SympyPow):
        exponent, log_term = get_exponent(term)
        if not exponent == 2:
            return False
        if not (log_term == sym.log(symbol)):
            return False
    else:
        return False
    
    # coeff1, coeff2, unit
    if unit != 1:
        coeff1 /= unit
        coeff2 /= unit
        unit = 1

    return coeff1 * sym.cos(sym.sqrt(coeff2) * t)


def lb_unit(term):
    """
    
    """
    correct_form = is_unit_form(term)
    
    if correct_form:
        return term
    else:
        return False


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