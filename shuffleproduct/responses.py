# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:57:45 2023

@author: trist
"""
import os
from math import factorial, comb, prod
import subprocess
from collections import defaultdict
from itertools import product
from operator import itemgetter

import numpy as np
import sympy as sym
from sympy import symbols, Wild
from sympy.core.numbers import Number as SympyNumber
from sympy.core.add import Add as SympyAdd
from sympy.core.mul import Mul as SympyMul
from sympy.functions.elementary.exponential import exp as sympyexp

import shuffle as shfl
from generating_series import GeneratingSeries as GS


def to_fraction(terms):
    """
    This converts from the array form of the generating series to the
    fraction form, so they can be sovled by a partial fraction calculator.
    
    The arive herer in "tuple form"
    """
    x0, x1 = symbols("x0 x1")
    output_list = []
    for term in terms:
        top_row = term[1][0].astype(dtype="O")
        bottom_row = term[1][1]
        top_row[np.equal(top_row, 0)] = x0
        top_row[np.equal(top_row, 1)] = x1
            
        numerator = prod(top_row)
        denominator = prod([(1 + i*x0) for i in bottom_row])
        
        output_list.append(term[0] * numerator / denominator)
    
    return output_list


def step_input(scheme, amplitude=1):
    """
    Converts the generating series to have a step input.
    """

    step_gs = []
    for term in to_fraction(scheme):
        step_gs.append(term.subs({symbols("x1"): amplitude * symbols("x0")}))
        
    return step_gs


def gwn_response(scheme, sigma=1):
    """
    Calculates the GWN response given an input generating series.
    """
    raise NotImplementedError("This hasnt been generalised to gs type")
    
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
    
    return to_fraction(gwn)
            

def impulse(scheme, amplitude=1):
    """
    Defined in "Functional Analysis of Nonlinear Circuits- a Generating Power
    Series Approach".
    
    This is used if the generating series have already been expanded. For
    efficiency use impulse_from_iter().
    """
    if scheme[0][1].dtype == object:
        x0, x1 = symbols("x0 x1")
    else:
        x0, x1, = 0, 1
    
    imp = []
    for coeff, term in scheme:
        x0_storage = []
        for i, x_i in enumerate(term[0, :]):
            if x_i == x1:
                if all(np.equal(term[0, i:], x1)):
                    n = term.shape[1] - i
                    frac = (
                        coeff/factorial(int(n)) / (1+term[1, i]*symbols("x0"))
                    )
                    if x0_storage:
                        for x0_term in x0_storage:
                            frac *= x0_term
                    imp.append(amplitude**n * frac)
                break
            elif x_i == x0:
                x0_storage.append(symbols("x0") / (1+term[1, i]*symbols("x0")))
            else:
                raise ValueError("Unknown term in 0th row.")

    return imp


def impulse_from_iter(
        g0, multipliers, n_shuffles, iter_depth=2, amplitude=1):
    """
    The idea here centers around the fact that most of the term in the impulse
    response are thrown away. So when iterating the generating series, the
    terms that will definitely not result in any terms later on can be thrown
    away early, therefore we no longer have to expand these terms, this results
    in a efficiency gains when comparing to expanding the generating series
    and then applying the impulse response.
    """
    x0, x1 = g0.get_words()
    is_npy = isinstance(g0, np.ndarray)
    
    multipliers = shfl.wrap_term(multipliers)
    g0 = shfl.wrap_term(g0)
    
    term_storage = defaultdict(list)
    term_storage[0].extend(g0)
    
    for depth in range(iter_depth):
        for part in shfl.partitions(depth, n_shuffles):
            # Cartesian product of all the inputs, instead of nested for-loop.
            terms = itemgetter(*part)(term_storage)
            for in_perm in product(*terms):
                term_storage[depth + 1].extend(shfl.nShuffles(*in_perm))
            term_storage[depth + 1] = g0[0].collect(term_storage[depth + 1])
        
        # After the shuffles for this iteration's depth have been caluclated,
        # prepend the multiplier to each term.
        next_terms = []
        for gs_term in term_storage[depth + 1]:
            been_1 = False
            for x in gs_term.get_words():
                if x == x1:
                    been_1 = True
                
                elif been_1 and x == x0:
                    break
            else:
                for multiplier in multipliers:
                    if is_npy:
                        temp = gs_term.prepend_multiplier(multiplier)
                        next_terms.append(temp)
                        term_storage[depth + 1] = next_terms
                    else:
                        gs_term.prepend_multiplier(multiplier)

        term_storage[depth + 1] = next_terms
    
    tuple_form = g0[0].handle_output_type(term_storage, tuple)
    
    return impulse(tuple_form, amplitude)

      
def matlab_partfrac(
        scheme, filename="terms", precision=0, delete_files=True):
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
            str_term = str(term).replace("**", '^').replace('I', 'i') + "\n"
            f.write(str_term)
    run_str = "matlab -nosplash -nodesktop -wait -r"
    run_str += " \"addpath('../shuffleproduct/');"
    run_str += f"partial_fractions('{filename}', {precision}); exit\""
    subprocess.run(run_str)
    
    x0, x1, a1, a2, k1, k2, k3, a, b, b1, b2, A = sym.symbols(
        "x0 x1 a1 a2 k1 k2 k3 a b b1 b2 A"
    )
    
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
def convert_term(term):
    """
    Checks against each of the required forms, if the correct form is
    identified the inverse laplace borel transform of it is returned. If the
    term fits no form, an error it raised.
    """
    func_pairs = (
        (is_exponential_form, lb_exponential),
        (is_unit_form,        lb_unit),
        (is_polynomial_form,  lb_polynomial),
        (is_cosine_form,      lb_cosine),
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
    ts = []
    for term in sum_of_fractions:
        if isinstance(term, SympyAdd):
            for term1 in term.make_args(term):
                ts.append(convert_term(term1))
        else:
            term_ts = convert_term(term)
            ts.append(term_ts)
        
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
    
    x0 = symbols("x0")
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
    x0, t = symbols("x0 t")
    a = Wild("a", exclude=[x0, 0])
    n = Wild("n", exclude=[0])
        
    polynomial_form = a * x0 ** n
    
    match = term.match(polynomial_form)
    n = match[n]
    a = match[a]
    
    return (a / sym.factorial(n)) * t ** n


def is_exponential_form(term):
    """
    Tests whether the term is of exponential form.
    
    This method of reducing all the denominator coefficients and only matching
    the crucial part is much faster.
    """
    if is_unit_form(term):
        return False
    
    x0 = symbols("x0")

    b = Wild("b", exclude=[x0])
    c = Wild("c", exclude=[x0])

    n = Wild("n")
        
    a, denom = term.as_numer_denom()
    
    if x0 in a.free_symbols:
        return False
    
    den_args = SympyMul.make_args(denom)
    den_coeffs = 1
    crucial = None
    for den_arg in den_args:
        if x0 in den_arg.free_symbols:
            if not crucial:
                crucial = den_arg
            else:
                return False
        else:
            den_coeffs *= den_arg
    
    try:
        if x0 in den_coeffs.free_symbols:
            return False
    except AttributeError:
        pass
        
    # Match the crucial part of the denominator.
    denom_form = (b + c * x0) ** n
    match = crucial.match(denom_form)
        
    if match:
        if not match[n].is_integer:
            print(
                "responses.is_exponential_form:",
                "failing because n is not an integer"
            )
            return False
    
    return bool(match)


def lb_exponential(term):
    """
    a / (den_coeffs * (b + c*x0) ** n)
    """
    x0, t = symbols("x0 t")

    b = Wild("b", exclude=[x0])
    c = Wild("c", exclude=[x0])
    
    n = Wild("n")
    
    a, denom = term.as_numer_denom()
    
    den_args = SympyMul.make_args(denom)
    den_coeffs = 1
    for den_arg in den_args:
        if x0 in den_arg.free_symbols:
            crucial = den_arg
        else:
            den_coeffs *= den_arg

    denom_form = (b + c * x0) ** n
    match = crucial.match(denom_form)

    b = match[b]
    c = -match[c]
    n = match[n]
    
    coeff1 = a / (b ** n * den_coeffs)
    coeff2 = c / b
    
    ts = 0
    for i in range(n):
        ts += (comb(n-1, i) / sym.factorial(i)) * (coeff2 * t) ** i
    ts *= (coeff1 * sympyexp(coeff2 * t))

    return ts


def is_cosine_form(term):
    """
    Tests whether the term is of cosine form.
    """
    print("COSINE IS SLOW")
    if is_unit_form(term):
        return False
    
    x0 = symbols("x0")
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
    print("COSINE IS SLOW")
    x0, t = symbols("x0 t")
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
    
    coeff1 = a / b**n
    coeff2 = sym.sqrt(c / b)
    
    return coeff1 * sym.cos(coeff2 * t)


def time_function(time_domain):
    """
    Get a function of the response wth respect to time.
    
    """
    return sym.lambdify(symbols('t'), time_domain)


if __name__ == "__main__":
    x0, x1, a1, a2, k1, k2, k3, A = symbols(
        "x0 x1 a1 a2 k1 k2 k3 A"
    )
    import time
    
    test1 = (36*A**6*a1**7*k2*k3**2 - 450*A**6*a1**6*a2*k2*k3**2 + 504*A**6*a1**5*a2**2*k2*k3**2 + 1998*A**6*a1**4*a2**3*k2*k3**2 - 1728*A**6*a1**3*a2**4*k2*k3**2 - 2196*A**6*a1**2*a2**5*k2*k3**2 + 324*A**6*a1*a2**6*k2*k3**2 + 216*A**6*a2**7*k2*k3**2)/((2*a1 + 2*a2)*(2*x0*(2*a1 + 2*a2) + 2)*(a1**4 - 2*a1**2*a2**2 + a2**4)*(4*a1**12*a2**2 - 28*a1**11*a2**3 + 25*a1**10*a2**4 + 208*a1**9*a2**5 - 430*a1**8*a2**6 - 308*a1**7*a2**7 + 1225*a1**6*a2**8 - 352*a1**5*a2**9 - 680*a1**4*a2**10 + 192*a1**3*a2**11 + 144*a1**2*a2**12))
    test2 = (36*A**6*a1**8*k2*k3**2 - 558*A**6*a1**7*a2*k2*k3**2 + 270*A**6*a1**6*a2**2*k2*k3**2 + 3654*A**6*a1**5*a2**3*k2*k3**2 - 378*A**6*a1**4*a2**4*k2*k3**2 - 6228*A**6*a1**3*a2**5*k2*k3**2 - 2736*A**6*a1**2*a2**6*k2*k3**2 + 540*A**6*a1*a2**7*k2*k3**2 + 216*A**6*a2**8*k2*k3**2)/((a1 + a2)*(2*a1 + 2*a2)*(2*x0*(2*a1 + 2*a2) + 2)*(-a1**4 - 2*a1**3*a2 + 2*a1*a2**3 + a2**4)*(4*a1**12*a2**2 - 28*a1**11*a2**3 + 25*a1**10*a2**4 + 208*a1**9*a2**5 - 430*a1**8*a2**6 - 308*a1**7*a2**7 + 1225*a1**6*a2**8 - 352*a1**5*a2**9 - 680*a1**4*a2**10 + 192*a1**3*a2**11 + 144*a1**2*a2**12))
    
    t0 = time.perf_counter()
    print(convert_term(test2))
    print(f"{time.perf_counter() - t0:2f}s")