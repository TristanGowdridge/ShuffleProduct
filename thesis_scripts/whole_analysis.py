# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:12:17 2023

@author: trist
"""

import os
import sys
from collections import defaultdict
from operator import itemgetter
from itertools import product
import subprocess

import numpy as np
from sympy import symbols, lambdify, factorial

from params import A, m, c, k1, k2, k3, plot, t
from vci_quad_cube import y1 as y1_volt
from vci_quad_cube import y2 as y2_volt
from vci_quad_cube import y3 as y3_volt

sys.path.insert(0, os.path.dirname(os.getcwd()) + r"\shuffleproduct")
import shufflesym as shfl
import responses as rsps
from generating_series import GeneratingSeriesSym as GS


_k2, _k3, _x0, _x1, _a1, _a2, _A = symbols("k2 k3 x0 x1 a1 a2 A")

_g0 = GS([
    [  1, _x0, _x1],
    [_a1, _a2,   0]
])

_mult_quad = GS([
    [-_k2, _x0, _x0],
    [ _a1, _a2,   0]
])
_mult_cube = GS([
    [-_k3, _x0, _x0],
    [ _a1, _a2,  0]
])


def remove_nonimp(terms):
    store = []
    for term in terms:
        has_been_x1 = False
        for val in term.words:
            if val == _x1:
                has_been_x1 = True
                continue
            elif val == _x0 and (not has_been_x1):
                continue
            else:
                break
        else:
            store.append(term)
            
    return store


def iterate_quad_cubic(iter_depth):
    """
    A very hastily written iterative expansion of a SDOF oscillator with
    quadratic and cubic nonlinearities.
    
    This function is reliant on global variables, be careful! It also isn't
    generalisable at all, but it's what we need for our specific example.
    """
    term_storage = defaultdict(list)
    term_storage[0].append(_g0)
    
    term_storage_quad = defaultdict(list)
    term_storage_cube = defaultdict(list)
    
    for depth in range(iter_depth):
        for part in shfl.partitions(depth, 2):
            terms = itemgetter(*part)(term_storage)
            for in_perm in product(*terms):
                term_storage_quad[depth+1].extend(shfl.nShuffles(*in_perm))
        term_storage_quad[depth+1] = shfl.collect(term_storage_quad[depth+1])
        term_storage_quad[depth+1] = remove_nonimp(term_storage_quad[depth+1])
        
        for part in shfl.partitions(depth, 3):
            terms = itemgetter(*part)(term_storage)
            for in_perm in product(*terms):
                term_storage_cube[depth+1].extend(shfl.nShuffles(*in_perm))
        term_storage_cube[depth+1] = shfl.collect(term_storage_cube[depth+1])
        term_storage_cube[depth+1] = remove_nonimp(term_storage_cube[depth+1])
    
        # After the shuffles for this iteration's depth have been caluclated,
        # prepend the multiplier to each term.
        for gs_term in term_storage_quad[depth+1]:
            gs_term.prepend_multiplier(_mult_quad)
        term_storage[depth+1].extend(term_storage_quad[depth+1])
        
        for gs_term in term_storage_cube[depth+1]:
            gs_term.prepend_multiplier(_mult_cube)
        term_storage[depth+1].extend(term_storage_cube[depth+1])
        
        term_storage[depth+1] = shfl.collect(term_storage[depth+1])
    
    return shfl.handle_output_type(term_storage, tuple)


def impulsehere(terms, A, iter_depth):
    imp = defaultdict(list)
    x1_sym, x0_sym = symbols("x1 x0")
    for coeff, term in scheme:
        x0_storage = []
        for i, x_i in enumerate(term[0, :]):
            if x_i == x1_sym:
                if all(np.equal(term[0, i:], x1_sym)):
                    n = term.shape[1] - i
                    frac = (
                        (coeff / factorial(int(n))) / (1 + term[1, i]*x0_sym)
                    )
                    if x0_storage:
                        for x0_term in x0_storage:
                            frac *= x0_term
                    imp[n].append(A**n * frac)
                break
            elif x_i == x0_sym:
                x0_storage.append(x0_sym / (1 + term[1, i]*x0_sym))
            else:
                raise ValueError("Unknown term in 0th row.")

    return {key: imp[key+1] for key in range(iter_depth+1)}
   

def matlab_partfrac_here(scheme, filename="terms", delete_files=True):
    """
    Calculates the matlab partial fractions for multiple calls.
    """
    for key, vals in scheme.items():
        with open(f"{filename}{key}_python.txt", 'w') as f:
            for term in vals:
                str_term = str(term).replace("**", '^').replace('I', 'i')+"\n"
                f.write(str_term)
    
    run_str = "matlab -nosplash -nodesktop -wait -r"
    run_str += " \"addpath('../shuffleproduct/');"
    run_str += f"pf_dict('{filename}', {list(scheme.keys())}); exit\""
    subprocess.run(run_str)
    
    x0, x1, a1, a2, k1, k2, k3, a, b, b1, b2, A = symbols(
        "x0 x1 a1 a2 k1 k2 k3 a b b1 b2 A"
    )
    
    sum_of_partials = defaultdict(list)
    for key in scheme:
        with open(f"{filename}{key}_MATLAB.txt") as file:
            while line := file.readline():
                sum_of_partials[key].append(eval(line.rstrip()))
    
    if delete_files:
        for key in scheme:
            os.remove(f"{filename}{key}_python.txt")
            os.remove(f"{filename}{key}_MATLAB.txt")

    return dict(sum_of_partials)


def convert_gs_to_time(terms, amp, iter_depth):
    """
    Takes the list of generating series and returns the associated time
    function.
    """
    g = impulsehere(terms, amp, iter_depth)
    
    gpf = matlab_partfrac_here(g)
    
    y = []
    for pf in gpf.values():
        y.append(rsps.inverse_lb(pf))
    
    return y


iter_depth = 2
scheme = iterate_quad_cubic(iter_depth)

y_gs = convert_gs_to_time(scheme, _A, iter_depth)

a1, a2 = shfl.sdof_roots(m, c, k1)

vals = {
    _A: A,
    _a1: -a1,
    _a2: -a2,
    _k2: k2,
    _k3: k3,
}

y1_g = lambdify(symbols('t'), y_gs[0].subs(vals))(t)
y2_g = lambdify(symbols('t'), y_gs[1].subs(vals))(t)
y3_g = lambdify(symbols('t'), y_gs[2].subs(vals))(t)
# y4_g = lambdify(symbols('t'), y_gs[3].subs(vals))(t)

_figax = plot(y1_volt, None, "$y_1^v$")
_figax = plot(y2_volt, _figax, "$y_2^v$")
_figax = plot(y3_volt, _figax, "$y_3^v$")

_figax = plot(y1_g, _figax, "$y^g_1$", linestyle="--")
_figax = plot(y2_g, _figax, "$y^g_2$", linestyle="--")
_figax = plot(y3_g, _figax, "$y^g_3$", linestyle="--")
# _figax = plot(y4_g, _figax, "$y^g_4$", linestyle="--")
# _figax = plot(y1_g + y2_g + y3_g, _figax, "$y gen 3$", linestyle="--")
# _figax = plot(y1_g + y2_g + y3_g + y4_g, _figax, "$y gen 4$", linestyle="--")

_fig, _ax = _figax
_ax.set_title(
    f"Comparison of Duffing's equation solutions with Dirac delta {A}"
)
