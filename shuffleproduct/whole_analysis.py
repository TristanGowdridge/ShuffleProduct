# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:12:17 2023

@author: trist
"""

import os
from collections import defaultdict
from operator import itemgetter
from itertools import product
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sympy import symbols, lambdify, factorial, apart

from examples.params import A, m, c, k1, k2, k3, t, iter_depth
from auxilliary_funcs import worker, plot
from vci_quad_cube import y1 as y1_volt
from vci_quad_cube import y2 as y2_volt
from vci_quad_cube import y3 as y3_volt

import shuffle as shfl
from generating_series import GeneratingSeries as GS


from sympy.core.add import Add as SympyAdd


def remove_nonimp(terms):
    """
    Removes all the terms that have an x0 after an x1.
    """
    if not terms:
        return []
    
    if not isinstance(terms[0], np.ndarray):
        x0, x1 = symbols("x0 x1")
    else:
        x0, x1, = 0, 1
    
    store = []
    for term in terms:
        has_been_x1 = False
        for val in term.get_numer():
            if val == x1:
                has_been_x1 = True
                continue
            elif val == x0 and (not has_been_x1):
                continue
            else:
                break
        else:
            store.append(term)
            
    return store


def check_n_x1s_less_than_iter_depth(terms, iter_depth):
    """
    
    """
    count = 0
    for term in terms:
        count += term.n_excites
    
    return count <= (iter_depth + 1)


def iterate_quad_cubic(g0, mults, iter_depth):
    """
    A very hastily written iterative expansion of a SDOF oscillator with
    quadratic and cubic nonlinearities.
    
    This function is reliant on global variables, be careful! It also isn't
    generalisable at all, but it's what we need for our specific example.
    
    Should write this so the function takes in (m, c, k1) and then determines
    the number of nonlinearities by the size of the list passed in.
    """
    mult_quad, mult_cube = mults
    
    is_npy = isinstance(g0, np.ndarray)
    
    term_storage = defaultdict(list)
    term_storage[0].append(g0)
    
    term_storage_quad = defaultdict(list)
    term_storage_cube = defaultdict(list)
    
    for depth in range(iter_depth):
        for part in shfl.partitions(depth, 2):
            terms = itemgetter(*part)(term_storage)
            for in_perm in product(*terms):
                if check_n_x1s_less_than_iter_depth(in_perm, iter_depth):
                    term_storage_quad[depth+1].extend(shfl.nShuffles(*in_perm))
        
        term_storage_quad[depth+1] = g0.collect(term_storage_quad[depth+1])
        term_storage_quad[depth+1] = remove_nonimp(term_storage_quad[depth+1])
        
        for part in shfl.partitions(depth, 3):
            terms = itemgetter(*part)(term_storage)
            for in_perm in product(*terms):
                if check_n_x1s_less_than_iter_depth(in_perm, iter_depth):
                    term_storage_cube[depth+1].extend(shfl.nShuffles(*in_perm))
        term_storage_cube[depth+1] = g0.collect(term_storage_cube[depth+1])
        term_storage_cube[depth+1] = remove_nonimp(term_storage_cube[depth+1])
    
        # After the shuffles for this iteration's depth have been caluclated,
        # prepend the multiplier to each term.
        next_terms = []
        for gs_term in term_storage_quad[depth+1]:
            shfl.var_prepend(
                is_npy, gs_term, mult_quad, term_storage_quad,
                depth, next_terms
            )
        term_storage[depth+1].extend(term_storage_quad[depth+1])
        
        next_terms = []
        for gs_term in term_storage_cube[depth+1]:
            shfl.var_prepend(
                is_npy, gs_term, mult_cube, term_storage_cube,
                depth, next_terms
            )
        term_storage[depth+1].extend(term_storage_cube[depth+1])
        
        term_storage[depth+1] = g0.collect(term_storage[depth+1])
    
    return g0.handle_output_type(term_storage, tuple)


def impulsehere(terms, amp, iter_depth):
    """
    
    """
    imp = defaultdict(list)
    
    x0_sym = symbols("x0")
    if terms[0][1].dtype == object:
        x0, x1 = symbols("x0 x1")
    else:
        x0, x1 = 0, 1
    
    for coeff, term in terms:
        x0_storage = []
        for i, x_i in enumerate(term[0, :]):
            if x_i == x1:
                if all(np.equal(term[0, i:], x1)):
                    n = term.shape[1] - i
                    frac = (
                        (coeff / factorial(int(n))) / (1 - term[1, i]*x0_sym)
                    )
                    if x0_storage:
                        for x0_term in x0_storage:
                            frac *= x0_term
                    imp[n].append(amp**n * frac)
                break
            elif x_i == x0:
                x0_storage.append(x0_sym / (1 - term[1, i]*x0_sym))
            else:
                raise ValueError("Unknown term in 0th row.")

    return {key: imp[key+1] for key in range(iter_depth+1)}
   

def parallel_inverse_lb_and_save(pf):
    # In parallel compute the inverse Laplace-Borel transform
    result = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for r in executor.map(worker, pf):
            result.extend(r)
    return tuple(result)


def convert_gs_to_time(terms, amp, iter_depth):
    """
    Takes the list of generating series and returns the associated time
    function.
    """
    g = impulsehere(terms, amp, iter_depth)
    
    gs_pf = sympy_partfrac_here(g)
    
    time_terms = {}
    for key, pf in gs_pf.items():
        # Pickle the SymPy versions.
        # with open(f"quad_cube_y{key+1}_partfrac_symbolic.txt", "wb") as f_sym:
        #     pkl.dump(tuple(pf), f_sym)
        
        time_terms[key] = parallel_inverse_lb_and_save(pf)
        
        # with open(f"quad_cube_y{key+1}_volt_sym.txt", "wb") as f_sym:
        #     pkl.dump(list_serial, f_sym)
    
    return time_terms


def partial_parallel(term, x):
    """
    Decompose a single term into partial fractions and return the simplified terms.
    """
    pf_terms = apart(term.simplify(), x)  # Decompose the term
    separated = SympyAdd.make_args(pf_terms)    # Make individual fraction terms
    return [i.simplify() for i in separated]  # Return simplified fractions

# Helper function to be used in ProcessPoolExecutor
def partial_parallel_wrapper(term, x):
    """
    Wrapper for partial_parallel function to make it pickleable in multiprocessing.
    """
    return partial_parallel(term, x)

def sympy_partfrac_here(g):
    """
    Function to decompose terms into partial fractions using parallel processing.
    """
    x = symbols('x0')  # Define the symbolic variable
    storage_of_terms = {}
    cpu_cnt = os.cpu_count()  # Get the CPU count for parallel processing
    
    for index, gs in g.items():
        individual_storage = []

        if len(gs) < cpu_cnt:  # If the number of terms is less than CPU count, process sequentially
            for term in gs:
                individual_storage.extend(partial_parallel(term, x))
        else:
            # Use ProcessPoolExecutor for parallel processing
            with ProcessPoolExecutor(max_workers=cpu_cnt) as executor:
                # Pass the wrapper function to executor.map
                results = executor.map(partial_parallel_wrapper, gs, [x]*len(gs))  # Pass `x` for each term
                for r in results:
                    individual_storage.extend(r)  # Extend the storage with the results
        
        storage_of_terms[index] = individual_storage  # Store the results for the index
    
    return storage_of_terms


if __name__ == "__main__":
    import pickle as pkl
    
    t0 = time.perf_counter()
    iter_depth=2
    # =============================================================================
    # Symbolic
    # =============================================================================
    _k2, _k3, _x0, _x1, _a1, _a2, _A = symbols("k2 k3 x0 x1 a1 a2 A")
    
    g0 = GS([
        [  1, _x0, _x1],
        [_a1, _a2,   0]
    ])
    
    mult_quad = GS([
        [-_k2, _x0, _x0],
        [ _a1, _a2,   0]
    ])
    mult_cube = GS([
        [-_k3, _x0, _x0],
        [ _a1, _a2,  0]
    ])
    
    mults = [mult_quad, mult_cube]
    
    scheme = iterate_quad_cubic(g0, mults, iter_depth)
    
    # for i in scheme:
    #     to_bmatrix(i)
    #     print(r"\\")
    
    y_gs = convert_gs_to_time(scheme, _A, iter_depth)


    a1, a2 = shfl.sdof_roots(m, c, k1)
    
    vals = {
        _A: A,
        _a1: a1,
        _a2: a2,
        _k2: k2,
        _k3: k3,
    }
    
    # for i in range(iter_depth+1):
    #     with open(f"quad_cube_y{i+1}_gen_sym.txt", "wb") as f_sym:
    #         pkl.dump(y_gs[i], f_sym)
    #     print(i)
    #     temp = lambdify(symbols('t'), sum(y_gs[i]).subs(vals))(t)
    #     np.save(f"quad_cube_y{i+1}_gen_num.npy", temp)
    
    y1_g = lambdify(symbols('t'), sum(y_gs[0]).subs(vals))(t)
    y2_g = lambdify(symbols('t'), sum(y_gs[1]).subs(vals))(t)  # iter_depth = 1
    y3_g = lambdify(symbols('t'), sum(y_gs[2]).subs(vals))(t)  # iter_depth = 2
    # y4_g = lambdify(symbols('t'), sum(y_gs[3]).subs(vals))(t)  # iter_depth = 3
    # y5_g = lambdify(symbols('t'), y_gs[4].subs(vals))(t)  # iter_depth = 4
    # y6_g = lambdify(symbols('t'), y_gs[5].subs(vals))(t)  # iter_depth = 5

    # np.save(f"quad_cube_y3_k3_only_gen_num.npy", y3_g)
    
    
    # for i in y_gs.values():
    #     print(i)
    
    # =============================================================================
    # Plotting
    # =============================================================================
    print(f"time taken for full calculation was {time.perf_counter()-t0:.2f}s.")
    _figax = plot(y1_volt, None, "$y_1^v$")
    _figax = plot(y2_volt, _figax, "$y_2^v$")
    _figax = plot(y3_volt, _figax, "$y_3^v$")
    
    _figax = plot(y1_g, _figax, "$y^g_1$", linestyle="--")
    _figax = plot(y2_g, _figax, "$y^g_2$", linestyle="--")
    _figax = plot(y3_g, _figax, "$y^g_3$", linestyle="--")
    # _figax = plot(y4_g, _figax, "$y^g_4$", linestyle="--")
    # _figax = plot(y5_g, _figax, "$y^g_5$", linestyle="--")
    
    # _figax = plot(y1_g + y2_g + y3_g, _figax, "$y gen 3$", linestyle="--")
    # _figax = plot(y1_g + y2_g + y3_g + y4_g, _figax, "$y gen 4$", linestyle="--")
    # _figax = plot(y1_g + y2_g + y3_g + y4_g + y5_g, _figax, "$y gen 5$", linestyle="--")
    
    # _fig, _ax = _figax
    # _ax.set_title(
    #     f"Comparison of Duffing's equation solutions with Dirac delta {A}"
    # )
