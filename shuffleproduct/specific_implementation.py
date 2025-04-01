import os
from collections import defaultdict
from operator import itemgetter
from itertools import product
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sympy import symbols, factorial, apart
from sympy.core.add import Add as SympyAdd


from . import shuffle as shfl
from .generating_series import GeneratingSeries as GS
from .responses import convert_term




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


def worker(term):
    """
    Worker function for the conversion.
    """
    if isinstance(term, SympyAdd):
        ts = []
        for term1 in term.make_args(term):
            ts.append(convert_term(term1))
        return tuple(ts)
    else:
        return (convert_term(term),)


def parallel_inverse_lb_and_save(pf):
    """
    In parallel compute the inverse Laplace-Borel transform.
    The multiprocessing logic will now be inside a main block.
    """
    # Ensure the following code only runs when the script is executed directly.
    result = []
    if False:
        
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Pass the worker function to the executor map
            for r in executor.map(worker, pf):
                result.extend(r)
        return tuple(result)
    else:
        result = []
    
        # Serial loop through pf and apply the worker function
        for term in pf:
            r = worker(term)
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

        # if len(gs) < cpu_cnt:  # If the number of terms is less than CPU count, process sequentially
        if True:  # If the number of terms is less than CPU count, process sequentially
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