# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:10:37 2023

@author: trist

Collect like terms in the cache.
"""
import functools
import copy
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from itertools import permutations, product, repeat
from operator import itemgetter
from typing import List, Tuple, Union

import numpy as np

from generating_series import GeneratingSeries


def shuffle_cacher(func):
    shuffle_cache = {}
    
    @functools.wraps(func)
    def wrapper(*args):
        # Sort the arguments to give ~2x cache hits.
        args = sorted(args, key=hash)
        
        key = tuple([gs.gs_hash for gs in args])
        coeff = 1
        for gs in args:
            coeff *= gs[0, 0]
        
        if key not in shuffle_cache:  # Store the result if unknown.
            result = func(*args)
            to_cache = copy.deepcopy(result)
            shuffle_cache[key] = (coeff, to_cache)
            return result
        
        else:  # Lookup and scale if known.
            prev_coeff, prev_result = shuffle_cache[key]
            
            if prev_coeff != coeff:
                scale = coeff / prev_coeff
                to_return = []
                for prev in copy.deepcopy(prev_result):
                    prev[0, 0] *= scale
                    to_return.append(prev)
                return to_return
            else:
                return copy.deepcopy(prev_result)
    
    return wrapper


@shuffle_cacher
def binary_shuffle(
        gs1: GeneratingSeries, gs2: GeneratingSeries
) -> List[GeneratingSeries]:
    """
    Given two input generating series, this returns the shuffle product of
    the two generating series. The output is a list of numpy arrays, with
    each instance of the list being a generating series.
    """
    term_stack = np.array([[], []])
    coefficient_count = defaultdict(int)
    term_storage = {}

    subshuffle(gs1, gs2, term_stack, coefficient_count, term_storage)
    
    return combine_term_and_coefficient(term_storage, coefficient_count)


def subshuffle_cacher(func):
    """
    If the current shuffle expansion is cached, then this accelerates the
    calculation to the shuffle terminating criterion (len(gs1) == 1 &
    len(gs2) == 1) with the term stack difference being appended for the
    calculation, essentially reducing the order of complexity in this case from
    O(2 ** (len(gs1) + len(gs2))) -> O(len(gs1) + len(gs2)).
    """
    global subshuf_cache
    # Storage for previously calculated expansions.
    subshuf_cache = dict()
    
    def collect_state_diffs(state_diffs: tuple) -> tuple:
        """
        Collects like terms in the cache, resulting in fewer loop iterations
        when the cached values are looked up.
        """
        if len(state_diffs) == 1:
            return (1, *state_diffs[0]),
        
        cache_hashes = {}
        instance_counter = defaultdict(int)
        for arr, gs1_f, gs2_f in state_diffs:
            key = hash(
                arr.tobytes() + gs1_f.array.tobytes() + gs2_f.array.tobytes()
            )
            cache_hashes[key] = (arr, gs1_f, gs2_f)
            instance_counter[key] += 1
            
        collected_states = []
        for key, (arr, gs1_f, gs2_f) in cache_hashes.items():
            collected_states.append((instance_counter[key], arr, gs1_f, gs2_f))
            
        return tuple(collected_states)
    
    @functools.wraps(func)
    def wrapper(
            gs1: GeneratingSeries, gs2: GeneratingSeries,
            term_stack: list, *stor_args) -> tuple:
        
        # Sort the generating series to ~2x the cache hits.
        if gs1.gs_hash > gs2.gs_hash:
            gs1, gs2 = gs2, gs1

        key = (gs1.gs_hash, gs2.gs_hash)
        coeff = gs1[0, 0] * gs2[0, 0]

        if key not in subshuf_cache:
            state_diffs, final_states = func(gs1, gs2, term_stack, *stor_args)
            # subshuf_cache[key] = (coeff, collect_state_diffs(state_diffs))
            subshuf_cache[key] = (coeff, state_diffs)
            
            return state_diffs, final_states
        
        else:
            prev_coeff, state_diffs = subshuf_cache[key]
            all_final_states = []
            # for count, stack_diffs, gs1_f, gs2_f in state_diffs:
            for stack_diffs, gs1_f, gs2_f in state_diffs:

                if coeff != prev_coeff:
                    gs1_f = copy.deepcopy(gs1_f)
                    gs1_f[0, 0] *= coeff / prev_coeff
                    # gs1_f[0, 0] *= count * coeff / prev_coeff
                # Append the stack_diff so this can now be called the
                # terminating criterion with gs1_f and gs2_f.
                full_stack = np.hstack([term_stack, stack_diffs])
                _, final_states = func(gs1_f, gs2_f, full_stack, *stor_args)
                all_final_states.extend(final_states)
            
            all_state_diffs = stack_difference(final_states, term_stack)
            
            return tuple(all_state_diffs), tuple(all_final_states)

    return wrapper
   

@subshuffle_cacher
def subshuffle(
        gs1: GeneratingSeries, gs2: GeneratingSeries, *term_args) -> tuple:
    """
    This takes two generating series in the 'array' form that is outlined
    in Fleiss's papers and calculates the shuffle product based on a
    recursion over the lengths of the two generating series. The shuffle
    product terminates when the lengths of the inputs cannot be reduced
    further.
    
    The shuffle product must fit into one of four categories, based on the
    lengths of the arguments:
        len(gs1) != 1 and len(gs2) != 1:
            This case means that both generating series need to be reduced.
            This case results in the calculation of two more shuffles at this
            level.
        len(gs1) != 1 and len(gs2) = 1:
            This means that only gs1 can be reduced further, therefore
            resulting in the calculation of one more shuffle at this level.
        len(gs1) = 1 and len(gs2) != 1:
            This means that only gs2 can be reduced further, therefore
            resulting in the calculation of one more shuffle at this level.
        len(gs1) = 1 and len(gs2) = 1:
            This is the terminating criterion.
    """
    
    if len(gs1) != 1 and len(gs2) != 1:
        state_diffs1, states_final1 = reduce_gs(gs1, gs2, *term_args)
        state_diffs2, states_final2 = reduce_gs(gs2, gs1, *term_args)
        return (*state_diffs1, *state_diffs2), (*states_final1, *states_final2)

    elif len(gs1) != 1 and len(gs2) == 1:
        state_diffs, states_final = reduce_gs(gs1, gs2, *term_args)
        return state_diffs, states_final
    
    elif len(gs1) == 1 and len(gs2) != 1:
        state_diffs, states_final = reduce_gs(gs2, gs1, *term_args)
        return state_diffs, states_final
        
    elif len(gs1) == 1 and len(gs2) == 1:
        final_states = handle_gs_constants(gs1, gs2, *term_args)
        term_stack = term_args[0]
        stack_diffs = stack_difference(final_states, term_stack)
        return stack_diffs, final_states


def reduce_gs(
        gs1: GeneratingSeries, gs2: GeneratingSeries,
        term_stack: List[np.ndarray], *stor_args) -> tuple:
    """
    Reduces gs1 and makes a call to the shuffle of the reduced term.
    """
    temp_term = np.array([
        [gs1[0, -1]            ],
        [gs1[1, -1] + gs2[1, -1]]
    ])
    # Add the term formed as a result of the reduction to the stack.
    term_stack = np.hstack([term_stack, temp_term])
    
    reduced_gs = GeneratingSeries(gs1[:, :-1])
    _, states_final = subshuffle(reduced_gs, gs2, term_stack, *stor_args)
    
    # Pop from the stack.
    term_stack = term_stack[:, :-1]
    
    state_diffs = stack_difference(states_final, term_stack)
    
    return state_diffs, states_final


def handle_gs_constants(
        gs1: GeneratingSeries, gs2: GeneratingSeries, term_stack: list,
        coefficient_count: defaultdict, term_storage: list
) -> tuple:
    """
    Handles when both generating series have length 1.
    """
    final_term = np.array([
        [gs1[0, 0] * gs2[0, 0]],
        [gs1[1, 0] + gs2[1, 0]]
    ])

    term_stack = np.hstack([term_stack, final_term])
    
    # Create the GeneratingSeries term now we are at the end of the recursion.
    temp = GeneratingSeries(term_stack[:, ::-1])
    temp_hash = temp.gs_hash
    
    coefficient_count[temp_hash] += temp[0, 0]
    term_storage[temp_hash] = temp
    final_stack = term_stack

    term_stack = term_stack[:, :-1]
    
    return ((final_stack, gs1, gs2),)


def stack_difference(
        final_states: tuple, current_term_stack: List[np.ndarray]
) -> tuple:
    """
    Determines the stack difference between the current stack and the final
    stack. The output of this function is what is stored in subshuf_cache.
    """
    current_stack_length = current_term_stack.shape[1]
        
    state_diffs = [
        (final_stack[:, current_stack_length:-1], gs1_f, gs2_f)
        for final_stack, gs1_f, gs2_f in final_states
    ]
        
    return tuple(state_diffs)
  

def combine_term_and_coefficient(
        term_storage: defaultdict, coefficient_count: defaultdict
) -> tuple[GeneratingSeries]:
    """
    Loops over the generating series and applies the coefficient to the term.
    """
    hash_output = []
    for gs_hash, gs_term in term_storage.items():
        temp = term_storage[gs_hash]
        temp[0, 0] = coefficient_count[gs_hash]
        hash_output.append(temp)
    
    return tuple(hash_output)


def collect(output: List[GeneratingSeries]) -> List[GeneratingSeries]:
    """
    This collects all like-terms loops over the generating series in the
    output.
    """
    coefficient_count = defaultdict(int)
    term_storage = {}
    output_collected = []
    
    for gs in output:
        coefficient_count[gs.gs_hash] += gs[0, 0]
        term_storage[gs.gs_hash] = gs

    for term_hash, coeff in coefficient_count.items():
        temp = term_storage[term_hash]
        temp[0, 0] = coeff
        output_collected.append(temp)

    return output_collected
 
    
def wrap_term(term: Union[List, GeneratingSeries], data_type) -> List:
    """
    This is used to wrap terms as some functions iterate over lists, but in
    some cases singular objects may be passed.
    """
    if isinstance(term, list):
        pass
    elif isinstance(term, data_type):
        term = [term]
    else:
        raise TypeError(f"Inputs needs to have type {data_type}, or list.")
    
    return term
 

def partitions(iter_depth: int, number_of_shuffles: int) -> Tuple[int]:
    """
    Rather than coming up with a fancy way of generating all the partitions of
    a number, this uses the itertools' product function to calculate the
    Cartesian product, and then eliminate all the entries that don't sum to
    iter_depth. Therefore giving all the partitions.
    
    There is an edge case at iter_depth == 1, in this case all the permutations
    of 1 and padding zeros are output.
    """
    if iter_depth == 0:
        parts = [(0, ) * number_of_shuffles]
    
    elif iter_depth == 1:
        parts = list(set(permutations([0] * (number_of_shuffles - 1) + [1])))
    
    else:
        parts = product(range(iter_depth), repeat=number_of_shuffles)
        parts = [i for i in parts if sum(i) == iter_depth]
        iter_depth_with_zeros = list(
            set(
                permutations([0] * (number_of_shuffles - 1) + [iter_depth])
            )
        )
        parts.extend(iter_depth_with_zeros)
    
    return parts


@shuffle_cacher
def nShuffles(
        *args: Union[GeneratingSeries, List[GeneratingSeries]]
) -> List[GeneratingSeries]:
    """
    This takes variadic input, greater than 2 and outputs the shuffle product.
    """
    if len(args) < 2:
        raise ValueError("nShuffles requires two or more inputs.")
    
    # Manually calculate the shuffle product of the first two arguments.
    output_gs = []
    gs1 = wrap_term(args[0], GeneratingSeries)
    gs2 = wrap_term(args[1], GeneratingSeries)
    for gs1_term in gs1:
        for gs2_term in gs2:
            temp_output = binary_shuffle(gs1_term, gs2_term)
            temp_output = collect(temp_output)
            output_gs.extend(temp_output)
        output_gs = collect(output_gs)
    output_gs = collect(output_gs)
    
    # Iterate over the remaining arguments.
    for gs1 in args[2:]:
        # gs1 = wrap_term(gs1, GeneratingSeries)
        storage = []
        for gs2 in output_gs:
            temp_output = binary_shuffle(gs1, gs2)
            temp_output = collect(temp_output)
            storage.extend(temp_output)
        output_gs = collect(storage)
    output_gs = collect(output_gs)
    
    return output_gs


def iter_gs_worker(part, term_storage, depth):
    """
    This is the CPU-intensive section the generating series expansion.
    """
    terms = itemgetter(*part)(term_storage)
    # Cartesian product of all the inputs, instead of nested for-loop. Same
    # thing but avoids unnecessary nesting.
    for in_perm in product(*terms):
        term_storage[depth + 1].extend(nShuffles(*in_perm))
    term_storage[depth + 1] = collect(term_storage[depth + 1])
    

def iterate_gs(
        g0: Union[GeneratingSeries, List[GeneratingSeries]],
        multipliers: List[np.ndarray],
        n_shuffles: int,
        iter_depth: int = 2,
        return_type=tuple
) -> Tuple[GeneratingSeries]:
    """
    This follows the iterative procedure of determining the generating series
    by summing over the shuffles of all the partitions of a number. The
    multiplier is the multiplier at the end of every interation step.
    g0 is the initial term used in the iteration. This function is only valid
    for when there is a singular shuffle product of length n. For instance
    when multiple shuffle products are required, such as in a duffing
    oscillator with a quadratic term, there are two distinct shuffle products
    required, one for the quadratic nonlinearity and one for the cubic.
    """
    multipliers = wrap_term(multipliers, np.ndarray)
    g0 = wrap_term(g0, GeneratingSeries)
    
    term_storage = defaultdict(list)
    term_storage[0].extend(g0)
    
    for depth in range(iter_depth):
        for part in partitions(depth, n_shuffles):
            iter_gs_worker(part, term_storage, depth)
        # After the shuffles for this iteration's depth have been caluclated,
        # prepend the multiplier to each term.
        next_terms = []
        for gs_term in term_storage[depth + 1]:
            for multiplier in multipliers:
                next_terms.append(gs_term.prepend_multiplier(multiplier))
        term_storage[depth + 1] = next_terms
        
    return handle_output_type(term_storage, return_type)
    

def iterate_gs_par(
        g0, multipliers, n_shuffles, iter_depth=2, return_type=tuple):
    """
    This follows the iterative procedure of determining the generating series
    by summing over the shuffles of all the partitions of a number. The
    multiplier is the multiplier at the end of every interation step.
    g0 is the initial term used in the iteration. This function is only valid
    for when there is a singular shuffle product of length n. For instance
    when multiple shuffle products are required, such as in a duffing
    oscillator with a quadratic term, there are two distinct shuffle products
    required, one for the quadratic nonlinearity and one for the cubic.
    """
    multipliers = wrap_term(multipliers, np.ndarray)
    g0 = wrap_term(g0, GeneratingSeries)
    
    term_storage = defaultdict(list)
    term_storage[0].extend(g0)
    
    with ProcessPoolExecutor() as executor:
        for depth in range(iter_depth):
            executor.map(
                iter_gs_worker,
                zip(
                    partitions(depth, n_shuffles),
                    repeat(term_storage),
                    repeat(depth)
                )
            )
            # After the shuffles for this iteration's depth have been
            # calculated, prepend the multiplier to each term.
            next_terms = []
            for gs_term in term_storage[depth + 1]:
                for multiplier in multipliers:
                    next_terms.append(gs_term.prepend_multiplier(multiplier))
            term_storage[depth + 1] = next_terms
        
    return handle_output_type(term_storage, return_type)
    

def handle_output_type(term_storage, return_type):
    """
    Three output forms are given. The dictionary output gives the most
    stucture, where the keys represent generating series terms specific to an
    iteration depth. The list output simply returns a list of all the
    generating series, whilst they do appear in order, nothing breaks the
    order apart (unlike the dictionary). The tuple output is the form required
    for converting the generating series into the time domain. A function in
    the responses module converts the generating series array form into a
    fractional form.
    """
    if return_type == dict:
        return dict(term_storage)
    
    list_form = [i for gs in term_storage.values() for i in gs]
    if return_type == list:
        return list_form
    
    elif return_type == tuple:
        # Unpack all the gs terms into a list
        tuple_form = []
        for gs in list_form:
            if gs.shape[1] > 1:
                array = GeneratingSeries(np.array([
                    gs[0,  1:],
                    gs[1, :-1]
                ]))
                tuple_form.append((gs[0, 0], array))
            else:
                coeff = gs[0, 0]
                print("2 indicates that there is no term.")
                gs[0, 0] = 2

                tuple_form.append((coeff, gs))
        return tuple_form
    else:
        raise TypeError("Invalid return type.")


def sdof_roots(m, c, k):
    """
    As a result of floating point precision, when the determinant == 0, some
    complex artefacts can be introduced, in this case, I take the real part of
    the roots.
    """
    det = c**2 - 2*m*k
    
    roots = np.roots([m, c, k])
    if det == 0:
        roots = np.real(roots)
    
    r1, r2 = roots
        
    return -r1, -r2


if __name__ == "__main__":
    x0 = 0
    x1 = 1
    
    a = 2
    b = 3

    multiplier = np.array([
        [-b, x0],
        [ a,  0]
    ])
    
    g0 = GeneratingSeries(np.array([
        [ 1, x1],
        [ a,  0]
    ]))
    
    from time import perf_counter
    iter_args = (g0, multiplier, 2)
    t0 = perf_counter()
    scheme_tab = iterate_gs(*iter_args, iter_depth=5)
    print(perf_counter()-t0)