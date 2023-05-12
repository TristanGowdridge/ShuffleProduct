# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:10:37 2023

@author: trist
"""
import functools
from collections import defaultdict
from itertools import permutations, product
from operator import itemgetter

import numpy as np
import copy

from generating_series import GeneratingSeries


def binshuffle_cacher(func):
    binshuf_cache = {}
    
    @functools.wraps(func)
    def wrapper(gs1, gs2):
        # Sort the arguments to give ~2x cache hits.
        if gs1.gs_hash > gs2.gs_hash:
            gs1, gs2 = gs2, gs1
        
        key = (gs1.gs_hash, gs2.gs_hash)
        coeff = gs1[0, 0] * gs2[0, 0]
        
        if key not in binshuf_cache:
            result = func(gs1, gs2)
            to_cache = copy.deepcopy(result)
            binshuf_cache[key] = (coeff, to_cache)
            return result
        
        else:
            prev_coeff, prev_result = binshuf_cache[key]
            
            if prev_coeff != coeff:
                print("Cache hit different coeffs")
                scale = coeff / prev_coeff
                to_return = []
                for prev in copy.deepcopy(prev_result):
                    prev[0, 0] *= scale
                    to_return.append(prev)
                return to_return
            else:
                # print("Cache hit same coeffs")
                return copy.deepcopy(prev_result)
    return wrapper



@binshuffle_cacher
def binary_shuffle(gs1, gs2):
    """
    Given two input generating series, this returns the shuffle product of
    the two generating series. The output is a list of numpy arrays, with
    each instance of the list being a generating series.
    """
    term_stack = []
    coefficient_count = defaultdict(int)
    term_storage = {}
    
    stack_diff = None
    gs1_f = None
    gs2_f = None
    
    cache_arg = (stack_diff, gs1_f, gs2_f)
    subshuffle(gs1, gs2, term_stack, coefficient_count, term_storage, cache_arg)
    
    return combine_term_and_coefficient(term_storage, coefficient_count)


def subshuffle_cacher(func):
    """
    Store the term stack in the cache at the end of the call.
    """
    global subshuf_cache
    subshuf_cache = {}

    @functools.wraps(func)
    def wrapper(gs1, gs2, term_stack, coefficient_count, term_storage, cache_arg):
        if gs1.gs_hash > gs2.gs_hash:
            gs1, gs2 = gs2, gs1
        key = (gs1.gs_hash, gs2.gs_hash)
        coeff = gs1[0, 0] * gs2[0, 0]
        
        
        if key not in subshuf_cache:
            cached_state = func(gs1, gs2, term_stack, coefficient_count, term_storage, cache_arg)
            subshuf_cache[key] = (coeff, copy.deepcopy(cached_state))
            return cached_state
        
        else:   
            prev_coeff, (cached_stack, gs1_f, gs2_f) = subshuf_cache[key]
            cached_stack = copy.deepcopy(cached_stack)
            if prev_coeff != coeff and cached_stack:
                
                # scale = coeff / prev_coeff              
                # term_stack1 = term_stack + cached_stack
                # return func(gs1_f, gs2_f, term_stack1, coefficient_count, term_storage, cache_arg)
                
                # Safe fall back return.
                return func(gs1, gs2, term_stack, coefficient_count, term_storage, cache_arg)
            
            elif cached_stack:
                # print("Cache hit same coeffs sub shuffle")
                # term_stack1 = term_stack + cached_stack
                # return func(gs1_f, gs2_f, term_stack1, coefficient_count, term_storage, cache_arg)
                
                return func(gs1, gs2, term_stack, coefficient_count, term_storage, cache_arg)
            
            else:
                # When the cached stack is 0.
                return func(gs1, gs2, term_stack, coefficient_count, term_storage, cache_arg)
        
    return wrapper


def handle_gs_constants(gs1, gs2, *terms_args):
    """
    Handles when both generating series have length 1.
    """
    term_stack, coefficient_count, term_storage, stack_diff = terms_args

    final_term = np.array([
        [gs1[0, 0] * gs2[0, 0]],
        [gs1[1, 0] + gs2[1, 0]]
    ])

   
    term_stack.append(final_term)
    
    # Save the system state in final state.
    final_stack = copy.deepcopy(term_stack)
    gs1_f = copy.deepcopy(gs1)
    gs2_f = copy.deepcopy(gs2)
    final_state = (final_stack, gs1_f, gs2_f)
    
    
    # Create the GeneratingSeries term now we are at the end of the recursion.
    temp = GeneratingSeries(np.hstack(term_stack[::-1]))
    temp_hash = temp.gs_hash
    
    coefficient_count[temp_hash] += temp[0, 0]
    term_storage[temp_hash] = temp

    term_stack.pop()
    
    return final_state


def reduce_gs(gs1, gs2, *terms_args):
    """
    Reduces gs1.
    """
    terms_args[0].append(
        np.array([
        [      gs1[0, -1]       ],
        [gs1[1, -1] + gs2[1, -1]]
    ]))

    final_state = subshuffle(GeneratingSeries(gs1[:, :-1]), gs2, *terms_args)
    terms_args[0].pop()
    
    return final_state


@subshuffle_cacher
def subshuffle(gs1, gs2, *terms_args):
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
            This case results in the calculation of two more shuffles.  
        len(gs1) != 1 and len(gs2) = 1:
            This means that only gs1 can be reduced further, therefore
            resulting in the calculation of one more shuffle.
        len(gs1) = 1 and len(gs2) != 1:
            This means that only gs2 can be reduced further, therefore
            resulting in the calculation of one more shuffle.
        len(gs1) = 1 and len(gs2) = 1:
            This is the terminating criterion.
            
    SUBSHUFFLE RETURNS (stack_diff, gs1_f, gs2_f)
    """
    if len(gs1) != 1 and len(gs2) != 1:
        # Only caching the first reduction here.
        final_stack, gs1_f, gs2_f = reduce_gs(gs1, gs2, *terms_args)
        stack_diff1 = final_stack[len(terms_args[0]) : -1]
        
        reduce_gs(gs2, gs1, *terms_args)
        
        return stack_diff1, gs1_f, gs2_f

        

    elif len(gs1) != 1 and len(gs2) == 1:
        final_stack, gs1_f, gs2_f = reduce_gs(gs1, gs2, *terms_args)
        stack_diff = final_stack[len(terms_args[0]) : -1]
        return stack_diff, gs1_f, gs2_f


    elif len(gs1) == 1 and len(gs2) != 1:
        final_stack, gs1_f, gs2_f = reduce_gs(gs2, gs1, *terms_args)
        stack_diff = final_stack[len(terms_args[0]) : -1]
        return stack_diff, gs1_f, gs2_f
     
    
    elif len(gs1) == 1 and len(gs2) == 1:
        final_state = handle_gs_constants(gs1, gs2, *terms_args)
        return final_state



def combine_term_and_coefficient(term_storage, coefficient_count):
    hash_output = []
    for gs_hash, gs_term in term_storage.items():
        temp = term_storage[gs_hash]
        temp[0, 0] = coefficient_count[gs_hash]
        hash_output.append(temp)
    
    return tuple(hash_output)


def collect(output):
    """
    This collects all like-terms by adding the coefficients and marks the
    second instance as "to_delete".
    """
    coefficient_count = defaultdict(int)
    term_storage = {}
    output_collected = []
    for gs in output:
        if gs[0, 0] == 0:
            continue
        coefficient_count[gs.gs_hash] += gs[0, 0]
        term_storage[gs.gs_hash] = gs

    for term_hash, coeff in coefficient_count.items():
        temp = term_storage[term_hash]
        temp[0, 0] = coeff
        output_collected.append(temp)

    return output_collected
 
    
def wrap_term(term, data_type):
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
 

def partitions(iter_depth, number_of_shuffles):
    """
    Rather than coming up with a fancy way of generating all the partitions of
    a number, this uses the itertools' product function to calculate the
    Cartesian product, and then eliminate all the entries that don't sum to
    iter_depth. Therefore giving all the partitions.
    
    There is an edge case at iter_depth == 1, in this case all the permutations
    of 1 and padding zeros are output.
    """
    if iter_depth == 0:
        parts = [tuple([0] * number_of_shuffles)]
    
    elif iter_depth == 1:
        parts = list(set(permutations([0] * (number_of_shuffles-1) + [1])))
    
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

    
def nShuffles(*args):
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
        storage = []
        for gs2 in output_gs:
            temp_output = binary_shuffle(gs1, gs2)
            temp_output = collect(temp_output)
            storage.extend(temp_output)
        output_gs = collect(storage)
    output_gs = collect(output_gs)
    
    return output_gs


def iterate_gs(g0, multipliers, n_shuffles, iter_depth=2, return_type=tuple):
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
    # global inputs
    for depth in range(iter_depth):
        for part in partitions(depth, n_shuffles):
            # Cartesian product of all the inputs, instead of nested for-loop.
            terms = itemgetter(*part)(term_storage)
            for in_perm in product(*terms):
                term_storage[depth+1].extend(nShuffles(*in_perm))
            term_storage[depth+1] = collect(term_storage[depth+1])
        
        # After the shuffles for this iteration's depth have been caluclated,
        # prepend the multiplier to each term.
        next_terms = []
        for gs_term in term_storage[depth+1]:
            for multiplier in multipliers:
                next_terms.append(gs_term.prepend_multiplier(multiplier))
        term_storage[depth+1] = next_terms
        
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


if __name__ == "__main__":
    a = 3
    b = 2
    
    x0 = 0
    x1 = 1
    
    multiplier = np.array([
        [-b, x0],
        [ a,  0]
    ])
    
    g0 = GeneratingSeries(np.array([
        [ 1, x1],
        [ a,  0]
    ]))

    g1a = GeneratingSeries(np.array([
        [-2*b,   x0,   x1,  x1],
        [   a,  2*a,    a,   0]
    ]))
    
    g2a = [
        GeneratingSeries(np.array([
            [4*b**2,  x0, x1,  x0, x1, x1],
            [     a, 2*a,  a, 2*a,  a,  0]
        ])),
        GeneratingSeries(np.array([
            [12*b**2, x0,  x0,  x1,  x1, x1],
            [     a, 2*a, 3*a, 2*a,   a,  0]
        ]))
    ]
    g1 = binary_shuffle(g0, g0)    
    g1_terms = []
    for g1_term in g1:
        g1_terms.append(g1_term.prepend_multiplier(multiplier))
    g1a.hard_equals(g1_terms[0])
    
    g2 = []
    for g1_term in g1_terms:   
        g2.extend(binary_shuffle(g1_term, g0))
        g2.extend(binary_shuffle(g0, g1_term))
    assert (len(g2) == 4)
    
    g2 = collect(g2)
    print(f"g2 has a length {len(g2)}")
    print(*g2, sep="\n")
