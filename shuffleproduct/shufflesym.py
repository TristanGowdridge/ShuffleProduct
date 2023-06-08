# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:10:37 2023

@author: trist


To Do:
    * Collect 001, 010, 100 -> 3 * 001. Caching and sorting probably do this
    anyway?
"""
import functools
import copy
from collections import defaultdict
from itertools import permutations, product
from operator import itemgetter
from typing import List, Tuple, Union

import numpy as np

from generating import GeneratingSeries


def shuffle_cacher(func):
    shuffle_cache = {}
    
    @functools.wraps(func)
    def wrapper(*args):
        # Sort the arguments to give ~2x cache hits.
        args = sorted(args, key=hash)
        
        key = tuple([hash(gs) for gs in args])
        coeff = 1
        for gs in args:
            coeff *= gs.coeff
        
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
                    prev.coeff *= scale
                    to_return.append(prev)
                return to_return
            else:
                return copy.deepcopy(prev_result)
    
    return wrapper


def reduction_term(g1, g2):
    """
    Gets the term to append to the stack when reducing g1.
    """
    return (g1[0], g1[1] + g2[1])


def add_to_stack(grid_sec, count, new_term, current_stack):
    """
    appends the term to the stack and places it in then calls the function to
    collect the grid
    """
    current_stack = copy.deepcopy(current_stack)
    current_stack.words.appendleft(new_term[0])
    current_stack.dens.appendleft(new_term[1])
    
    grid_sec.append((count, current_stack))


def collect_grid(terms):
    """
    
    """
    instance_counter = defaultdict(int)
    term_storage = dict()
    
    for count, term in terms:
        gs_hash = hash(term)
        instance_counter[gs_hash] += count
        if gs_hash not in term_storage:
            term_storage[gs_hash] = term
    
    collected_terms = []
    for key, term in term_storage.items():
        temp_term = (instance_counter[key], term)
        collected_terms.append(temp_term)
    
    return collected_terms


@shuffle_cacher
def binary_shuffle(gs1, gs2):
    """
    For the grid first index is number of reductions for gs2, second index is
    number of reductions of gs1).
    
    Reorder if statements so most likely is first.
    
    Dont store whole grid, only a selection is required.
    """
    gs1_len, gs2_len = len(gs1), len(gs2)
    
    grid = defaultdict(list)
    
    for i1 in range(gs1_len+1):
        is_reducible1 = (i1 < gs1_len)
        if is_reducible1:
            g1 = gs1[gs1_len-i1-1]

        for i2 in range(gs2_len+1):
            is_reducible2 = (i2 < gs2_len)
            if is_reducible2:
                g2 = gs2[gs2_len-i2-1]

            if is_reducible1 and is_reducible2:
                gs1_reduct = reduction_term(g1, g2)
                gs2_reduct = reduction_term(g2, g1)
            
            elif is_reducible1 and not is_reducible2:
                gs1_reduct = reduction_term(g1, (None, gs2.dens[0]))
                
            elif not is_reducible1 and is_reducible2:
                gs2_reduct = reduction_term(g2, (None, gs1.dens[0]))
            
            current = grid[(i2, i1)]
            if not current:
                grid[(0, 1)].append((1, GeneratingSeries(
                    1, [gs1_reduct[0]]
                )))
                grid[(1, 0)].append((1, GeneratingSeries(
                    1, [gs2_reduct[0]]
                )))
                continue
            
            for count, curr in collect_grid(current):
                if is_reducible1 and is_reducible2:
                    add_to_stack(
                        grid[(i2, i1+1)], count, gs1_reduct, curr
                    )
                    add_to_stack(
                        grid[(i2+1, i1)], count, gs2_reduct, curr
                    )

                elif is_reducible1 and not is_reducible2:
                    add_to_stack(
                        grid[(i2, i1+1)], count, gs1_reduct, curr
                    )
                
                elif not is_reducible1 and is_reducible2:
                    add_to_stack(
                        grid[(i2+1, i1)], count, gs2_reduct, curr
                    )
                    grid[(i2+1, i1)] = collect_grid(grid[(i2+1, i1)])

    to_return = []
    for count, term in grid[(gs2_len, gs1_len)]:
        term.coeff *= count * gs1.coeff * gs2.coeff
        term.dens.appendleft(gs1.dens[0] + gs2.dens[0])
        to_return.append(term)
        
    return tuple(to_return)


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


def collect(output: List[GeneratingSeries]) -> List[GeneratingSeries]:
    """
    This collects all like-terms loops over the generating series in the
    output.
    """
    coefficient_count = defaultdict(int)
    term_storage = {}
    output_collected = []
    
    for gs in output:
        coefficient_count[hash(gs)] += gs.coeff
        term_storage[hash(gs)] = gs

    for term_hash, coeff in coefficient_count.items():
        temp = term_storage[term_hash]
        temp.coeff = coeff
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


# @shuffle_cacher
# def nShuffles_FAST(
#         *args: Union[GeneratingSeries, List[GeneratingSeries]]
# ) -> List[GeneratingSeries]:
#     """
#     This takes variadic input, greater than 2 and outputs the shuffle product.
    
#     Don't know what to do on the reductions?
#     """
#     gs_lens = np.array([len(gs) for gs in args])
    
#     grid = defaultdict(list)
    
#     for index in product(*[range(gs_len+1) for gs_len in gs_lens]):
#         index = np.array(index)
#         is_reducible = index < gs_lens
        
        
def iter_gs_worker(part, term_storage, depth):
    """
    This is the CPU-intensive section the generating series expansion.
    """
    terms = itemgetter(*part)(term_storage)
    # Cartesian product of all the inputs, instead of nested for-loop. Same
    # thing but avoids unnecessary nesting.
    for in_perm in product(*terms):
        term_storage[depth + 1].extend(nShuffles(*in_perm))
    

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
    multipliers = wrap_term(multipliers, GeneratingSeries)
    g0 = wrap_term(g0, GeneratingSeries)
    
    term_storage = defaultdict(list)
    term_storage[0].extend(g0)
    
    for depth in range(iter_depth):
        for part in partitions(depth, n_shuffles):
            iter_gs_worker(part, term_storage, depth)
        term_storage[depth + 1] = collect(term_storage[depth + 1])
        
        # After the shuffles for this iteration's depth have been caluclated,
        # prepend the multiplier to each term.
        for gs_term in term_storage[depth + 1]:
            for multiplier in multipliers:
                gs_term.prepend_multiplier(multiplier)
        
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
            tuple_form.append(
                (gs.coeff, np.array([gs.words, gs.dens]))
            )
        return tuple_form
    
    else:
        raise TypeError("Invalid return type.")


def sdof_roots(m, c, k):
    """
    Values are passed in their usual orders, however the quadratic coefficient
    is k, therefore need to reverse the order when passed into np.roots.
    
    As a result of floating point precision, when the determinant == 0, some
    complex artefacts can be introduced, in this case, I take the real part of
    the roots.
    """
    det = c**2 - 4*k*m
    
    roots = np.roots([m, c, k])  # Reversed as quadratic is kx^2 + cx + m.
    if det == 0:
        roots = np.real(roots)
    
    a1, a2 = roots
        
    return -a1, -a2


if __name__ == "__main__":
    from sympy import Symbol
    x0 = Symbol("x0")
    x1 = Symbol("x1")
    
    b = Symbol("b")
    a = Symbol("a")
    
    multiplier = GeneratingSeries([
        [-b, x0],
        [ a,  0]
    ])
        
    g0 = GeneratingSeries([
        [ 1, x1],
        [ a, 0]
    ])
    
    from time import perf_counter
    t0 = perf_counter()
    test = iterate_gs(g0, multiplier, 2, 6)
    print(perf_counter()-t0)
