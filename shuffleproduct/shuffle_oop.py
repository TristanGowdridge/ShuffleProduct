# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:34:17 2023

@author: trist

Write some notes about the module...

===============================================================================
 To Do list
===============================================================================
Implement unit tests for a cubic shuffle.

Implement my own caching that actually works.
"""
from collections import defaultdict
from itertools import permutations, product
from operator import itemgetter

import numpy as np


class GeneratingSeries():    
    __slots__ = ("array", "gs_hash")
    
    def __init__(self, array):
        """
        gs_hash is used as a key for the term exclusive of the coefficient,
        doing it this way means we can make use of the O(1) lookup of
        dictionaries when adding like terms. A defaultdictionary is used later
        on where the coefficients are added based on gs_hash.
        """
        if array.dtype == complex:
            self.array = array.astype(np.cdouble)
        elif array.dtype == object:
            raise TypeError(
                "Numpy object arrays give nondeterministic caching as they"\
                " contain pointers."
                )
        else:
            self.array = array.astype(np.double)

        self.gs_hash = hash(self)
               
    
    def __hash__(self):
        """
        I was doing everything symbolically, until I realised that you can't 
        have a reliable hash function in the symbolic case. As you need a numpy
        array with dtype='O', which is an array of pointers. When taking the 
        hash, this was nondeterminstic as the values were changing memory
        addresses, therefore the pointers were different and therefore the hash
        was changing. Now, all the operations regarding the shuffle product
        are performed numrically, and converted to a symbolic case after all
        the terms have been collected.
        
        This is the hash function which is used for caching.
        """
        top = self.array[0, 1:]
        bottom = self.array[1, :]

        return hash(top.tobytes()) + hash(bottom.tobytes())
    
    
    def prepend_multiplier(self, multiplier):
        """
        Rather than doing in place, I return here as I'm pretty sure doing it
        in place screws around with the referecing and hashes,
        """
        multiplier = multiplier.astype(self.array.dtype)
        
        if self.array.shape[1] == 1:
            raise IndexError("Hmmm, very curious case.")
            
        
        if multiplier.shape[1] == 1:
            arr_copy = np.copy(self.array)
            arr_copy[0, 0] = arr_copy[0, 0] * multiplier[0, 0]
            return GeneratingSeries(arr_copy)
        
        elif multiplier.shape[1] == 2:
            pre = np.zeros((2, 2), dtype=self.array.dtype)
            pre[0, 0] = multiplier[0, 0] * self.array[0, 0]
            pre[0, 1] = multiplier[0, 1]
            pre[1, 0] = multiplier[1, 0]
            pre[1, 1] = self.array[1, 0]
            temp = np.delete(self.array, 0, axis=1)
            
            return GeneratingSeries(np.hstack((pre, temp)))
        
        elif multiplier.shape[1] >= 3:
            mult_copy = np.copy(multiplier)
            mult_copy[1, -1] = self.array[1, 0]
            mult_copy[0,  0] = self.array[0, 0] * mult_copy[0, 0]
            arr = np.delete(self.array, 0, 1)
            
            return GeneratingSeries(np.hstack((mult_copy, arr)))
            
    
    def __len__(self):
        return self.shape[1]
    
    
    def __str__(self):
        return str(self.array)
    
    
    def __getitem__(self, indices):
        return self.array[indices]
    
    
    def __setitem__(self, indices, obj):
        self.array[indices] = obj
        
        
    def __eq__(self, other_obj):
        """
        Check if everything other than the coefficient are the same.
        
        This could be optimised with something like:
            return self.gs_hash == other_obj.gs_hash
        """
        if not isinstance(other_obj, GeneratingSeries):
            return False
        
        return self.gs_hash == other_obj.gs_hash
        
        
    def hard_equals(self, other_obj):
        """
        This is used for unit testing, when we want to include the coefficient
        in the comparison.
        """
        return np.array_equal(self.array, other_obj.array)
    
    
    def __add__(self, other_obj):
        if not (self == other_obj):
            raise ValueError("Cannot add different Generating Series.")
        else:
            return self.array[0, 0] + other_obj[0, 0]  
    
    
    def _fast_iadd(self, other_obj):
        """
        Used in functions that satisfy the conditions, could give errors if 
        not careful.
        """
        self.array[0, 0] = self.array[0, 0] + other_obj[0, 0]  
        
        return self
        
    
    def __iadd__(self, other_obj):
        if not isinstance(other_obj, GeneratingSeries):
            raise TypeError("Cannot add GeneratingSeries to another type.")
        
        if not self == other_obj:
            raise ValueError("Cannot add different Generating Series.")
        else:
            self._fast_iadd(other_obj)
            
        return self
    
    @property
    def shape(self):
        return self.array.shape
    
    def __repr__(self):
        return self.__str__()
 
    
class BinaryShuffle:
    def __init__(self): 
        self.term_stack = []
        self.coefficient_count = defaultdict(int)
        self.term_storage = {}
        self.hash_output = []
        
        # Only works if hasn't already been called. Therefore, when reusing 
        # instances (which is desireable for hashing purposes), this flag is
        # used to reset the instance.
        self._been_called = False 
    
    
    def __call__(self, gs1, gs2):
        """
        Given two input generating series, this returns the shuffle product of
        the two generating series. The output is a list of numpy arrays, with
        each instance of the list being a generating series.
        """
        if self._been_called:
            # Reset the shuffle product for next call. Hoping that this saves
            # caching conditions.
            self.term_storage = {} #Not entirely sure this is necessary
            self.hash_output = []
            self.coefficient_count = defaultdict(int)
        
        gs1, gs2 = self.sort_arguments(gs1, gs2)

        return self._subshuffle_wrapper(gs1, gs2)


    def sort_arguments(self, gs1, gs2):
        """
        Shuffle product is commutative, this will save time with the caching.
        """
        if gs1.gs_hash > gs2.gs_hash:
            gs1, gs2 = gs2, gs1
            
        return gs1, gs2
    
    
    def _subshuffle_wrapper(self, gs1, gs2):
        """
        Wrapper function for caching, should give a slight speed boost.
        """
        self._subshuffle(gs1, gs2)
        self._been_called = True
        self._combine_term_and_coefficient()
        return self.hash_output
    
             
    def _subshuffle(self, gs1, gs2):
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
        """
        gs1_len = len(gs1) # Saves 5.66% on iter_depth=6 
        gs2_len = len(gs2)
        if gs1_len != 1 and gs2_len != 1:
            # This is to decrease the size of the first term.
            gs1_reduced = np.array([
                [gs1[0, gs1_len-1]],
                [gs1[1, gs1_len-1] + gs2[1, gs2_len-1]]
            ])
            
            self.term_stack.append(gs1_reduced)
            self._subshuffle(
                GeneratingSeries(gs1[:, :(gs1_len-1)]),
                gs2
            )
            self.term_stack.pop()
            
            # This is to decrease the size of the second term.
            gs2_reduced = np.array([
                [gs2[0, gs2_len-1]],
                [gs1[1, gs1_len-1] + gs2[1, gs2_len-1]]
            ])
            self.term_stack.append(gs2_reduced)
            self._subshuffle(
                gs1,
                GeneratingSeries(gs2[:, :(gs2_len-1)]),
            )
            self.term_stack.pop()
            
        
        elif gs1_len != 1 and gs2_len == 1:
            gs1_reduced = np.array([
                [gs1[0, gs1_len - 1]],
                [gs1[1, gs1_len - 1] + gs2[1, 0]]
            ])
            self.term_stack.append(gs1_reduced)
            self._subshuffle(
                GeneratingSeries(gs1[:, :(gs1_len-1)]), 
                gs2,
            )
            self.term_stack.pop()
            
        
        elif gs1_len == 1 and gs2_len != 1:
            gs2_reduced = np.array([
                [gs2[0, gs2_len-1]],
                [gs1[1, 0] + gs2[1, gs2_len-1]]
            ])
            self.term_stack.append(gs2_reduced)
            self._subshuffle(
                gs1,
                GeneratingSeries(gs2[:, :(gs2_len-1)]),
            )
            self.term_stack.pop()
            
            
        elif gs1_len == 1 and gs2_len == 1:
            final_term = np.array([
                [gs1[0, 0] * gs2[0, 0]],
                [gs1[1, 0] + gs2[1, 0]]
            ])
            self.term_stack.append(final_term)
            
            temp = GeneratingSeries(np.hstack(self.term_stack[::-1]))
            temp_hash = temp.gs_hash
            self.coefficient_count[temp_hash] += temp[0, 0] 
            self.term_storage[temp_hash] = temp
                
            self.term_stack.pop()


    def _combine_term_and_coefficient(self):
        for (gs_hash, gs_term) in self.term_storage.items():
            temp = self.term_storage[gs_hash]
            temp[0, 0] = self.coefficient_count[gs_hash]
            
            self.hash_output.append(temp)


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
    shuff_obj = BinaryShuffle()
    output_gs = []
    gs1 = wrap_term(args[0], GeneratingSeries)
    gs2 = wrap_term(args[1], GeneratingSeries)
    for gs1_term in gs1:
        for gs2_term in gs2:
            temp_output = shuff_obj(gs1_term, gs2_term)
            temp_output = collect(temp_output)
            output_gs.extend(temp_output)
        output_gs = collect(output_gs)
    output_gs = collect(output_gs)
    
    # Iterate over the remaining arguments.
    for gs1 in args[2:]:
        storage = []
        for gs2 in output_gs:
            temp_output = shuff_obj(gs1, gs2)
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


def sdof_roots(a, b, c):
    """
    Floating point arithmetic is a bit off for exact roots, so specifically
    checking for integer roots and removing imaginary part.
    """
    threshold = 1e-5
    if b**2 - 4*a*c < 0:
        print("Complex roots")
    
    r = np.roots([a, b, c])
    
    if np.abs(r[0].imag) < threshold:
        r = np.real(r)
    
    return -r[0], -r[1]

if __name__ == "__main__":
    a = 1
    b = 1
    
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
    # Profile was ran for 7 iters.
    # Precache tests take 0.051-0.055s
    iterate_gs(g0, multiplier, 2, 4)