# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:34:17 2023

@author: trist

Write some notes about the module...

===============================================================================
 To Do list
===============================================================================
Implement unit tests for a cubic shuffle.
"""
import numpy as np
import functools
from collections import defaultdict
from itertools import permutations, product
import warnings

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
            warnings.warn("Array is complex, not entirely sure if this is appropriate or if maths needs changing.")
            self.array = array.astype(np.cdouble)
        else:
            self.array = array.astype(np.double)
        self.gs_hash = self.term_hash()
        

    def term_hash(self):
        """
        This is the hash used for matching the generating series term exclusive
        of the coefficient.
        """
        top = np.copy(self.array[0, 1:])
        bottom = np.copy(self.array[1, :])

        return hash(top.tobytes()) + hash(bottom.tobytes())
        
    
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
        
        This is the hash function which is used for cahching.
        """
        # top = np.copy(self.array[0, 1:])
        # bottom = np.copy(self.array[1, :])

        # return hash(top.tobytes()) + hash(bottom.tobytes())
        return hash(self.array.tobytes())
        
    
    def prepend_multiplier(self, multiplier):
        """
        Rather than doing in place, I return here as I'm pretty sure doing it
        in place screws around with the referecing and hashes,
        """
        if multiplier.shape[1] == 1:
            return GeneratingSeries(self.array[0, 0] * multiplier[0, 0])
        
        elif multiplier.shape[1] == 2:
            
            pre = np.zeros((2, 2), dtype='O') # 'O' Required for sympy objects.
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
    
    
    # @functools.lru_cache(maxsize=None)
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
        coefficient_count[gs.gs_hash] += gs[0, 0]
        term_storage[gs.gs_hash] = gs

    for term_hash, coeff in coefficient_count.items():
        temp = term_storage[term_hash]
        temp[0, 0] = coeff
        output_collected.append(temp)

    return output_collected
 
    
def wrap_generating_series(gs):
    """
    This is used for nShuffles(), as it is assumed that the generating series
    are a list of numpy arrays. If the input is a numpy array, it will iterate
    over the rows, therefore giving unexpected and erroneos results.
    """    
    if isinstance(gs, list):
        pass
    elif isinstance(gs, GeneratingSeries):
        gs = [gs]
    else:
        raise TypeError("Inputs needs to have type GeneratingSeries, or list.")
    
    return gs   
 

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
    
    # Calculate the shuffle product of the first two args.
    shuff_obj = BinaryShuffle()
    output_gs = []
    gs1, gs2 = wrap_generating_series(args[0]), wrap_generating_series(args[1])
    for gs1_term in gs1:
        for gs2_term in gs2:
            temp_output = shuff_obj(gs1_term, gs2_term)
            temp_output = collect(temp_output)
            output_gs.extend(temp_output)
        output_gs = collect(output_gs)
    output_gs = collect(output_gs)
    
    for gs1 in args[2:]:
        storage = []
        for gs2 in output_gs:
            temp_output = shuff_obj(gs1, gs2)
            temp_output = collect(temp_output)
            storage.extend(temp_output)
        output_gs = collect(storage)
    output_gs = collect(output_gs)
    
    return output_gs


def iterate_gs(g0, multiplier, n_shuffles, iter_depth=2, return_type=tuple):
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
    term_storage = defaultdict(list)
    term_storage[0].append(g0)
    # global inputs
    for depth in range(iter_depth):
        parts = partitions(depth, n_shuffles)
        for part in parts:
            inputs = []
            for index in part:
                inputs.append(term_storage[index])
            
            # Cartesian product of all the inputs, instead of nested for-loop
            for in_perm in product(*inputs):
                term_storage[depth+1].extend(nShuffles(*in_perm))
            
            term_storage[depth+1] = collect(term_storage[depth+1])
        
        # After the shuffles for this iteration's depth have been caluclated,
        # prepend the multiplier to each term.
        temp = []
        for gs_term in term_storage[depth+1]:
            gs_term = gs_term.prepend_multiplier(multiplier)
            temp.append(gs_term)
        term_storage[depth+1] = temp
        
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
            array = GeneratingSeries(np.array([
                gs[0,  1:],
                gs[1, :-1]           
            ]))
            tuple_form.append((gs[0, 0], array))
        
        return tuple_form
    else:
        raise TypeError("Invalid return type.")


if __name__ == "__main__":   
    x0 = 0
    x1 = 1
    a = 10
    b = 7
    
    g0 = np.array([
        [1,   x0],
        [a,   0]
    ])
    g0 = GeneratingSeries(g0)
    
    multiplier = np.array([
        [-b,   x0],
        [ a,   0]
    ])

    # bs = BinaryShuffle()
    
    # a = bs(g0, g0)
    # a[0].prepend_multiplier(multiplier)

    # import time
    
    # t0 = time.time()
    # scheme = iterate_gs(g0, multiplier, n_shuffles=2, iter_depth=5)
    # print(f"Process took: {time.time()-t0:.3f}s.")
    
    shuf_obj = BinaryShuffle()
    
    for i in range(3):
        shuff_dict = shuf_obj.__dict__
        g1a = shuf_obj(g0, g0)
        g1a_terms = []
        for g1a_term in g1a:
            g1a_terms.append(g1a_term.prepend_multiplier(multiplier))
        
        g2a = []
        for g1a_term in g1a_terms:   
            g2a.extend(shuf_obj(g1a_term, g0))
            g2a.extend(shuf_obj(g0, g1a_term))
        g2a = collect(g2a)
        
        g2a_terms = []
        for g2a_term in g2a:
            g2a_terms.append(g2a_term.prepend_multiplier(multiplier))       




  

