# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 09:35:48 2022

@author: trist

Throughout this document I will abbreviate Generating Series to GS.
"""
from math import factorial, prod, comb
from collections import defaultdict
from itertools import product, permutations
import functools
import time

import numpy as np
import sympy as sym
from sympy import Symbol, apart
from sympy.core.numbers import Number as SympyNumber
from sympy.core.mul import Mul as SympyMul
from sympy.core.power import Pow as SympyPow
from sympy.core.symbol import Symbol as SympySymbol
from sympy.core.add import Add as SympyAdd
from sympy.functions.elementary.exponential import exp as sympyexp

# Global variable required for the time series terms.
t = Symbol('t')

def get_longest_task(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        output = func(*args, **kwargs)
        end = time.perf_counter()
        duration = end - start
        if duration > wrapper.duration:
            wrapper.duration = duration
            print(f"New longest function: {func.__name__}")
            print(f"New longest duration: {duration}")
            
        return output
    wrapper.duration=0
    
    return wrapper


def counter(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        print(f"{func.__name__} call count: {wrapper.count}.")
        
        return func(*args, **kwargs)
    wrapper.count = 0
    
    return wrapper


def cumulative_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        output = func(*args, **kwargs)
        end_time = time.perf_counter()
        wrapper.cumulative_time += (end_time - start_time)    
        print(f"Cum time of {func.__name__} = {wrapper.cumulative_time}s")
        
        return output
    wrapper.cumulative_time = 0
    
    return wrapper

# This is not done?
class CumulativeTime:
    def __init__(self, func):
        self.cumulative_time = 0
        self._func = func
        
    def __call__(self, *args, **kwargs):
        start_time = time.perf_counter()
        output = self._func(*args, **kwargs)
        end_time = time.perf_counter()
        self.cumulative_time += (end_time - start_time)    
        print(f"Cum time of {self._func.__name__} = {self.cumulative_time}s")
        return output
    
    def __del__(self):
        print("Object ")
    

class GS:
    def __init__(self, array):
        self.array = array
        
    def __hash__(self):
        return hash(self.array.tobytes())



class GeneratingSeries:
    """
    This is a data structure, used for storing relevant information about each
    generating series term, used in the Shuffle class.
    
    This is used a glorified dictionary, but it helps make the code clearer,
    and the constructor is easier than typing out a whole dictionary 
    initialisation everytime it is required.
    """
    
    def __init__(self, handedness, depth, array):
        """
        Parameters
        ----------------------------------
        handedness : str
            handedness indicates which generating series was reduced to get to
            this point. If handedness = 0, the LHS GS was reduced. If
            handedness = 1, the RHS GS was reduced.
        
        depth : int
            depth is the recursion depth.
        
        array : (2 x i) - np.array
            array stores the pieces of the generating series that will be later
            stacked together to create the output.
            
        path : str
            is the sum of all the previous handednesses to get to that point.
            For instance a path of 1001 would indicate that the following
            generating series reductions happened: RHS > LHS > LHS > RHS. E is
            used in the path to indicate that this is the end.
        """
        self.handedness = handedness
        self.depth = depth
        self.array = array
        self.path = ""
    
    
    def __str__(self):
        """
        Useful when debugging, to print the values stored in the class. 
        """
        message = f"""
            handedness = {self.handedness},
            depth = {self.depth}
            array = {self.array}
            path = {self.path}        
        """   
        return message
    
    
    def __repr__(self):
        return self.__str__()


    
class Shuffle:
    def __init__(self):
        """
        This creates some data structures used throughout the calculations.
        The [-1] index is used as a reference for the first calculation,
        indexed at 0. The [-1] index is deleted in _find_complete_paths(), as
        it is no longer required after this point, and an error is raised if
        [-1] is not deleted.
        """
        
        self.index = 0
        self.recursion_depth = 0
        self.term_storage = {
            -1 : GeneratingSeries("X", -1, np.array(None))
            }
        self.term_storage[-1].path = ""
    
    
    def return_gs(self, gs1, gs2):
        """
        Given two input generating series, this returns the shuffle product of
        the two generating series. The output is a list of numpy arrays, with
        each instance of the list being a generating series.
        """
        numerator = factorial(gs1.shape[1] + gs2.shape[1] - 2)
        denominator = factorial(gs1.shape[1] - 1) * factorial(gs2.shape[1] - 1)
        self.number_of_outputs = int(numerator / denominator)
        
        self.gs1 = gs1
        self.gs2 = gs2

        self._subshuffle(gs1, gs2)
            
        self._find_complete_paths()
        self._return_output_index()
        self._return_outputs()
        self._group_arrays()
        self._collect_like_terms()
        self._delete_copies()
        
        return self.output
    
       
    def _group_arrays(self):
        """
        This takes the list of list of arrays and merges all the arrays inside
        each list of arrays, to result in a list of arrays.
        
        Before this we have a list of n x (2 x 1) arrays, and after we have a
        list of (2 x n) arrays.
        """
        for term_index in range(self.number_of_outputs):
            self.output[term_index] = np.hstack(self.output[term_index])        
    
    
    def _delete_copies(self):
        """
        This deletes all of the "to_delete" instances marked by
        _collect_like_terms().
        """
        self.output = [gs for gs in self.output if not isinstance(gs, str)]
        
    
    def _collect_like_terms(self):
        """
        This collects all like-terms by adding the coefficients and marks the
        second instance as "to_delete".
        """
        
        for i1 in range(len(self.output) - 1):
            for i2 in range(i1 + 1, len(self.output)):
                gs1 = self.output[i1]
                gs2 = self.output[i2]
                is1_array = isinstance(gs1, np.ndarray)
                is2_array = isinstance(gs2, np.ndarray)
                if is1_array and is2_array:
                    if gs1.shape == gs2.shape:
                        compare_elementwise = (gs1 == gs2)
                        # The number at [0, 0], is the coefficient, these don't
                        # necessarily have to be the same. Therefore this is
                        # hardcoded to True.
                        compare_elementwise[0, 0] = True 
                        if compare_elementwise.all():
                            coeff = gs1[0, 0] + gs2[0, 0]
                            self.output[i1][0, 0] = coeff
                            self.output[i2] = "to_delete"
        
    
    def _increment_index(self):
        """
        Does what it says on the tin. The index is essentially a global
        variable, used store the generating series arrays in self.term_storage,
        as they are calculted. Index is similar to recursion depth, but index
        never decrements.
        """
        self.index += 1
    
    
    def _next_recursion_level(self, gs1, gs2):
        """
        This keeps track of the current recursion depth.
        """
        self.recursion_depth += 1
        self._subshuffle(gs1, gs2)
        self.recursion_depth -= 1
    
    
    def _update_path(self): #find_parent
        """
        This updates the path by searching for the parent, which will be the 
        most recent entry in term_storage that is one less than the depth of 
        the current one being anaylsed.
        
        When the parent is found, the current handedness is added to the
        previous path. This keeps track of all the terms that have been created
        and the shuffle that creates it.
        """
        how_many_before = 1
        while True:
            parent_depth = self.term_storage[self.index-how_many_before].depth
            current_depth = self.term_storage[self.index].depth
            
            if parent_depth == (current_depth - 1):
                path = self.term_storage[self.index-how_many_before].path
                handedness = self.term_storage[self.index].handedness
                self.term_storage[self.index].path = path + handedness
                break
            
            else:
                how_many_before += 1
    
    
    def _find_complete_paths(self): #shufindex part 1
        """
        This returns a list of all the full paths.
        """
        # The -1 index term can now be deleted, as this would cause issues
        # later in the program. This was only used as a reference up until now,
        # and had no meaning.
        del self.term_storage[-1]
        
        full_paths = []
        for term in self.term_storage.values():
            if term.path[-1] == "E":
                full_paths.append(term.path)
        
        self.full_paths = full_paths
    
        
    def _return_output_index(self): #shufindex part 2
        term_index = 0
        output_index = np.empty((self.number_of_outputs, 0)).tolist()
        for path in self.full_paths:
            for joutind in range(1, self.gs1.shape[1] + self.gs2.shape[1]):
                for index, gs_obj in self.term_storage.items():
                    if gs_obj.path == path[:joutind]:
                        output_index[term_index].append(index)
            term_index += 1
        
        self.output_index = output_index
    
    
    def _return_outputs(self): #shufindex part 3
        output = np.empty((self.number_of_outputs, 0)).tolist()
        for i in range(self.number_of_outputs):
            for j in range(len(self.output_index[0])-1, -1, -1):
                to_array = self.term_storage[(self.output_index[i][j])].array
                output[i].append(to_array)
    
        self.output = output

    # @counter
    # @get_longest_task
    # @cumulative_time
    # @functools.cache
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
        gs1_length, gs2_length = gs1.shape[1], gs2.shape[1]
        
        if gs1_length != 1 and gs2_length != 1:
            # This is to decrease the size of the first term.
            gs1_reduced = np.array([
                [gs1[0, gs1_length - 1]],
                [gs1[1, gs1_length - 1] + gs2[1, gs2_length - 1]]
            ])
            self.term_storage[self.index] = GeneratingSeries(
                handedness = "0",
                depth = self.recursion_depth,
                array = gs1_reduced
            )
            self._update_path()
            self._increment_index()
            self._next_recursion_level(gs1[:, :(gs1_length - 1)], gs2)
            
            # This is to decrease the size of the second term.
            gs2_reduced = np.array([
                [gs2[0, gs2_length - 1]],
                [gs1[1, gs1_length - 1] + gs2[1, gs2_length - 1]]
            ])
            self.term_storage[self.index] = GeneratingSeries(
                handedness = "1",
                depth = self.recursion_depth,
                array = gs2_reduced
            )
            self._update_path()
            self._increment_index()
            self._next_recursion_level(gs1, gs2[:, :(gs2_length - 1)])
            
            
        elif gs1_length != 1 and gs2_length == 1:
            gs1_reduced = np.array([
                [gs1[0, gs1_length - 1]],
                [gs1[1, gs1_length - 1] + gs2[1, 0]]
            ])
            self.term_storage[self.index] = GeneratingSeries(
                handedness = "0",
                depth = self.recursion_depth,
                array = gs1_reduced
            )
            self._update_path()
            self._increment_index()
            self._next_recursion_level(gs1[:, :(gs1_length - 1)], gs2)
            
            
        elif gs1_length == 1 and gs2_length != 1:
            gs2_reduced = np.array([
                [gs2[0, gs2_length - 1]],
                [gs1[1, 0] + gs2[1, gs2_length - 1]]
            ])
            
            self.term_storage[self.index] = GeneratingSeries(
                handedness = "1",
                depth = self.recursion_depth,
                array = gs2_reduced
            )
            self._update_path()
            self._increment_index()
            self._next_recursion_level(gs1, gs2[:, :(gs2_length - 1)])
            
            
        elif gs1_length == 1 and gs2_length == 1:
            final_term = np.array([
                [gs1[0, 0] * gs2[0, 0]],
                [gs1[1, 0] + gs2[1, 0]]
            ])
            
            self.term_storage[self.index] = GeneratingSeries(
                handedness = "E",
                depth = self.recursion_depth,
                array = final_term
            )
            self._update_path()
            self._increment_index()


def prepend_multiplier(multiplier, output):
    """
    This takes a list of generating series, called outputs and prepends the 
    multiplier.
    """
    # Ensure that numpy arrays are elements of a list, for looping over.
    output = wrap_numpy_array(output)
  
    if multiplier.shape[1] == 1:
        pre = np.array([
            [None, None],
            [None, None]
        ])
        for i in range(len(output)):
            output[i][0, 0] = multiplier[0, 0] * output[i][0, 0]
    
    elif multiplier.shape[1] == 2:
        pre = np.array([
            [None, None],
            [None, None]
        ])
        for i in range(len(output)):
            pre[0, 0] = multiplier[0, 0] * output[i][0, 0]
            pre[0, 1] = multiplier[0, 1]
            pre[1, 0] = multiplier[1, 0]
            pre[1, 1] = output[i][1, 0]
            output[i] = np.delete(output[i], 0, axis=1)
            output[i] = np.hstack((pre, output[i]))
    
    elif multiplier.shape[1] >= 3:
        for i in range(len(output)):
            pre = np.array([
                [None],
                [None]
            ])
            pre[0, 0] = multiplier[0, 0] * output[i][0, 0]
            pre[1, 0] = multiplier[1, 0]
            output[i][0, 0] = multiplier[0, -1]
            output[i] = np.hstack((pre, multiplier[:, 1:], output[i]))
    
    return output            


def collect_like_terms(output):
    """
    This collects all like-terms by adding the coefficients and marks the
    second instance as "to_delete".
    """
    
    for i1 in range(len(output) - 1):
        for i2 in range(i1 + 1, len(output)):
                gs1 = output[i1]
                gs2 = output[i2]
                is1_array = isinstance(gs1, np.ndarray)
                is2_array = isinstance(gs2, np.ndarray)
                if is1_array and is2_array:
                    if gs1.shape == gs2.shape:
                        compare_elementwise = (gs1 == gs2)
                        # The number at [0, 0], is the coefficient, these don't
                        # necessarily have to be the same. Therefore this is
                        # hardcoded to True.
                        compare_elementwise[0, 0] = True 
                        if compare_elementwise.all():
                            coeff = gs1[0, 0] + gs2[0, 0]
                            output[i1][0, 0] = coeff
                            output[i2] = "to_delete"
    return output


def delete_copies(output):
    """
    This deletes all of the "to_delete" instances marked by
    collect_like_terms().
    """
    return [gs for gs in output if not isinstance(gs, str)]


def collect(x):
    """
    This is just a composition of two functions that are commonly used
    together.
    """
    return delete_copies(collect_like_terms(x))
 
    
def wrap_numpy_array(gs):
    """
    This is used for nShuffles(), as it is assumed that the generating series
    are a list of numpy arrays. If the input is a numpy array, it will iterate
    over the rows, therefore giving unexpected and erroneos results.
    """    
    if isinstance(gs, list):
        pass
        
    elif isinstance(gs, np.ndarray):
        gs = [gs]
    
    else:
        raise TypeError("Inputs needs to have type np.ndarray, or list.")
    
    return gs   
 
    
# def _partition(number):
#     answer = []
#     answer.append((number, ))
#     for x in range(1, number):
#         for y in _partition(number - x):
#             answer.append(tuple(sorted((x, ) + y)))
    
#     return answer


# def partition(number, length):
#     """
#     This returns all the ordered partitions of a number of a certain length.
#     might be bad for repeated numbers i.e, (1, 1) or (2, 2, 2)...
#     """
#     return [part for part in _partition(number) if len(part) == length]


def partitions_new(iter_depth, number_of_shuffles):
    """
    Rather than coming up with a fancy way of generating all the partitions of
    a number, this uses the itertools' product function to calculate the
    Cartesian product, and then eliminate all the entries that don't sum to
    iter_depth. Therefore giving all the partitions.
    
    There is an edge case at iter_depth == 1, in this case all the permutations
    of 1 and padding zeros are output.
    """
    if iter_depth == 0:
        return [tuple([0] * number_of_shuffles)]
    
    if iter_depth == 1:
        return list(set(permutations([0] * (number_of_shuffles - 1) + [1])))
    
    partitions = list(product(range(iter_depth), repeat=number_of_shuffles))
    
    return [i for i in partitions if sum(i) == iter_depth]
    

def nShuffles(*args):
    """
    This takes variadic input, greater than 2 and outputs the shuffle product.
    """
    if len(args) < 2:
        raise IndexError("There needs to more than 2 inputs")
    
    # Calculate the shuffle product of the first two args.
    output_gs = []
    gs1, gs2 = wrap_numpy_array(args[0]), wrap_numpy_array(args[1])
    
    for gs1_term in gs1:
        for gs2_term in gs2:
            shuff_obj = Shuffle()
            temp_output = shuff_obj.return_gs(gs1_term, gs2_term)
            temp_output = collect(temp_output)
            output_gs.extend(temp_output)
    output_gs = collect(output_gs)
    
    for gs1 in args[2:]:
        storage = []
        for gs2 in output_gs:
            shuff_obj = Shuffle()
            temp_output = shuff_obj.return_gs(gs1, gs2)
            temp_output = collect(temp_output)
            storage.extend(temp_output)
        output_gs = collect(storage)
    
    output_gs = collect(output_gs)
    
    return output_gs


def iterate_gs(g0, multiplier, n_shuffles, iteration_depth=4,
                                               return_type=tuple):
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
    global inputs
    for iter_depth in range(iteration_depth):
        partitions = partitions_new(iter_depth, n_shuffles)
        for part in partitions:
            inputs = []
            for index in part:
                inputs.append(term_storage[index])
            # Cartesian product of all the inputs, instead of nested for-loop
            all_input_perms = product(*inputs)
            for in_perm in all_input_perms:
                term_storage[iter_depth + 1].extend(nShuffles(*in_perm))
            
            term_storage[iter_depth + 1]= collect(term_storage[iter_depth + 1])
            prepend_multiplier(multiplier, term_storage[iter_depth + 1])
            
    if return_type == dict:
        return dict(term_storage)
    elif return_type == list:
        # Unpack all the gs terms into a list
        return [i for gs in term_storage.values() for i in gs]
    elif return_type == tuple:
        return convert_to_tuple_from_dict(term_storage)
    else:
        raise TypeError("Invalid return type.")


def convert_to_tuple_from_list(output):
    """
    
    """
    output = wrap_numpy_array(output)
    to_return = []
    for gs in output:
        array = np.array([
            gs[0,  1:],
            gs[1, :-1]           
        ])
    
        to_return.append((gs[0, 0], array))
    
    return to_return 


def convert_to_tuple_from_dict(output):
    """
    
    """
    output = [i for gs in output.values() for i in gs]
    
    return convert_to_tuple_from_list(output)

# =============================================================================
# Responses
# =============================================================================

def calculate_response(gs, input_type):
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
        gs_response = gwn_response(gs)
    
    elif input_type == "impulse":
        gs_response = impulse_response(gs)
        
    fractions = array_to_fraction(gs_response)
    partial_fractions = return_partial_fractions(fractions)
    summer = convert_to_sum(partial_fractions)
    ts = inverse_lb(summer)
    
    return ts
    

def gwn_response(gs):
    """
    Calculates the GWN response given an input generating series. 
    """
    wrap_numpy_array(gs)
    
    for _i in gs:
        print(_i)
        
    raise(NotImplementedError)


def impulse_response(outputs):
    """
    Calculates the impulse response given an input generating series. This
    requires the tuple form as an input.
    """
    # Check for the type of input, if the input is a single numpy array,
    # enclose this is in a list, as it is going to be iterated over.
    x0 = Symbol("x0")
    x1 = Symbol("x1")
    
    outputs = wrap_numpy_array(outputs)
    
    all_outputs = []
    for gs in outputs:
        # This for loop over all the terms in the generating series numerator.
        all_outputs.append([1, []])
        for index, term in enumerate(gs[1][0, :]):
            if term == x0:
                array = np.array([
                    [gs[1][0, index]],
                    [gs[1][1, index]]
                ])
                all_outputs[-1][1].append(array)
                continue
            
            elif term == x1 and gs[1].shape[1] == 1:
                array = np.array([
                    [          1],
                    [gs[1][1, 0]]
                ])
                all_outputs[-1][1].append(array)
                all_outputs[-1][1] = all_outputs[-1][1][0]
                break

            elif term == x1 and all(gs[1][0, index:] == x1):
                remaining_xs = gs[1].shape[1] - index
                all_outputs[-1][0] /= factorial(remaining_xs)
# =============================================================================
# I've added x0 in the array term below, the paper says it should be a 1.
# =============================================================================
                array = np.array([
                    [             x0], 
                    [gs[1][1, index]]
                ])
                all_outputs[-1][1].append(array)
                all_outputs[-1][1] = np.hstack(all_outputs[-1][1])
                break
            
            elif term == x1:
                del all_outputs[-1]
                break
                
    return all_outputs
 
    
# def optimised_impulse_iteration(g0, multiplier, n_shuffles, iteration_depth=5,
#                                                return_type=tuple):
#     term_storage = defaultdict(list)
#     term_storage[0].append(g0)
#     global inputs
#     for iter_depth in range(iteration_depth):
#         partitions = partitions_new(iter_depth, n_shuffles)
#         for part in partitions:
#             inputs = []
#             for index in part:
#                 inputs.append(term_storage[index])
#             # Cartesian product of all the inputs, instead of nested for-loop
#             all_input_perms = product(*inputs)
#             for in_perm in all_input_perms:
#                 term_storage[iter_depth + 1].extend(nShuffles(*in_perm))
            
#             term_storage[iter_depth + 1] = collect(term_storage[iter_depth + 1])
#             prepend_multiplier(multiplier, term_storage[iter_depth + 1])
            
#             term_storage[iter_depth + 1] = impulse_response(term_storage[iter_depth + 1])
    
#     if return_type == dict:
#         return dict(term_storage)
#     elif return_type == list:
#         # Unpack all the gs terms into a list
#         return [i for gs in term_storage.values() for i in gs]
#     elif return_type == tuple:
#         return convert_to_tuple_from_dict(term_storage)
#     else:
#         raise TypeError("Invalid return type.")
  
        
def merge_impulse_response_terms(arrays):
    """
    This merges the 
    """
    temp = np.hstack(arrays[:-1])
    output = prepend_multiplier(arrays[-1], temp)
    
    return output[0]


# =============================================================================
# Converting to the time domain.
# =============================================================================
def array_to_fraction(gs):
    """
    This converts from the array form of the generating series to the fraction
    form, so they can be sovled by sympy's partial fraction calculator.
    """
    gs = wrap_numpy_array(gs)
    output_list = []
    x0 = Symbol("x0")
    for term in gs:
        numerator = prod(term[1][0, :])
        denominator = [(1 + i*x0) for i in term[1][1, :]]
        denominator = prod(denominator)
        
        output_list.append((term[0], numerator / denominator))
    
    return output_list


def return_partial_fractions(fractions):
    """
    Using Sympy's apart module, this calculates the partial fractions of the
    generating series. So the Laplace-Borel transform can be calculated.
    """
    fractions = wrap_numpy_array(fractions)
    output = []
    for fraction in fractions:
        output.append(apart(fraction))
    
    return output


def get_symbol(term):
    symbol = list(term.free_symbols)
    only_one_symbol = (len(symbol) == 1)

        
    return symbol[0], only_one_symbol


def get_exponent(term):
    if isinstance(term, (SympySymbol, SympyPow)):
        return sym.log(term).expand(force=True).as_coeff_Mul()
    else:
        raise TypeError("Needs to be a sympy.core.power.Pow object.")
 
    
def inverse_lb(sum_of_fractions):
    """
    Converts the partial fractions into the time domain.
    
    Assuming a term of the form a / (b + cx0).
    """
    ts = 0
    for term in sum_of_fractions.args:
        for form in [lb_unit, lb_polynomial, lb_exponential, lb_cosine]:
            term_ts = form(term)
            if term_ts:
                ts += term_ts
                break
        else:
            print(f"\nbad term = {term}\n")
            # raise TypeError(f"Term is of an unknown form.\n\n{term}")
        
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
    
    if not np.sign(coeff2) != np.sign(unit):
        return False
    
    # Handling the conversion of the laplace-borel transform.
    a = -coeff2
    n = -exponent
    if unit != 1:
        a /= unit
        coeff1 /= unit ** n
        unit = 1
    
    ts = 0
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


def convert_to_sum(fractions):
    """
    Takes a list of tuples with the coefficient in index 0 and the term in 
    index 1, and converts them into a SympyAdd type by summing each element
    in the list. 
    """
    y = 0
    for fraction in fractions:
        y += prod(fraction)
        
    return y


if __name__ == "__main__":       
    x0 = Symbol("x0")
    x1 = Symbol("x1")
    β = Symbol("β")
    α = Symbol("α")
    
    g0 = np.array([
        [1, x1],
        [α,  0]
    ])
    
    multiplier = np.array([
        [-β, x0],
        [ α,  0]
    ])
    
    scheme = iterate_gs(g0, multiplier, 4, iteration_depth=2)
    # imrep = impulse_response(scheme)
    # fractions = array_to_fraction(imrep)
    
    # pf = return_partial_fractions(fractions)
    