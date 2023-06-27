# -*- coding: utf-8 -*-
"""
Created on Fri May 12 07:57:47 2023

@author: trist

to do:
    * Split the numpy array into coeff, numerator and coefficients. This will
    make the data types easier to handle. Something like
    zip(arr[0, 1:], arr[1, :-1]) would do the trick.
"""
import abc
import copy
from collections import deque, defaultdict
import numpy as np
from sympy import symbols, Matrix


class GeneratingSeries:
    """
    Effectively a conditional inheritence based on the input as there are two
    forms that the generating series can be handled: numerically or
    symbolically. They have a completely different data-structure under the
    hood. The numeric form is a fair bit faster (owing to inheritence of numpy
    arrays directly) than the symbolic form so I don't want to entirely scrap
    this method entirely. The symbolic form is much more generalisable.
    
    Most of the action here is inside __init__ and __getattr__. I've manually
    had to call the dunder methods for the instance specifically. I suspect
    there is a metaclass solution to this to handle the dunder methods but this
    works fine for now.
    Pretty close to my problem: https://stackoverflow.com/questions/70723265/
    """
    
    def __init__(self, *args):
        if len(args) != 1:
            self.instance = GeneratingSeriesSym(args)
        
        elif isinstance(args[0], list):
            if np.asarray(args[0]).dtype != object:
                self.instance = GeneratingSeriesNum(args[0])
            else:
                self.instance = GeneratingSeriesSym(args[0])
        else:
            self.instance = GeneratingSeriesSym(*args)
            
    @property
    def __class__(self):
        """
        This means that when object of the GeneratingSeries class are called
        into isinstance(), the class in self.instance is included in this
        evaluation.
        """
        return self.instance.__class__
    
    # def shuffle_cacher(self):
    #     """
    #     Since wrappers are instatiated upon import, and the type of generating
    #     series isn't decided then, need to define a pass through shuffle cacher
    #     method.
    #     """
    #     return self.instance.shuffle_cacher()
    
    @property
    def n_excites(self):
        return self.instance.n_excites
    
    def __getattr__(self, name):
        return self.instance.__getattribute__(name)
    
    def get_end(self, gs):
        return self.instance.get_end(gs)
    
    def reduction_term(self, g1, g2):
        return self.instance.reduction_term(g1, g2)
    
    def __hash__(self):
        return self.instance.__hash__()

    def __eq__(self, other_obj):
        return self.instance.__eq__(other_obj)
            
    def __len__(self):
        return self.instance.__len__()

    def __repr__(self):
        return self.instance.__repr__()
        
    def __getitem__(self, index):
        return self.instance.__getitem__(index)

    def __str__(self):
        return self.instance.__str__()


class GS_Base(abc.ABC):
    """
    Base class to ensure that the two types have the same methods and
    attributes.
    """
    @property
    @abc.abstractmethod
    def n_excites(self):
        pass
    
    @abc.abstractmethod
    def prepend_multiplier():
        pass
    
    @abc.abstractmethod
    def get_end():
        pass
    
    @abc.abstractmethod
    def first_term():
        pass
    
    @abc.abstractmethod
    def get_term():
        pass
    
    @abc.abstractmethod
    def reduction_term():
        pass
    
    @abc.abstractmethod
    def add_to_stack():
        pass
    
    @abc.abstractmethod
    def handle_end():
        pass
    
    @abc.abstractmethod
    def collect():
        pass
   
    @abc.abstractmethod
    def handle_output_type():
        pass
        
    @abc.abstractmethod
    def collect_grid():
        pass
    
    @abc.abstractmethod
    def to_array():
        pass
    
    @abc.abstractmethod
    def get_words():
        pass
    
    @abc.abstractmethod
    def get_numer():
        pass
    
    
class GeneratingSeriesNum(np.ndarray, GS_Base):
    """
    The multiple inheritence isn't wokring here.
    """
    
    def __new__(cls, array):
        """
        Assumes an input with the coefficient in the index (0, 0) and also
        assumes (1, -1) is 0.
        """
        arr = np.asarray(array)
        
        if arr.dtype == float:
            arr = arr.astype(np.float64)
        elif arr.dtype == int:
            arr = arr.astype(np.int64)
        elif arr.dtype == complex:
            arr = arr.astype(np.complex64)
        
        return arr.view(cls)
    
    def __hash__(self):
        """
        Hash of all the terms except for the coefficient.
        """
        top = self[0, 1:]
        bottom = self[1, :]

        return hash(top.tobytes() + bottom.tobytes())

    def __len__(self):
        return self.shape[1]
    
    def __eq__(self, other_obj):
        return hash(self) == hash(other_obj)
    
    def get_coeff(self):
        return self[0, 0]
    
    def scale_coeff(self, scale):
        self[0, 0] *= scale
        
    def get_words(self):
        return 0, 1
    
    def get_numer(self):
        return self[0, 1:]
    
    @property
    def n_excites(self):
        return np.sum(self[0, 1:])
    
    def prepend_multiplier(self, multiplier):
        """
        Rather than doing in place, I return here as I'm pretty sure doing it
        in place screws around with the referecing and hashes,
        """
        # multiplier = multiplier.astype(self.dtype)
        
        if len(self) == 1:
            raise IndexError("Hmmm, very curious case.")
            
        if multiplier.shape[1] == 1:
            arr_copy = np.copy(self)
            arr_copy[0, 0] = arr_copy[0, 0] * multiplier[0, 0]
            return GeneratingSeriesNum(arr_copy)
        
        elif multiplier.shape[1] == 2:
            pre = np.zeros((2, 2), dtype=self.dtype)
            pre[0, 0] = multiplier[0, 0] * self[0, 0]
            pre[0, 1] = multiplier[0, 1]
            pre[1, 0] = multiplier[1, 0]
            pre[1, 1] = self[1, 0]
            temp = np.delete(self, 0, axis=1)
            
            return GeneratingSeriesNum(np.hstack((pre, temp)))
        
        elif multiplier.shape[1] >= 3:
            mult_copy = np.copy(multiplier)
            mult_copy[1, -1] = self[1, 0]
            mult_copy[0,  0] = self[0, 0] * mult_copy[0, 0]
            arr = np.delete(self, 0, 1)
            
            return GeneratingSeriesNum(np.hstack((mult_copy, arr)))
    
    def get_end(self, gs):
        end, gs = np.hsplit(gs, [1])
        length = len(gs)
        
        return end.reshape(-1), gs[:, ::-1].T, length
    
    def get_term(self, index):
        return self[index]
    
    def first_term(self, gs_reduct):
        return (1, gs_reduct)
    
    def reduction_term(self, g_reduce, *g_others):
        """
        Gets the term to append to the stack when reducing g1.
        """
    
        den_reduction = g_reduce[1]
        for g_other in g_others:
            den_reduction += g_other[1]
        
        reduction = np.array([
            [g_reduce[0]],
            [ den_reduction]
        ])
        
        return reduction
    
    def add_to_stack(self, grid_sec, count, new_term, current_stack):
        """
        appends the term to the stack and places it in then calls the function
        to collect the grid
        """
        # if len(new_term.shape) == 1:
        #     new_term.reshape(2, 1)
        
        grid_sec.append(
            (count, np.hstack([new_term, current_stack]))
        )
    
    def handle_end(self, grid, gs1_len, gs2_len, end1, end2, gs1, gs2):
        end = np.array([
            [end1[0] * end2[0]],
            [end1[1] + end2[1]]
        ])
        
        to_return = []
        for count, term in grid[(gs2_len, gs1_len)]:
            temp_term = np.hstack([end, term])
            temp_term[0, 0] *= count
            to_return.append(GeneratingSeriesNum(temp_term))
            
        return to_return
    
    def collect(self, output):
        """
        This collects all like-terms loops over the generating series in the
        output.
        """
        coefficient_count = defaultdict(int)
        term_storage = {}
        output_collected = []
        
        for gs in output:
            coefficient_count[hash(gs)] += gs[0, 0]
            term_storage[hash(gs)] = gs

        for term_hash, coeff in coefficient_count.items():
            temp = term_storage[term_hash]
            temp[0, 0] = coeff
            output_collected.append(temp)

        return output_collected
    
    def handle_output_type(self, term_storage, return_type):
        """
        Three output forms are given. The dictionary output gives the most
        stucture, where the keys represent generating series terms specific to
        an iteration depth. The list output simply returns a list of all the
        generating series, whilst they do appear in order, nothing breaks the
        order apart (unlike the dictionary). The tuple output is the form
        required for converting the generating series into the time domain. A
        function in the responses module converts the generating series array
        form into a fractional form.
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
                array = GeneratingSeriesNum([
                    gs[0,  1:],
                    gs[1, :-1]
                ])
                tuple_form.append((gs[0, 0], array))

            return tuple_form
        else:
            raise TypeError("Invalid return type.")
            
    def collect_grid(self, terms):
        """
        
        """
        instance_counter = defaultdict(int)
        term_storage = dict()
        
        for count, term in terms:
            gs_hash = hash(term.tobytes())
            instance_counter[gs_hash] += count
            if gs_hash not in term_storage:
                term_storage[gs_hash] = term
        
        collected_terms = []
        for key, term in term_storage.items():
            temp_term = (instance_counter[key], term)
            collected_terms.append(temp_term)
        
        return collected_terms
 
           
class GeneratingSeriesSym(GS_Base):
    __slots__ = ("coeff", "words", "dens")
    
    def __init__(self, *args):
        if len(args) == 1:
            """
            Legacy code for array form.
            """
            self.coeff = np.real(args[0][0][0])
            self.words = deque(np.real(args[0][0][1:]))
            self.dens = deque(args[0][1][:-1])
        
        elif len(args) == 3:
            """
            Better form
            """
            self.coeff = args[0]
            self.words = deque(args[1])
            self.dens = deque(args[2])
                
        elif len(args) == 2:
            self.coeff = args[0]
            self.words = deque(args[1])
            self.dens = deque()
    
    @property
    def n_excites(self):
        count = 0
        x1 = symbols("x1")
        for word in self.words:
            if word == x1:
                count += 1
        return count
    
    def __repr__(self):
        """
        Makes use of sympy's printing should be implemented.
        """
        # return Matrix(self)
        return self.__str__()
    

    def __hash__(self):
        """
        Hash of all the terms except for the coefficient.
        """

        return hash(tuple(self.words)) + hash(tuple(self.dens))
    
    def __eq__(self, other_obj):
        return hash(self) == hash(other_obj)
    
    def __getitem__(self, index):
        if len(self) in (index+1, 1):
            return (self.words[-1],  0)
        else:
            return (self.words[index], self.dens[index+1])
            
    def __len__(self):
        return len(self.words)
    
    def __str__(self):
        if len(self.dens) == len(self.words):
            return str(np.array([[self.coeff, *self.words], [*self.dens, 0]]))
        else:
            return f"coeff:{self.coeff}\nwords:{self.words}\ndens:{self.dens}"
 
    def get_words(self):
        return symbols("x0 x1")
    
    def get_numer(self):
        return self.words
    
    def get_coeff(self):
        return self.coeff
    
    def scale_coeff(self, scale):
        self.coeff *= scale
        
    def prepend_multiplier(self, multiplier):
        """
        Rather than doing in place, I return here as I'm pretty sure doing it
        in place screws around with the referecing and hashes,
        
        probably doesn't have to be copy.deepcopy and the casting to float on
        the coeff could be an issue.
        """
        if isinstance(multiplier, np.ndarray):
            if multiplier.shape[1] == 1:
                self.coeff *= multiplier[0, 0]
            
            elif multiplier.shape[1] >= 2:
                self.coeff *= multiplier[0, 0]
                self.words.extendleft(multiplier[0, 1:])
                self.dens.extendleft(multiplier[1, :-1])
        
        elif isinstance(multiplier, GeneratingSeriesSym):
            self.coeff *= multiplier.coeff
            self.words.extendleft(multiplier.words)
            self.dens.extendleft(multiplier.dens)
            
        else:
            raise TypeError("Unknown multiplier type")
           
    def get_array_form(self):
        numer = [self.coeff] + list(self.words)
        denom = list(self.den) + [0]
        
        return np.array([numer, denom])
    
    def get_term(self, index):
        return self[len(self)-index-1]
    
    def first_term(self, gs_reduct):
        return (1, GeneratingSeriesSym(1, [gs_reduct[0]]))
    
    def reduction_term(self, g_reduce, *g_others):
        """
        Gets the term to append to the stack when reducing g1.
        """
        den_reduction = g_reduce[1]
        for g_other in g_others:
            den_reduction += g_other[1]
            
        return (g_reduce[0], den_reduction)
    
    def add_to_stack(self, grid_sec, count, new_term, current_stack):
        """
        appends the term to the stack and places it in then calls the function
        to collect the grid
        """
        current_stack = copy.deepcopy(current_stack)
        current_stack.words.appendleft(new_term[0])
        current_stack.dens.appendleft(new_term[1])
        
        grid_sec.append((count, current_stack))
        
    def get_end(self, gs):
        return (None, gs.dens[0]), gs, len(gs)
    
    def handle_end(self, grid, gs1_len, gs2_len, end1, end2, gs1, gs2):
        to_return = []
        for count, term in grid[(gs2_len, gs1_len)]:
            term.coeff *= count * gs1.coeff * gs2.coeff
            term.dens.appendleft(gs1.dens[0] + gs2.dens[0])
            to_return.append(term)
            
        return to_return
            
    def collect(self, output):
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
    
    def handle_output_type(self, term_storage, return_type):
        """
        Three output forms are given. The dictionary output gives the most
        stucture, where the keys represent generating series terms specific to
        an iteration depth. The list output simply returns a list of all the
        generating series, whilst they do appear in order, nothing breaks the
        order apart (unlike the dictionary). The tuple output is the form
        required for converting the generating series into the time domain. A
        function in the responses module converts the generating series array
        form into a fractional form.
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
    
    def collect_grid(self, terms):
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
    
    def to_array(self):
        return np.array([self.coeff, *self.words], [*self.dens, 0])
     

if __name__ == "__main__":
    import sympy as sym
    ob = sym.symbols("a")
    num_l = [[0,0,1,0],[0,0,0,0]]
    num_gs = GeneratingSeries(num_l)