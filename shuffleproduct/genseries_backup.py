# -*- coding: utf-8 -*-
"""
Created on Fri May 12 18:04:01 2023

@author: trist
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 12 07:57:47 2023

@author: trist

"""
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
        Hash of all the terms except for the coefficient.
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