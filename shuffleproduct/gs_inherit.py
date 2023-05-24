# -*- coding: utf-8 -*-
"""
Created on Tue May 23 19:50:49 2023

@author: trist
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 12 07:57:47 2023

@author: trist

"""
import numpy as np


class GeneratingSeries(np.ndarray):
    __slots__ = ("gs_hash")
    
    def __new__(cls, array):
        return np.asarray(array).view(cls)
               
    def __hash__(self):
        """
        Hash of all the terms except for the coefficient.
        """
        top = self[0, 1:]
        bottom = self[1, :]

        return hash(top.tobytes() + bottom.tobytes())
    
    def prepend_multiplier(self, multiplier):
        """
        Rather than doing in place, I return here as I'm pretty sure doing it
        in place screws around with the referecing and hashes,
        """
        multiplier = multiplier.astype(self.array.dtype)
        
        if len(self) == 1:
            raise IndexError("Hmmm, very curious case.")
            
        if multiplier.shape[1] == 1:
            arr_copy = np.copy(self.array)
            arr_copy[0, 0] = arr_copy[0, 0] * multiplier[0, 0]
            return GeneratingSeries(arr_copy)
        
        elif multiplier.shape[1] == 2:
            pre = np.zeros((2, 2), dtype=self.dtype)
            pre[0, 0] = multiplier[0, 0] * self.array[0, 0]
            pre[0, 1] = multiplier[0, 1]
            pre[1, 0] = multiplier[1, 0]
            pre[1, 1] = self[1, 0]
            temp = np.delete(self, 0, axis=1)
            
            return GeneratingSeries(np.hstack((pre, temp)))
        
        elif multiplier.shape[1] >= 3:
            mult_copy = np.copy(multiplier)
            mult_copy[1, -1] = self[1, 0]
            mult_copy[0,  0] = self[0, 0] * mult_copy[0, 0]
            arr = np.delete(self, 0, 1)
            
            return GeneratingSeries(np.hstack((mult_copy, arr)))
            
    def __len__(self):
        return self.shape[1]
          
    def __eq__(self, other_obj):
        """
        Check if everything other than the coefficient are the same.
        """
        if not isinstance(other_obj, GeneratingSeries):
            return False
        
        return hash(self) == hash(other_obj)
           
    def hard_equals(self, other_obj):
        """
        This is used for unit testing, when we want to include the coefficient
        in the comparison.
        """
        return np.array_equal(self, other_obj)

    def __add__(self, other_obj):
        if not (self == other_obj):
            raise ValueError("Cannot add different Generating Series.")
        else:
            return self[0, 0] + other_obj[0, 0]
    
    def _fast_iadd(self, other_obj):
        """
        Used in functions that satisfy the conditions, could give errors if
        not careful.
        """
        self[0, 0] = self[0, 0] + other_obj[0, 0]
        
        return self
        
    def __iadd__(self, other_obj):
        if not isinstance(other_obj, GeneratingSeries):
            raise TypeError("Cannot add GeneratingSeries to another type.")
        
        if not self == other_obj:
            raise ValueError("Cannot add different Generating Series.")
        else:
            self._fast_iadd(other_obj)
            
        return self