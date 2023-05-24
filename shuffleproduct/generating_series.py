# -*- coding: utf-8 -*-
"""
Created on Fri May 12 07:57:47 2023

@author: trist

"""

import numpy as np


class GeneratingSeries(np.ndarray):
    # __slots__ = ("gs_hash")
    
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

    def __eq__(self, other_obj):
        """
        Check if everything other than the coefficient are the same. If you
        want to include the coefficient then use np.array_equal(gs_obj, b).
        """
        return hash(self) == hash(other_obj)
            
    def __len__(self):
        return self.shape[1]
        
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
            return GeneratingSeries(arr_copy)
        
        elif multiplier.shape[1] == 2:
            pre = np.zeros((2, 2), dtype=self.dtype)
            pre[0, 0] = multiplier[0, 0] * self[0, 0]
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

