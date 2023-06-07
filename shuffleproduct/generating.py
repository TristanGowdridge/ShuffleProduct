# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:48:37 2023

@author: trist
"""
import numpy as np
from collections import deque


class GeneratingSeries():
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
    
    def __repr__(self):
        return self.__str__()
        
    def __hash__(self):
        """
        Hash of all the terms except for the coefficient.
        """
        return hash(tuple(self.words)) + hash(tuple(self.dens))
    
    def __getitem__(self, index):
        if len(self) in (index+1, 1):
            return (self.words[-1],  0)
        else:
            return (self.words[index], self.dens[index+1])
    
    def __eq__(self, other_obj):
        """
        Check if everything other than the coefficient are the same. If you
        want to include the coefficient then use np.array_equal(gs_obj, b).
        """
        return hash(self) == hash(other_obj)
            
    def __len__(self):
        return len(self.words)
    
    def __str__(self):
        if len(self.dens) == len(self.words):
            return str(np.array([[self.coeff, *self.words], [*self.dens, 0]]))
        else:
            return f"coeff:{self.coeff}\nwords:{self.words}\ndens:{self.dens}"

    def prepend_multiplier(self, multiplier):
        """
        Rather than doing in place, I return here as I'm pretty sure doing it
        in place screws around with the referecing and hashes,
        """
        if isinstance(multiplier, np.ndarray):
            if multiplier.shape[1] == 1:
                self.coeff *= np.real(multiplier[0, 0])
            
            elif multiplier.shape[1] >= 2:
                self.coeff *= np.real(multiplier[0, 0])
                self.words.extendleft(multiplier[0, 1:])
                self.dens.extendleft(multiplier[1, :-1])
        
        if isinstance(multiplier, GeneratingSeries):
            self.coeff *= multiplier.coeff
            self.words.extendleft(multiplier.words)
            self.dens.extendleft(multiplier.dens)
            
    def get_array_form(self):
        numer = [self.coeff] + list(self.words)
        denom = list(self.den) + [0]
        
        return np.array([numer, denom])
