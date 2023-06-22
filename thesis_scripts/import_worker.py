# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 09:02:35 2023

@author: trist
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.getcwd()) + r"/shuffleproduct")
import responses as rsps


def worker(L, term):
    rsps.is_exponential_form(term)
    temp = rsps.lb_exponential(term)
    
    L.append(temp)