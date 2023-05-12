# -*- coding: utf-8 -*-
"""
Created on Tue May  9 13:44:58 2023

@author: trist
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.getcwd()) + "\shuffleproduct")

import numpy as np
import shuffle as shfl


x0 = 0
x1 = 1

k1 = 1
k2 = 1

multiplier = np.array([
    [-k2, x0],
    [ k1,  0]
])


g0 = shfl.GeneratingSeries(np.array([
    [ 1, x1],
    [k1,  0]
]))


scheme = shfl.iterate_gs(g0, multiplier, n_shuffles=2, iter_depth=2, return_type=list)

# def cos_gs(frequency, amplitude):
#     """
#     Creates the GeneratingSeries terms in array form for a cosine excitation.
#     """
#     terms = [
#         shfl.GeneratingSeries(np.array([
#             [ amplitude/2],
#             [frequency*1j]
#         ])),
#         shfl.GeneratingSeries(np.array([
#             [  amplitude/2],
#             [-frequency*1j]
#         ]))
#     ]
#     return terms












