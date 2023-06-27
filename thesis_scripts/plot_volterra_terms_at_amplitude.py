# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 17:16:06 2023

@author: trist
"""

import dill

from params import plot, t, iter_depth

y_volt = []

A = 0.5

for i in range(1, iter_depth+2):
    with open(f"quad_cube_y{i}_lambdify_A_t_gen.txt", "rb") as f_read:
        func = dill.load(f_read)
        y_volt.append(func(A, t))

figax = plot(y_volt[0], None, "$y_1^v$")
figax = plot(y_volt[1], figax, "$y_2^v$")
figax = plot(y_volt[2], figax, "$y_3^v$")
figax = plot(y_volt[3], figax, "$y_4^v$")
figax = plot(y_volt[4], figax, "$y_5^v$")
figax = plot(y_volt[5], figax, "$y_6^v$")