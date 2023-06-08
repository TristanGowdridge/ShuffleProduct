# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 08:31:46 2023

@author: trist
"""

import numpy as np

from rk_quad_cube import y as y_rk

from params import plot, A


print("SCALING GENERATING SERIES")
y1_sym = np.load("y1_sym_quad_cube.npy")
y2_sym = 0.01 * np.load("y2_sym_quad_cube.npy")
y3_sym = 0.01**2 * np.load("y3_sym_quad_cube.npy")

y1_num = np.load("y1_num_quad_cube.npy")
y2_num = 0.01 * np.load("y2_num_quad_cube.npy")
y3_num = 0.01**2 * np.load("y3_num_quad_cube.npy")


figax = plot(y_rk, None, "Runge")

figax = plot(y1_sym, figax, "$y_1^s$", c="y", linestyle="--")
figax = plot(y1_sym + y2_sym, figax, "$y_1^s + y_2^s$", linestyle="--")
figax = plot(
    y1_sym + y2_sym + y3_sym, figax, "$y_1^s + y_2^s + y_3^s", linestyle="--"
)

figax = plot(y1_num, figax, "$y_1^n$", c="y", linestyle="--")
figax = plot(y1_num + y2_num, figax, "$y_1^n + y_2^n$", linestyle="--")
figax = plot(
    y1_num + y2_num + y3_num, figax, "$y_1^n + y_2^n + y_3^n", linestyle="--"
)

fig, ax = figax
ax.set_title(
    f"Comparison of numeric and symbolic quad cubbe with Dirac delta {A}"
)

print("y2 within bounds of numeric error:", all(abs(y2_sym - y2_num) < 1e-10))
print("y3 within bounds of numeric error:", all(abs(y3_sym - y3_num) < 1e-10))
print()