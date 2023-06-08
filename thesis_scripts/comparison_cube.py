# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 08:20:36 2023

@author: trist
"""

import numpy as np

from vci_cube import y1 as y1_volt
from vci_cube import y3 as y3_volt

from rk_cube import y as y_rk

from params import plot, A


print("SCALING GENERATING SERIES")
y1_gen = np.load("y1_sym_cube.npy")
y2_gen = 0.01 * np.load("y2_sym_cube.npy")
y3_gen = 0.01**2 * np.load("y3_sym_cube.npy")

figax = plot(y1_volt, None, "$y_1^v$")
figax = plot(y1_volt + y3_volt, figax, "$y_1^v + y_3^v$")

figax = plot(y_rk, figax, "Runge")

figax = plot(y1_gen, figax, "$y_1^g$", c="y", linestyle="--")
figax = plot(y1_gen + y2_gen, figax, "$y_1^g + y_2^g$", linestyle="--")
figax = plot(
    y1_gen + y2_gen + y3_gen, figax, "$y_1^g + y_2^g + y_3^g$", linestyle="--"
)

fig, ax = figax
ax.set_title(
    f"Comparison of Cube with Dirac delta {A}"
)
