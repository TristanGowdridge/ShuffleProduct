# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 08:09:28 2023

@author: trist
"""

import numpy as np

from vci_quad import y1 as y1_volt
from vci_quad import y2 as y2_volt
from vci_quad import y3 as y3_volt

from rk_quad import y as y_rk

from params import plot, A

print("SCALING GENERATING SERIES")
y1_gen = np.load("y1_sym_quad.npy")
y2_gen = A * np.load("y2_sym_quad.npy")
y3_gen = A ** 2 * np.load("y3_sym_quad.npy")

figax = plot(y_rk, None, "Runge")

# figax = plot(y1_volt, figax, "$y_1^v$")
figax = plot(y1_volt + y2_volt, figax, "$y_1^v + y_2^v$")
figax = plot(y1_volt + y2_volt + y3_volt, figax, "$y_1^v + y_2^v + y_3^v$")

# figax = plot(y1_gen, figax, "$y_1^g$", c="y", linestyle="--")
figax = plot(y1_gen + y2_gen, figax, "$y_1^g + y_2^g$", linestyle="--")
figax = plot(
    y1_gen + y2_gen + y3_gen, figax, "$y_1^g + y_2^g + y_3^g$", linestyle="--"
)

fig, ax = figax
ax.set_title(
    f"Comparison of Quad with Dirac delta {A}"
)
