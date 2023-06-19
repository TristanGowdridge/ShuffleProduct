# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:30:39 2023

@author: trist
"""
import numpy as np
import pickle as pkl

from vci_quad_cube import y1 as y1_volt
from vci_quad_cube import y2 as y2_volt
from vci_quad_cube import y3 as y3_volt

from rk_quad_cube import y as y_rk

from params import plot, A


y1_gen = np.load("quad_cube_y1_gen_num.npy")
y2_gen = np.load("quad_cube_y2_gen_num.npy")
y3_gen = np.load("quad_cube_y3_gen_num.npy")
y4_gen = np.load("quad_cube_y4_gen_num.npy")
y5_gen = np.load("quad_cube_y5_gen_num.npy")

figax = plot(y_rk, None, "Runge")

# figax = plot(y1_volt, figax, "$y_1^v$")
# figax = plot(y1_volt + y2_volt, figax, "$y_1^v + y_2^v$")
# figax = plot(y1_volt + y2_volt + y3_volt, figax, "$y_1^v + y_2^v + y_3^v$")

# figax = plot(y1_gen, figax, "$y_1^g$", c="y", linestyle="--")
# figax = plot(y1_gen + y2_gen, figax, "$y_1^g + y_2^g$", linestyle="--")
# figax = plot(
#     y1_gen + y2_gen + y3_gen, figax, "$y_1^g + y_2^g + y_3^g$", linestyle="--"
# )
figax = plot(y5_gen, figax, "$y_5^g$")
figax = plot(
    y1_gen + y2_gen + y3_gen + y4_gen, figax, "$y_1^g + y_2^g + y_3^g + y_4^g$", linestyle="--"
)
figax = plot(
    y1_gen + y2_gen + y3_gen + y4_gen + y5_gen, figax, "$y_1^g + y_2^g + y_3^g + y_4^g + y_5^g$", linestyle="--"
)

fig, ax = figax
ax.set_title(
    f"Comparison of Duffing's equation solutions with Dirac delta {A}"
)
