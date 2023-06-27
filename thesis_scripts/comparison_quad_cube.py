# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:30:39 2023

@author: trist
"""
import numpy as np
import pickle as pkl
import dill

from vci_quad_cube import y1 as y1_volt
from vci_quad_cube import y2 as y2_volt
from vci_quad_cube import y3 as y3_volt

from rk_quad_cube import y as y_rk

from params import plot, A, t

iter_depth = 4


for i in range(1, iter_depth+2):
    with open(f"quad_cube_y{i}_lambdify_A_t_gen.txt", "rb") as f_read:
        func = dill.load(f_read)
        exec(f"y{i}_gen = func(A, t)")

# figax = plot(y_rk, None, "Runge")

figax = plot(y1_volt, None, "$y_1^v$", c="k")
figax = plot(y1_volt + y2_volt, figax, "$y_1^v + y_2^v$", c="k")
figax = plot(y1_volt + y2_volt + y3_gen, figax, "$y_1^v + y_2^v + y_3^v$", c="k")

figax = plot(
    y1_gen, figax, "$y_1^g$", linestyle="--", dashes=(5, 5), c="red"
)
figax = plot(
    y1_gen + y2_gen, figax, "$y_1^g + y_2^g$", linestyle="--", dashes=(5, 5), c="orange"
)
figax = plot(
    y1_gen + y2_gen + y3_gen, figax, "$y_1^g + y_2^g + y_3^g$", linestyle="--",
    dashes=(5, 5), c="cyan"
)
# figax = plot(y5_gen, figax, "$y_5^g$")
# figax = plot(
#     y1_gen + y2_gen + y3_gen + y4_gen, figax, "$y_1^g + y_2^g + y_3^g + y_4^g$", linestyle="--"
# )
# figax = plot(
#     y1_gen + y2_gen + y3_gen + y4_gen + y5_gen, figax, "$y_1^g + y_2^g + y_3^g + y_4^g + y_5^g$", linestyle="--"
# )

fig, ax = figax
ax.set_title(
    f"Comparison of Duffing's equation solutions with Dirac delta {A}"
)
