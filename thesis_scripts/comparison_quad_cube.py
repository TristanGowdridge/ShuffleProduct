# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:30:39 2023

@author: trist
"""
import numpy as np
import pickle as pkl
import dill
import matplotlib.pyplot as plt
import matplotlib as mpl

from vci_quad_cube import y1 as y1_volt
from vci_quad_cube import y2 as y2_volt
from vci_quad_cube import y3 as y3_volt

from rk_quad_cube import y as y_rk

from params import plot, A, t, FONTSIZE, t_window

iter_depth = 4
SAVE_PATH = "C:/Users/trist/Desktop/University of Sheffield/PhD/Year 3/Thesis/LaTex/Figures/"

FORMAT = ".pdf"

font = {'family': 'sans-serif',
        'size': 12}
mpl.rc('font', **font)
plt.style.use("classic")
LINEWIDTH = 2
TRANSPARENCY = 0.6


for i in range(1, iter_depth+2):
    with open(f"quad_cube_y{i}_lambdify_A_t_gen.txt", "rb") as f_read:
        func = dill.load(f_read)
        exec(f"y{i}_gen = func(A, t)")


fig0 = plt.figure(figsize=(10, 7))
axs0 = fig0.gca()

axs0.plot(
    t, y1_volt, label="$\sum^{1} y_i^{ci}$", linewidth=LINEWIDTH,
     c="pink"
)
axs0.plot(
    t, y1_volt+y2_volt, label="$\sum^{2} y_i^{ci}$", linewidth=LINEWIDTH,
    c="pink"
)
axs0.plot(
    t, y1_volt+y2_volt+y3_volt, label="$\sum^{3} y_i^{ci}$", linewidth=LINEWIDTH,
    c="pink"
)
axs0.plot(
    t, y1_gen, label=f"$\sum^{1} y_i^g$", linewidth=LINEWIDTH,
    alpha=TRANSPARENCY, linestyle="--", dashes=(10, 10)
)
axs0.plot(
    t, y1_gen+y2_gen, label=f"$\sum^{2} y_i^g$", linewidth=LINEWIDTH,
    alpha=TRANSPARENCY, linestyle="--", dashes=(10, 10)
)
axs0.plot(
    t, y1_gen+y2_gen+y3_gen, label=f"$\sum^{3} y_i^g$", linewidth=LINEWIDTH,
    alpha=TRANSPARENCY, linestyle="--", dashes=(10, 10)
)
    

fig0.patch.set_facecolor('white')


axs0.legend(
    loc=4, prop={"size": 12}, mode="expand", ncol=7, handletextpad=0.1
)
axs0.set_xlabel("Time $(s)$", fontsize=FONTSIZE)
axs0.set_ylabel("Amplitude $(m)$", fontsize=FONTSIZE)
axs0.set_xlim(t_window)
axs0.grid(True)

fig0.savefig(
    SAVE_PATH + "gs_comparison_contour_int" + FORMAT, dpi=500, bbox_inches="tight"
)



# figax = plot(y1_volt, None, "$y_1^v$", c="k")
# figax = plot(y1_volt + y2_volt, figax, "$y_1^v + y_2^v$", c="k")
# figax = plot(y1_volt + y2_volt + y3_volt, figax, "$y_1^v + y_2^v + y_3^v$", c="k")

# figax = plot(
#     y1_gen, figax, "$y_1^g$", linestyle="--", dashes=(5, 5), c="red"
# )
# figax = plot(
#     y1_gen + y2_gen, figax, "$y_1^g + y_2^g$", linestyle="--", dashes=(5, 5), c="orange"
# )
# figax = plot(
#     y1_gen + y2_gen + y3_gen, figax, "$y_1^g + y_2^g + y_3^g$", linestyle="--",
#     dashes=(5, 5), c="cyan"
# )
# figax = plot(y5_gen, figax, "$y_5^g$")
# figax = plot(
#     y1_gen + y2_gen + y3_gen + y4_gen, figax, "$y_1^g + y_2^g + y_3^g + y_4^g$", linestyle="--"
# )
# figax = plot(
#     y1_gen + y2_gen + y3_gen + y4_gen + y5_gen, figax, "$y_1^g + y_2^g + y_3^g + y_4^g + y_5^g$", linestyle="--"
# )

# fig, ax = figax
# ax.set_title(
#     f"Comparison of Duffing's equation solutions with Dirac delta {A}"
# )
