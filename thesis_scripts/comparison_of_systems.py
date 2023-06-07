# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:30:39 2023

@author: trist
"""
import numpy as np

from grahams_impulse import y1 as y1_volt
from grahams_impulse import y2 as y2_volt
from grahams_impulse import y3 as y3_volt

from rk_of_grahams import y as y_rk

from params import plot, A

y1_gen = np.load("y11.npy")
y2_gen = 0.01 * np.load("y21.npy")
y3_gen = 0.01**2 * np.load("y31.npy")

figax = plot(y1_volt, None, "$y_1^v$")
figax = plot(y1_volt + y2_volt, figax, "$y_1^v + y_2^v$")
figax = plot(y1_volt + y2_volt + y3_volt, figax, "$y_1^v + y_2^v + y_3^v$")

figax = plot(y_rk, figax, "Runge")

figax = plot(y1_gen, figax, "$y_1^g$", c="y", linestyle="--")
figax = plot(y1_gen + y2_gen, figax, "$y_1^g + y_2^g$", linestyle="--")
figax = plot(y1_gen + y2_gen + y3_gen, figax, "$y_1^g + y_2^g + y_3^g$", linestyle="--")


fig, ax = figax
ax.set_title(
    f"Comparison of Duffing's equation solutions with Dirac delta {A}"
)
