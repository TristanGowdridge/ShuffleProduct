# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 10:37:37 2025

@author: trist
"""

import time

from sympy import symbols, lambdify

from params import A, m, c, k1, k2, k3, t

import matplotlib.pyplot as plt

# =============================================================================
# import shuffleproduct.shuffle as shfl
# from shuffleproduct.generating_series import GeneratingSeries as GS
# from shuffleproduct.specific_implementation import iterate_quad_cubic, convert_gs_to_time
# =============================================================================
import shuffle as shfl
from generating_series import GeneratingSeries as GS
from specific_implementation import iterate_quad_cubic, convert_gs_to_time

t0 = time.perf_counter()
iter_depth = 2

# =============================================================================
# Symbolic
# =============================================================================
_k2, _k3, _x0, _x1, _a1, _a2, _A = symbols("k2 k3 x0 x1 a1 a2 A")

g0 = GS([
    [  1, _x0, _x1],
    [_a1, _a2,   0]
])

mult_quad = GS([
    [-_k2, _x0, _x0],
    [ _a1, _a2,   0]
])
mult_cube = GS([
    [-_k3, _x0, _x0],
    [ _a1, _a2,  0]
])

mults = [mult_quad, mult_cube]

scheme = iterate_quad_cubic(g0, mults, iter_depth)
y_gs = convert_gs_to_time(scheme, _A, iter_depth)


a1, a2 = shfl.sdof_roots(m, c, k1)

vals = {
    _A: A,
    _a1: a1,
    _a2: a2,
    _k2: k2,
    _k3: k3,
}


y1_g = lambdify(symbols('t'), sum(y_gs[0]).subs(vals))(t)
y2_g = lambdify(symbols('t'), sum(y_gs[1]).subs(vals))(t)  # iter_depth = 1
y3_g = lambdify(symbols('t'), sum(y_gs[2]).subs(vals))(t)  # iter_depth = 2


# =============================================================================
# Plotting
# =============================================================================
print(f"time taken for full calculation was {time.perf_counter()-t0:.2f}s.")

fig = plt.figure()
ax = fig.gca()
ax.plot(t, y1_g, linestyle="--")
ax.plot(t, y2_g, linestyle="--")
ax.plot(t, y3_g, linestyle="--")

