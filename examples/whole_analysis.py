# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:12:17 2023

@author: trist
"""
import time

from sympy import symbols, lambdify

from params import A, m, c, k1, k2, k3, t
from shuffleproduct.auxilliary_funcs import plot

from exclusively_contour_integration.vci_quad_cube import y1 as y1_volt
from exclusively_contour_integration.vci_quad_cube import y2 as y2_volt
from exclusively_contour_integration.vci_quad_cube import y3 as y3_volt

import shuffleproduct.shuffle as shfl
from shuffleproduct.generating_series import GeneratingSeries as GS
from shuffleproduct.responses import convert_gs_to_time
from shuffleproduct.impulse import iterate_quad_cubic, impulsehere

t0 = time.perf_counter()
iter_depth = 2

# =============================================================================
# Symbolic
# =============================================================================
_k2, _k3, _x0, _x1, _a1, _a2, _A = symbols("k2 k3 x0 x1 a1 a2 A")

# Create the first generating series term
g0 = GS([
    [  1, _x0, _x1],
    [_a1, _a2,   0]
])

# Form the generating series multipliers
mult_quad = GS([
    [-_k2, _x0, _x0],
    [ _a1, _a2,   0]
])
mult_cube = GS([
    [-_k3, _x0, _x0],
    [ _a1, _a2,  0]
])
mults = [mult_quad, mult_cube]

# Apply the generating series iterative expansion
scheme = iterate_quad_cubic(g0, mults, iter_depth)

# Convert into the time domain
gs = impulsehere(scheme, _A, iter_depth)
y_gs = convert_gs_to_time(gs)

# Calculate response parameters
a1, a2 = shfl.sdof_roots(m, c, k1)

vals = {
    _A: A,
    _a1: a1,
    _a2: a2,
    _k2: k2,
    _k3: k3,
}

# Sub in the specific values
y_g = []
for i in range(iter_depth+1):
    y_g.append(lambdify(symbols('t'), sum(y_gs[i]).subs(vals))(t))


# =============================================================================
# Plotting
# =============================================================================
print(f"time taken for full calculation was {time.perf_counter()-t0:.2f}s.")

_figax = plot(t, y1_volt, None, "$y_1^v$")
_figax = plot(t, y2_volt, _figax, "$y_2^v$")
_figax = plot(t, y3_volt, _figax, "$y_3^v$")

for i in range(iter_depth+1):
    _figax = plot(t, y_g[i], _figax, f"$y^g_{i+1}$", linestyle="--")
