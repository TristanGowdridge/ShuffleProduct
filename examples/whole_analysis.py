# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:12:17 2023

@author: trist
"""
import time

from sympy import symbols, lambdify

from params import A, m, c, k1, k2, k3, t, iter_depth
from shuffleproduct.auxilliary_funcs import plot
from vci_quad_cube import y1 as y1_volt
from vci_quad_cube import y2 as y2_volt
from vci_quad_cube import y3 as y3_volt

import shuffleproduct.shuffle as shfl
from shuffleproduct.generating_series import GeneratingSeries as GS
from shuffleproduct.specific_implementation import iterate_quad_cubic, convert_gs_to_time

import pickle as pkl

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

# for i in range(iter_depth+1):
#     with open(f"quad_cube_y{i+1}_gen_sym.txt", "wb") as f_sym:
#         pkl.dump(y_gs[i], f_sym)
#     print(i)
#     temp = lambdify(symbols('t'), sum(y_gs[i]).subs(vals))(t)
#     np.save(f"quad_cube_y{i+1}_gen_num.npy", temp)

y1_g = lambdify(symbols('t'), sum(y_gs[0]).subs(vals))(t)
y2_g = lambdify(symbols('t'), sum(y_gs[1]).subs(vals))(t)  # iter_depth = 1
y3_g = lambdify(symbols('t'), sum(y_gs[2]).subs(vals))(t)  # iter_depth = 2
# y4_g = lambdify(symbols('t'), sum(y_gs[3]).subs(vals))(t)  # iter_depth = 3
# y5_g = lambdify(symbols('t'), y_gs[4].subs(vals))(t)  # iter_depth = 4
# y6_g = lambdify(symbols('t'), y_gs[5].subs(vals))(t)  # iter_depth = 5

# np.save(f"quad_cube_y3_k3_only_gen_num.npy", y3_g)


# for i in y_gs.values():
#     print(i)

# =============================================================================
# Plotting
# =============================================================================
print(f"time taken for full calculation was {time.perf_counter()-t0:.2f}s.")
_figax = plot(y1_volt, None, "$y_1^v$")
_figax = plot(y2_volt, _figax, "$y_2^v$")
_figax = plot(y3_volt, _figax, "$y_3^v$")

_figax = plot(y1_g, _figax, "$y^g_1$", linestyle="--")
_figax = plot(y2_g, _figax, "$y^g_2$", linestyle="--")
_figax = plot(y3_g, _figax, "$y^g_3$", linestyle="--")
# _figax = plot(y4_g, _figax, "$y^g_4$", linestyle="--")
# _figax = plot(y5_g, _figax, "$y^g_5$", linestyle="--")

# _figax = plot(y1_g + y2_g + y3_g, _figax, "$y gen 3$", linestyle="--")
# _figax = plot(y1_g + y2_g + y3_g + y4_g, _figax, "$y gen 4$", linestyle="--")
# _figax = plot(y1_g + y2_g + y3_g + y4_g + y5_g, _figax, "$y gen 5$", linestyle="--")

# _fig, _ax = _figax
# _ax.set_title(
#     f"Comparison of Duffing's equation solutions with Dirac delta {A}"
# )
