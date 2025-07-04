# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 16:52:55 2025

@author: trist
"""
from sympy import exp, symbols, arg, Abs, cos, simplify, sqrt, lambdify
from vci_quad_cube import y3_k3

from params import A, k1, k3, c, m, t

import matplotlib.pyplot as plt

_k3, _a1, _a2, _A, _t = symbols("k3 a1 a2 A t")

gs_y3_k3 = -3*_A**3*_k3*exp(_t*(_a1 + 2*_a2))/(2*_a2*(_a1 - _a2)**3*(_a1 + _a2)) - _A**3*_k3*exp(3*_a2*_t)/(2*_a2*(_a1 - 3*_a2)*(_a1 - _a2)**3) - _A**3*_k3*exp(3*_a1*_t)/(2*_a1*(_a1 - _a2)**3*(3*_a1 - _a2)) + 3*_A**3*_k3*exp(_t*(2*_a1 + _a2))/(2*_a1*(_a1 - _a2)**3*(_a1 + _a2)) + 3*_A**3*_k3*exp(_a2*_t)/(2*_a1*_a2*(_a1 - _a2)*(_a1 + _a2)*(3*_a1 - _a2)) + 3*_A**3*_k3*exp(_a1*_t)/(2*_a1*_a2*(_a1 - 3*_a2)*(_a1 - _a2)*(_a1 + _a2))

_wd, _dr, _wn, _m, _k1, _c = symbols("\omega_d \zeta \omega_n m k_1 c")

_D = 24*_wd*_dr**2*_wn**2    +     16j*_wd**2*_dr*_wn    -     8j*_dr**3*_wn**3
_E = -96*_wd**3   -   24*_wd*_dr**2*_wn**2   +   96j*_wd**2*_dr*_wn   +   24j*_dr**3*_wn**3


_F = 96 * _wd**3   +   48j*_wd**2*_dr*_wn   +   24j*_dr**3*_wn**3

ci_y3_k3_1 = Abs(_D) * exp(-3*_dr*_wn*_t) * cos(3*_wd*_t + arg(_D))
ci_y3_k3_2 = Abs(_E) * exp(-3*_dr*_wn*_t) * cos(_wd*_t + arg(_E))
ci_y3_k3_3 = Abs(_F) * exp(-_dr*_wn*_t) * cos(_wd*_t + arg(_F))
ci_coeff = (_A**3 * _k3)/((32 * _m**4) * (_dr*_wn) * _wd**3 * _wn**2 * (12*_wd**2 + 4*_wn**2))

ci_terms_sym = (
    ci_y3_k3_1 * ci_coeff,
    ci_y3_k3_2 * ci_coeff,
    ci_y3_k3_3 * ci_coeff
)


a1_val = (-_dr + sqrt(_dr**2 - 1)) * _wn
a2_val = (-_dr - sqrt(_dr**2 - 1)) * _wn



subbed_gs = gs_y3_k3.subs({_a1: a1_val, _a2: a2_val})
subbed_gs = subbed_gs.make_args(subbed_gs)

not_subbed_gs = sum(gs_y3_k3.make_args(gs_y3_k3)[:-2])

subbed_gs = [simplify(i) for i in subbed_gs]

subbed_gs = [
    simplify(subbed_gs[0] + subbed_gs[3]),
    simplify(subbed_gs[1] + subbed_gs[2]),
    simplify(subbed_gs[4] + subbed_gs[5])
]

wd_sym = _wn * sqrt(1 - _dr**2)
wn_sym = sqrt(_k1 / _m)
zeta_sym = _c / (2*sqrt(_m*_k1))

zeta_val = zeta_sym.subs({_c:c, _k1:k1, _m:m})
wn_val = wn_sym.subs({_k1:k1, _m:m})
wd_val = wd_sym.subs({_wn:wn_val, _dr:zeta_val})

gs_terms = [i.subs({_dr: zeta_val, _k3:k3, _wn:wn_val, _A:A}) for i in subbed_gs]

ci_terms = [i.subs({_dr: zeta_val, _k3:k3, _wn:wn_val, _A:A, _wd:wd_val, _m:m}) for i in ci_terms_sym]


ci_vecs = []
for ci_term in ci_terms:
    ci_vecs.append(lambdify(_t, ci_term)(t))

gs_vecs = []
for gs_term in gs_terms:
    gs_vecs.append(lambdify(_t, gs_term)(t))



fig, axs = plt.subplots(3, 3)

for i in range(3):
    for j in range(3):
        axs[i, j].plot(ci_vecs[i] - gs_vecs[j])

# mismatch is between the terms ci_vecs[2] and gs_vecs[2]

