# -*- coding: utf-8 -*-
"""
Created on Thu May  4 17:44:13 2023

@author: trist
"""
import shuffleproduct.responses as rsps
import shuffleproduct.shuffle as shfl
from shuffleproduct.generating_series import GeneratingSeries

x0 = 0
x1 = 1

multipliers = [
    GeneratingSeries([
        [1/3],
        [  0]
    ]),
    GeneratingSeries([
        [-1/3, x0],
        [   1,  0]
    ])
]

g0 = GeneratingSeries([
    [1, x1],
    [1,  0]
])

imp = shfl.impulse_from_iter(
    g0, multipliers, n_shuffles=3, iter_depth=4, amplitude=0.63
)


imp_pf = rsps.sympy_partfrac_here(imp)


# imp_time_domain = rsps.inverse_lb(imp_pf)
# imp_func = rsps.time_function(imp_time_domain)


# t = Symbol("t")
# time_vec = np.linspace(0, 3, 1000)
# y_imp = imp_func(time_vec)


# fig1 = plt.figure()
# ax1 = fig1.gca()
# fig1.suptitle("Impulse")
# ax1.plot(time_vec, y_imp)
