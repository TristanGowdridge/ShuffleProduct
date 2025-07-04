import time

import numpy as np
import matplotlib.pyplot as plt

import shuffleproduct.shuffle as shfl
import shuffleproduct.responses as rsps
from shuffle_product.specific_implementation import iterate_quad_cubic
from shuffleproduct.generating_series import GeneratingSeries
# import paper_funcs as pf

x0 = 0
x1 = 1

# Keiths Fave values
m = 1
c = 20
k1 = 1e4
k3 = 1e9
x_init = 0
v_init = 0
amplitude = 1  # No greater than 10.

a1, a2 = shfl.sdof_roots(m, c, k1)

g0 = GeneratingSeries([
    [ 1, x0, x1],
    [a1, a2,  0]
])

if x_init:
    g0.append(GeneratingSeries(np.array([
        [x_init, x0],
        [    a2,  0]
    ])))
    print("one of these should be x1")

if v_init - x_init * c:
    g0.append(shfl.GeneratingSeries(np.array([
        [v_init-x_init*c, x0],
        [             a1,  0]
    ])))
    print("one of these should be x1")

multiplier = [np.array([
    [-k3, x0, x0],
    [ a1, a2,  0]
])]


imp_response = iterate_quad_cubic(g0, multiplier, 2)

t0 = time.perf_counter()
imp_response_partfrac = rsps.matlab_partfrac(
    imp_response, precision=5
)

print(time.perf_counter() - t0)
time_domain = rsps.inverse_lb(imp_response_partfrac)
t_func = rsps.time_function(time_domain)

t = np.linspace(0, 1, 1000)
y = t_func(t)
fig = plt.figure()
ax = fig.gca()
ax.plot(t, y)
