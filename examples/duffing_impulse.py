import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.getcwd()) + r"\shuffleproduct")
import shuffle as shfl
import responses as rsps
from generating_series import GeneratingSeries

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


g0 = shfl.GeneratingSeries(np.array([
    [ 1, x0, x1],
    [a1, a2,  0]
]))

if x_init:
    g0.append(shfl.GeneratingSeries(np.array([
        [x_init, x0],
        [    a2,  0]
    ])))
print("one of these should be x1")
if v_init - x_init * c:
    g0.append(shfl.GeneratingSeries(np.array([
        [v_init-x_init*c, x0],
        [             a1,  0]
    ])))

multiplier = [np.array([
    [-k3, x0, x0],
    [ a1, a2,  0]
])]

imp_response = shfl.iterate_gs(
    g0, multiplier, n_shuffles=3, iter_depth=2
)
imp_response = rsps.impulse(imp_response)

t0 = time.perf_counter()
imp_response_partfrac = rsps.matlab_partfrac(
    imp_response, precision=5, delete_files=False
)
print(time.perf_counter() - t0)
time_domain = rsps.inverse_lb(imp_response_partfrac)
t_func = rsps.time_function(time_domain)

t = np.linspace(0, 1, 1000)
y = t_func(t)
fig = plt.figure()
ax = fig.gca()
ax.plot(t, y)
