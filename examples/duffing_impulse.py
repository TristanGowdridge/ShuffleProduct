import os
import sys
sys.path.insert(0, os.path.dirname(os.getcwd()) + "\shuffleproduct")

from math import sqrt

import numpy as np
import matplotlib.pyplot as plt

from shuffle import GeneratingSeries, iterate_gs
import responses as rsps

x0 = 0
x1 = 1
k1 = 10
k3 = 50
m = 10


g0 = GeneratingSeries(np.array([
    [         1/m,            x0, x1],
    [sqrt(k1 / m), -sqrt(k1 / m),  0]
]))

multiplier = np.array([
    [     -k3 / m,            x0, x0],
    [sqrt(k1 / m), -sqrt(k1 / m),  0]
])

scheme = iterate_gs(g0, multiplier, 3, iter_depth=2)

imp_response = rsps.impulse(scheme)
imp_response = rsps.array_to_fraction(imp_response)

imp_response_partfrac = rsps.matlab_partfrac(imp_response)

time_domain = rsps.inverse_lb(imp_response_partfrac)

t_func = rsps.time_function(time_domain)

t = np.linspace(0, 4, 1000)
y = t_func(t)

fig = plt.figure()
ax = fig.gca()
ax.plot(t, y)


