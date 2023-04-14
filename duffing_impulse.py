from sympy import Symbol
import numpy as np
from math import sqrt
import shuffle as shuf

x0 = Symbol("x0")
x1 = Symbol("x1")
k1 = 10
k3 = 50
m = 10

g0 = np.array([
    [         1/m,            x0, x1],
    [sqrt(k1 / m), -sqrt(k1 / m),  0]
])

multiplier = np.array([
    [     -k3 / m,            x0, x0],
    [sqrt(k1 / m), -sqrt(k1 / m),  0]
])

scheme = optimised_impulse_iteration(g0, multiplier, 3, iteration_depth=3)
