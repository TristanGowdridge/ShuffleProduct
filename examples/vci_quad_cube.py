# -*- coding: utf-8 -*-
"""
Created on Mon Jun 5 10:03:18 2023

@author: trist
"""
from numpy import exp, cos, sin, angle, abs

from examples.params import m, k2, k3, dr, wn, wd, A, t
from auxilliary_funcs import plot



y1 = A / (m * wd) * exp(-dr*wn*t) * sin(wd * t)

# =============================================================================
# y2
# =============================================================================
B = (-3*wd + 1j*dr*wn)*(-wd + 1j*dr*wn)
C = (3*wd + 1j*dr*wn)*(wd + 1j*dr*wn)

y2 = exp(-dr*wn*t)*cos(wd*t) - exp(-2*dr*wn*t)
y2 += abs(C)*exp(-dr*wn*t)*cos(wd*t+angle(C)) / (8*wd**2 + wn**2)
y2 -= abs(B)*exp(-2*dr*wn*t)*cos(2*wd*t+angle(B)) / (8*wd**2 + wn**2)
y2 *= (A**2 * k2) / (2 * m**3 * wd**2 * wn**2)


# =============================================================================
# y3
# =============================================================================
D = 24 * wd * dr**2 * wn**2 + 16j * wd**2 * dr * wn - 8j * dr**3 * wn**3
E = -96 * wd**3 - 24 * wd * dr**2 * wn**2 + 96j * wd**2 * dr * wn + 24j * dr**3 * wn**3
F = 96 * wd**3 + 48j * wd**2 * dr * wn + 24j * dr**3 * wn**3

y3_k3 = abs(D) * exp(-3*dr*wn*t) * cos(3*wd*t + angle(D))
y3_k3 += abs(E) * exp(-3*dr*wn*t) * cos(wd*t + angle(E))
y3_k3 += abs(F) * exp(-dr*wn*t) * cos(wd*t + angle(F))
y3_k3 *= (A**3 * k3)/((32 * m**4) * (dr*wn) * wd**3 * wn**2 * (12*wd**2 + 4*wn**2))


G = 17 * wd**3 * dr * wn - 7 * wd * dr**3 * wn**3 + 6j * wd**4 - 17j * wd**2 * dr**2 * wn**2
G += 1j * dr**4 * wn**4

H = -4 * wd * dr * wn - 3j * wd**2 + 1j * dr**2 * wn**2 - 1j * wn**2
J = 13 * wd**3 + 2 * wd * wn**2 + 5 * wd * dr**2 * wn**2 - 9j* wd**2 * dr * wn - 2j * dr * wn**3
J -= 1j * dr**3 * wn**3
K = 9 * wd**5 - 32 * wd**3 * dr**2 * wn**2 + 3*wd**3 * wn**2 + 7 * wd * dr**4 * wn**4
K += -5 * wd * dr**2 * wn**4 + 27j * wd**4 * dr * wn - 20j * wd**2 * dr**3 * wn**3
K += -1j * dr**3 * wn**5 + 1j * dr**5 * wn**5 + 1j*7*wd**2*dr*wn**3

y3_k2 = (
    ((A**3 * k2**2 * abs(G) * exp(-3*dr*wn*t)) / (
        8 * m**5 * wd**3 * wn**4 * (3*wd**2 + wn**2)*(8*wd**2 + wn**2)
    )) * cos(3*wd*t + angle(G))
)
y3_k2 += (
    ((A**3 * k2**2 * abs(H) * exp(-2*dr*wn*t)) / (
        2 * m**5 * wd**3 * wn**4 * (8*wd**2 + wn**2)
    )) * cos(2*wd*t + angle(H))
)
y3_k2 += (2*A**3 * k2**2 * dr * exp(-2*dr*wn*t)) / (m**5 * wd**2 * wn**3 * (8*wd**2 + wn**2))
y3_k2 += (
    ((A**3 * k2**2 * abs(J) * exp(-3*dr*wn*t)) / (
        8 * m**5 * dr * wd**3 * wn**5 * (8*wd**2 + wn**2)
    )) * cos(wd*t + angle(J))
)
y3_k2 -= (A**3 * k2**2 * exp(-dr*wn*t) / (4 * m**5 * dr * wd**2 * wn**5)) * cos(wd*t)
y3_k2 += (
    ((A**3 * k2**2 * abs(K) * exp(-dr*wn*t)) / (
        8 * m**5 * dr * wd**3 * wn**5 * (3*wd**2 + wn**2)*(8*wd**2 + wn**2)
    )) * cos(wd*t + angle(K))
)

y3 = y3_k2 + y3_k3


if __name__ == "__main__":
    # Plot the results
    figax = plot(y1, None, "y1")
    figax = plot(y1+y2, figax, "y1+y2")
    figax = plot(y1+y2+y3, figax, "y1+y2+y3")
    
    # fig, ax = figax
    # ax.set_title(
    #     f"Volterra of Duffing's equation with Dirac delta {A}"
    # )
