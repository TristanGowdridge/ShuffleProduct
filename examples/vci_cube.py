# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 08:17:44 2023

@author: trist
"""

from numpy import exp, cos, sin, angle, abs
from params import m, k3, dr, wn, wd, A, t
from shuffleproduct.auxilliary_funcs import plot


y1 = A / (m * wd) * exp(-dr*wn*t) * sin(wd * t)


# =============================================================================
# y3
# =============================================================================
D = 24 * wd * dr**2 * wn**2 + 16j * wd**2 * dr * wn - 8j * dr**3 * wn**3
E = -96 * wd**3 - 24 * wd * dr**2 * wn**2 + 96j * wd**2 * dr * wn + 24j * dr**3 * wn**3
F = 96 * wd**3 + 48j * wd**2 * dr * wn + 24j * dr**3 * wn**3

y3 = abs(D) * exp(-3*dr*wn*t) * cos(3*wd*t + angle(D))
y3 += abs(E) * exp(-3*dr*wn*t) * cos(wd*t + angle(E))
y3 += abs(F) * exp(-dr*wn*t) * cos(wd*t + angle(F))
y3 *= (A**3 * k3)/((32 * m**4) * (dr*wn) * wd**3 * wn**2 * (12*wd**2 + 4*wn**2))


G = 17 * wd**3 * dr * wn - 7 * wd * dr**3 * wn**3 + 6j * wd**4 - 17j * wd**2 * dr**2 * wn**2
G += 1j * dr**4 * wn**4

H = -4 * wd * dr * wn - 3j * wd**2 + 1j * dr**2 * wn**2 - 1j * wn**2
J = 13 * wd**3 + 2 * wd * wn**2 + 5 * wd * dr**2 * wn**2 - 9j* wd**2 * dr * wn - 2j * dr * wn**3
J -= 1j * dr**3 * wn**3
K = 9 * wd**5 - 32 * wd**3 * dr**2 * wn**2 + 3*wd**2 * wn**2 + 7 * wd * dr**4 * wn**4
K += -5 * wd * dr**2 * wn**4 + 27j * wd**4 * dr * wn - 20j * wd**2 * dr**3 * wn**3
K += -1j * dr**3 * wn**5 + 1j * dr**5 * wn**5


if __name__ == "__main__":
    # Plot the results
    figax = plot(y1, None, "y1")

    figax = plot(y1+y3, figax, "y1+y3")
    
    fig, ax = figax
    ax.set_title(
        f"Volterra of Duffing's equation with Dirac delta {A}"
    )
