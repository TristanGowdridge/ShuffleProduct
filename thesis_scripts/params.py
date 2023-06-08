# -*- coding: utf-8 -*-
"""
Created on Mon Jun 5 11:38:48 2023

@author: trist
"""
import numpy as np
import matplotlib.pyplot as plt


def plot(y, figax=None, legend_label="y", **kwargs):
    # Plot the results
    if not figax:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.gca()
        ax.set_xlim(t_window)
        ax.set_xlabel('t')
        ax.set_ylabel('y')
        ax.grid(True)
    else:
        fig, ax = figax
    
    ax.plot(t, y, label=legend_label, **kwargs)
    ax.legend()

    return (fig, ax)


m = 1
c = 20
k1 = 1e4
k2 = 1e7
k3 = 5e9

A = 0.07

dr = c / (2 * np.sqrt(m * k1))
wn = np.sqrt(k1 / m)
wd = wn * np.sqrt(1 - dr ** 2)

# Time span
t_span = (0.0, 0.2)
t_window = (0.0, 0.2)
dt = 1e-4
t = np.arange(t_span[0], t_span[1], dt)

# Initial conditions
init_cond = np.array([0.0, 0.0])
