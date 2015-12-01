# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 14:17:46 2015

@author: kostas
"""
from __future__ import division
import numpy as np
from numpy import cos, sin, sqrt, pi
from scipy.integrate import odeint, quad
from scipy.optimize import brentq
from matplotlib import pyplot as plt
from numba import jit

from IPython.display import display, Math, Latex
from matplotlib import animation
from JSAnimation import IPython_display

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (10,8)

@jit
def d8ds(theta, s, phi, fx, fg=0.2):
    assert type(theta) == np.ndarray,\
    "Theta should be a numpy array"
    assert theta.ndim == 1,\
    "Theta should be a vector"
    assert type(s) and type(phi) and type(fx) == float,\
    "s, phi, fg and fx should be floats"
    
    d8_ds = np.zeros_like(theta)
    d8_ds[0] = theta[1]
    d8_ds[1] = s*fg*cos(theta[0]) + s*fx*cos(phi)*sin(theta[0])
    return d8_ds

@jit
def shoot(z0, s, theta, phi, fx, fg=0.2):
    #assert type(d8ds) == function,\
    #"d8ds (the derivative of theta with respect to s) must be a function"
    T8 = np.array([z0, 0.0])
    minz = odeint(d8ds, T8, s, args=(phi, fx, fg))
    #print z
    return minz[-1, 0] - theta


def hair_pos(R, L, fx, thL, phiL=[0.0]):
    thL = np.array(thL)
    if len(phiL) > 1:
        phiL = np.array(phiL)
        assert len(thL) == len(phiL),\
        "Phi and Theta must be of equal size"
        assert phiL.ndim == 1,\
        "Phi must be a vector"
    else:
        phiL = np.ones_like(thL) * phiL
    assert thL.ndim == 1,\
    "Theta must be a vector"
    N = 50
    h = L/float(N)
    s = np.linspace(0,L,N)
    x = np.zeros((len(thL),N))
    y = np.zeros((len(thL),N))
    z = np.zeros((len(thL),N))
    def dxds(theta, phi):
        return cos(theta) * cos(phi) + fx * sin(phi)
    def dyds(theta, phi):
        return -cos(theta) * sin(phi) + fx * cos(phi)
    def dzds(theta, phi):
        return sin(theta)
    
    for i in np.arange(len(thL)):
        msolve = brentq(shoot, thL[i]-2*pi,thL[i]+2*pi, args=(s, thL[i], phiL[i], fx, fg))
        #print msolve, thL[i], R
        thetas = odeint(d8ds, np.array([msolve, 0.0]), s, args=(phiL[i], fx, fg))
        x_start = R * cos(thL[i]) * cos(phiL[i])
        y_start = -R * cos(thL[i]) * sin(phiL[i])
        z_start = R * sin(thL[i])
        #print thetas[:, 0]
        #print x_start, y_start, z_start
        #xfun = lambda s: cos(thetas[:, 0]) * cos(phi) + fx * sin(phi)
        #print xfun(s)
        #yfun = lambda s: -cos(thetas[:, 0]) * sin(phi) + fx * cos(phi)
        #zfun = lambda s: sin(thetas[:, 0])
        x[i, :] = x_start
        y[i, :] = y_start
        z[i, :] = z_start
        for j in np.arange(1, N):
            x[i, j] = x[i, j-1] + h * dxds(thetas[-j, 0], phiL[i])
            y[i, j] = y[i, j-1] + h * dyds(thetas[-j, 0], phiL[i])
            z[i, j] = z[i, j-1] + h * dzds(thetas[-j, 0], phiL[i])
    return (x, y, z)


if __name__ == "__main__":
    R = 10.0
    L = 4.0
    fx = 0.2
    phi = 0.0
    hairs = 100
    fg = 0.2
    thL = np.linspace(0, pi, hairs)
    s = np.linspace(0, L, 100)[::-1]
    c = np.linspace(-R, R)
    x,y,z = hair_pos(R, L, fx, thL)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([-16, 16])
    ax.set_ylim([-10, 16])
    circ = plt.Circle((0, 0), radius=R, color='b', fill=False)
    ax.add_patch(circ)
    for i in np.arange(hairs):
        ax.plot(x[i, :], z[i, :])
        