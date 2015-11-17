# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 14:17:46 2015

@author: kostas
"""
from __future__ import division
import numpy as np
from numpy import cos, sin, exp, log
from matplotlib import pyplot as plt
from numba import jit

@jit
def rk3(A, bvector, y0, interval, N):
    """
    
    """
    # Assertions to check inputs
    assert np.shape(A) == np.shape(A.T),\
    "Please make sure 'A' is a square matrix"
    assert np.shape(y0) == (len(y0),),\
    "Check that y0 is a properly defined vector"
    assert np.shape(A)[0] == len(y0),\
    "Check the dimensions of y0 and A to ensure that A*y0 is possible"
    assert len(interval)==2, "Ensure that the interval has an upper and lower bound"
    
    x = np.linspace(interval[0], interval[1], N+1)
    h = (interval[1] - interval[0])/N
    y = np.zeros((len(y0), N+1))
    y[:,0] = y0
    for i in np.arange(len(x)-1):
        y1 = y[:, i] + h * (np.dot(A, y[:, i]) + bvector(x[i]))
        y2 = 0.75*y[:, i] + 0.25*y1 + 0.25*h * (np.dot(A, y1) + bvector(x[i+1]))
        y[:, i+1] = 1/3*y[:, i] + 2/3*y2 + 2/3*h * (np.dot(A, y2) + bvector(x[i+1]))
    return x, y


@jit
def bvector1(x):
    """
    
    """
    return np.zeros(len(y0))

@jit
def bvector2(x):
    """
    
    """
    b1 = cos(10*x) - 10*sin(10*x)
    b2 = 199*cos(10*x) - 10*sin(10*x)
    b3 = 208*cos(10*x) + 10**4*sin(10*x)
    return np.array([b1, b2, b3])

@jit
def dirk3(A, bvector, y0, interval, N):
    """
    """
    mu = 0.21132486540518708
    nu = 0.3660254037844386
    gam = 0.3169872981077807
    lam = 0.86602540378443871
    x = np.linspace(interval[0], interval[1], N+1)
    h = (interval[1] - interval[0])/N
    y = np.zeros((len(y0), N+1))
    y[:,0] = y0
    I = np.eye(len(A))
    for i in np.arange(len(x)-1):
        y1 = np.linalg.solve(I - h*mu*A, y[:, i] + h*mu*bvector(x[i] + h*mu))
        y2 = np.linalg.solve(I - h*mu*A, y1 + h*nu*(np.dot(A, y1) + bvector(x[i] + h*mu))\
                            + h * mu * bvector(x[i] + h*nu + 2*h*mu))
        y[:, i+1] = (1-lam) * y[:, i] + lam * y2 + h*gam*(np.dot(A, y2) + bvector(x[i] + h*nu + 2*h*mu))
    return x, y


@jit
def Yexact1(x):
    y1 = exp(-1000 * x)
    y2 = 1000/999 * (exp(-x) - y1)
    return np.array([y1, y2])

@jit
def Yexact2(x):
    y1 = cos(10*x) - exp(-x)
    y2 = cos(10*x) + exp(-x) - exp(-100*x)
    y3 = sin(10*x) + 2*exp(-x) - exp(-100*x) - exp(-10000*x)
    return np.array([y1, y2, y3])


if __name__ == "__main__":
    
    A = np.array([[-1000, 0],
                  [1000, -1]])
    interval=np.array([0, 0.1])
    y0 = np.array([1, 0])
    """
    A = np.array([[-1, 0, 0],
                  [-99, -100, 0],
                  [10098, 9900, -10000]])
    interval=np.array([0, 1])
    y0 = np.array([0, 1, 0])"""
    err = np.zeros(15)
    erri = np.zeros(15)
    h = np.zeros(15)
    for k in np.arange(1, 16):
        N = 40*(k)
        h[k-1] = (interval[1] - interval[0])/float(N)
        x, y = rk3(A[:], bvector, y0[:], interval[:], N)
        xi, yi = dirk3(A[:], bvector, y0[:], interval[:], N)
        y_exact = Yexact1(x)
        err[k-1] = h[k-1] * np.sum(np.abs((y[1, 1:] - y_exact[1, 1:])/y_exact[1, 1:]))
        erri[k-1] = h[k-1] * np.sum(np.abs((yi[1, 1:] - y_exact[1, 1:])/y_exact[1, 1:]))
    for k in np.arange(1, 16):
        N = 400*(k+1)
        h[k-1] = (interval[1] - interval[0])/float(N)
        x, y = rk3(A[:], bvector, y0[:], interval[:], N)
        xi, yi = dirk3(A[:], bvector, y0[:], interval[:], N)
        y_exact = Yexact1(x)
        err[k-1] = h[k-1] * np.sum(np.abs((y[1, 1:] - y_exact[1, 1:])/y_exact[1, 1:]))
        erri[k-1] = h[k-1] * np.sum(np.abs((yi[1, 1:] - y_exact[1, 1:])/y_exact[1, 1:]))
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = fig.add_subplot(111)
    ax1.plot(h, err)
    ax2.plot(h, erri)
    ax1.set_title("RK3")
    ax2.set_title("DIRK3")
    ax1.set_xlabel("h")
    ax2.set_xlabel("h")
    ax1.set_ylabel(r"\norm{Error}")
    ax2.set_ylabel(r"\norm{Error}")