# -*- coding: utf-8 -*-
"""
Set of functions that solve initial value problems using two 3 step Runge-Kutta
methods (explicit and implicit). Two test cases are provided with a moderately 
stiff and a stiff problem.

Created on Sat Nov 14 2015

@author: Konstantinos Drakopoulos
"""
from __future__ import division
import numpy as np
from numpy import cos, sin, exp, log
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['figure.figsize'] = (12,6)
from numba import jit

#@jit
def rk3(A, bvector, y0, interval, N):
    """
    Function that returns a vector 'x' containing the points at the values in 'y'
    are obtained. 'y' contains on each row the solution to a first order ODE problem
    and it has a number of rows corresponding to the overal degree of the equation.
    'y' is computed here using a third-order Runge-Kutta method.
    Inputs:
        A           n x n numpy array
                Contains the coefficients used to solve the ODE system
        bvector     function, takes 1 argument
                Contains corrections or the RHS of the ODE system
        y0          n size vector
                Contains the initial guess for y
        interval    List of two elements
                Contains the upper and lower bounds
        N           integer
                Specifies the number of points to be used
    Otuputs:
        x           N+1 size vector
                Contains the points at which the RK method is evaluated
        y           N+1 size vector
                Contains the results of RK evaluations
    """
    # Assertions to check inputs
    assert type(A) == np.ndarray, "Ensure A has been defined as a numpy array."
    assert type(N) == int or np.int64, "Ensure N is an integer."
    assert N > 1, "N must be larger than 0"
    assert np.shape(A) == np.shape(A.T),\
    "Please make sure 'A' is a square matrix"
    assert np.shape(y0) == (len(y0),),\
    "Check that y0 is a properly defined vector"
    assert np.shape(A)[0] == len(y0),\
    "Check the dimensions of y0 and A to ensure that A*y0 is possible"
    assert len(interval)==2, "Ensure that the interval has an upper and lower bound"
    # Prealocate vectors
    x = np.linspace(interval[0], interval[1], N+1)
    h = (interval[1] - interval[0])/N
    y = np.zeros((len(y0), N+1))
    y[:,0] = y0
    # Initialise the RK method with predictor corrector steps
    for i in np.arange(len(x)-1):
        y1 = y[:, i] + h * (np.dot(A, y[:, i]) + bvector(x[i]))
        y2 = 0.75*y[:, i] + 0.25*y1 + 0.25*h * (np.dot(A, y1) + bvector(x[i+1]))
        y[:, i+1] = 1/3*y[:, i] + 2/3*y2 + 2/3*h * (np.dot(A, y2) + bvector(x[i+1]))
    return x, y


#@jit
def bvector1(x):
    """
    Returns a vector of zeros using the size of the system for problem 1:
    moderatelly stiff case
    """
    return np.zeros(2)

@jit
def bvector2(x):
    """
    Returns a vector of different function evaluations of 'x' of size (3,) for
    problem 2: stiff case
    """
    b1 = cos(10*x) - 10*sin(10*x)
    b2 = 199*cos(10*x) - 10*sin(10*x)
    b3 = 208*cos(10*x) + 10**4*sin(10*x)
    return np.array([b1, b2, b3])

@jit
def dirk3(A, bvector, y0, interval, N):
    """
    Function that returns a vector 'x' containing the points at which the 
    diagonally implicit Runge-Kutta of 3rd order method is applied to compute 'y'.
    'y' contains on each row the solution to a first order ODE problem
    and it has a number of rows corresponding to the overal degree of the equation.
    
    Inputs:
        A           n x n numpy array
                Contains the coefficients used to solve the ODE system
        bvector     function, takes 1 argument
                Contains corrections or the RHS of the ODE system
        y0          n size vector
                Contains the initial guess for y
        interval    List of two elements
                Contains the upper and lower bounds
        N           integer
                Specifies the number of points to be used
    Otuputs:
        x           N+1 size vector
                Contains the points at which the RK method is evaluated
        y           n x N+1 size vector
                Contains the results of RK evaluations
    """
    # Coefficient definition
    mu = 0.21132486540518708
    nu = 0.3660254037844386
    gam = 0.3169872981077807
    lam = 0.86602540378443871
    # Preallocation of vectors
    x = np.linspace(interval[0], interval[1], N+1)
    h = (interval[1] - interval[0])/N
    y = np.zeros((len(y0), N+1))
    y[:,0] = y0
    I = np.eye(len(A))
    # Solve DIRK3 at each point in x
    for i in np.arange(len(x)-1):
        y1 = np.linalg.solve(I - h*mu*A, y[:, i] + h*mu*bvector(x[i] + h*mu))
        y2 = np.linalg.solve(I - h*mu*A, y1 + h*nu*(np.dot(A, y1) + bvector(x[i] + h*mu))\
                            + h * mu * bvector(x[i] + h*nu + 2*h*mu))
        y[:, i+1] = (1-lam) * y[:, i] + lam * y2 + h*gam*(np.dot(A, y2) + bvector(x[i] + h*nu + 2*h*mu))
    return x, y


@jit
def Yexact1(x):
    """Returns the exat solutions for problem 1 moderatelly stiff"""
    y1 = exp(-1000 * x)
    y2 = 1000/999 * (exp(-x) - y1)
    return np.array([y1, y2])

@jit
def Yexact2(x):
    """Returns the exact solutions for problem 2: stiff case"""
    y1 = cos(10*x) - exp(-x)
    y2 = cos(10*x) + exp(-x) - exp(-100*x)
    y3 = sin(10*x) + 2*exp(-x) - exp(-100*x) - exp(-10000*x)
    return np.array([y1, y2, y3])


if __name__ == "__main__":
    # Problem 1: moderately stiff
    A_1 = np.array([[-1000, 0],
                  [1000, -1]])
    interval_1 = np.array([0, 0.1])
    y0_1 = np.array([1, 0])
    err_1 = np.zeros(10)
    erri_1 = np.zeros(10)
    h_1 = np.zeros(10)
    for k in np.arange(1, 11):
        N = 40*(k)
        h_1[k-1] = (interval_1[1] - interval_1[0])/float(N)
        x_1, y_1 = rk3(A_1, bvector1, y0_1, interval_1, N)
        xi_1, yi_1 = dirk3(A_1, bvector1, y0_1, interval_1, N)
        y_exact_1 = Yexact1(x_1)
        err_1[k-1] = h_1[k-1] * np.sum(np.abs((y_1[1, 1:] - y_exact_1[1, 1:])/y_exact_1[1, 1:]))
        erri_1[k-1] = h_1[k-1] * np.sum(np.abs((yi_1[1, 1:] - y_exact_1[1, 1:])/y_exact_1[1, 1:]))
    # Outputs and plots
    a1 = np.polyfit(log(h_1), log(err_1), 1)[0]
    a2 = np.polyfit(log(h_1), log(erri_1), 1)[0]
    print("The RK3 method has convergence of order {} and the DIRK3 method convergence of order {}".format(a1, a2))
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.loglog(h_1, err_1)
    ax2.loglog(h_1, erri_1)
    ax1.set_title("RK3")
    ax2.set_title("DIRK3")
    ax1.set_xlabel("h")
    ax2.set_xlabel("h")
    ax1.set_ylabel('|Error|')
    ax2.set_ylabel('|Error|')
    #Problem 2: stiff
    A_2 = np.array([[-1, 0, 0],
                  [-99, -100, 0],
                  [10098, 9900, -10000]])
    interval_2=np.array([0, 1])
    y0_2 = np.array([0, 1, 0])
    err_2 = np.zeros(15)
    erri_2 = np.zeros(15)
    h_2 = np.zeros(15)
    for k in np.arange(1, 16):
        N = 400*(k+1)
        h_2[k-1] = (interval_2[1] - interval_2[0])/float(N)
        x_2, y_2 = rk3(A_2[:], bvector2, y0_2[:], interval_2[:], N)
        xi_2, yi_2 = dirk3(A_2[:], bvector2, y0_2[:], interval_2[:], N)
        y_exact_2 = Yexact2(x_2)
        err_2[k-1] = h_2[k-1] * np.sum(np.abs((y_2[2, 1:] - y_exact_2[2, 1:])/y_exact_2[2, 1:]))
        erri_2[k-1] = h_2[k-1] * np.sum(np.abs((yi_2[2, 1:] - y_exact_2[2, 1:])/y_exact_2[2, 1:]))
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.loglog(h_1, err_1)
    ax2.loglog(h_1, erri_1)
    ax1.set_title("RK3")
    ax2.set_title("DIRK3")
    ax1.set_xlabel("h")
    ax2.set_xlabel("h")
    ax1.set_ylabel('|Error|')
    ax2.set_ylabel('|Error|')
