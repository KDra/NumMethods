# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 00:10:21 2015

@author: kostas
"""

import numpy as np

a1 = np.mat([[1, 2], [3, 4]])
a2 = np.mat([[-3, 2], [3, 6]])

a1.T
a2.I

def wcond(a):
    c = np.linalg.cond(a)
    if c < 10:
        print "matrix is well conditioned"
    else:
        print "matrix is not well conditioned"

def bisect(f, a, b):
    x = (a+b)/2.0
    while np.abs(f(x)) > 1e-10:
        if f(x) < 0:
            a = x
            x = (a+b)/2.0
        else:
            b = x
            x = (a+b)/2.0
        print x
    return x

def f(x):
    return np.tan(x) - np.exp(-x)

bisect(f, 0, 1)