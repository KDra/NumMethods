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

def hair_pos(R, L, fx, thL, phiL=0):
    if not phiL == 0:
        assert len(thL) == len(phiL),\
        "Phi and Theta must be of equal size"
        assert np.shape(phiL) == (1,),\
        "Phi must be a vector"
    assert np.shape(thL) == (1,),\
    "Theta must be a vector"
    