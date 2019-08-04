#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:40:17 2019

@author: ben
"""

from __future__ import division
cimport numpy as np
import numpy as np
cimport cython
from math import pi

@cython.boundscheck(False)
@cython.cdivision(True)
def corr_no_mean(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] y, np.ndarray[double, ndim=1] x2, np.ndarray[int, ndim=1] mask, int N):
    """
        Efficient correlation calculation for two arrays, skippning mean subtraction

        Inputs :
            x: independent variable, (n,)
            y: dependent variable (n,)
            mask: which elements to include in the calculation
            N: number of elements in each array
        outputs:
            A: the scaling of x needed to match y in a least-squares sense
            R: the RMS difference between A*x and y
    """
    cdef double sum_xy=0.
    cdef double sum_x2=0.
    cdef int count=0
    cdef R=0.
    for i in range(N-1, -1, -1):
        #if np.isfinite(x[i]) and np.isfinite(y[i]):
        #    mask[i]=1
        if mask[i]>0:
            sum_xy += x[i]*y[i]
            sum_x2 += x2[i]
            count += 1
    A=sum_xy/sum_x2
    for i in range(N-1, -1, -1):
        if mask[i]>0:
            r = (y[i]-A*x[i])
            R += r*r
    R = np.sqrt(R/(count-2))
    return A, R

