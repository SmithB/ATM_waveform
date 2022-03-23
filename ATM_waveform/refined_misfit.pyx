#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 13:40:42 2022

@author: ben
"""
from __future__ import division
cimport numpy as np
import numpy as np
cimport cython
from math import sqrt
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)  # turn off negative index wrapping for entire function

def refined_misfit(int di0, np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] y, \
            np.ndarray[double, ndim=1] x2, np.ndarray[double, ndim=1] y2, \
               np.ndarray[int, ndim=1] mask_x, np.ndarray[int, ndim=1] mask_y, \
               int N):

    #def refined_misfit(di0, x, y,mask_x, mask_y, N):

    """
        Efficient misfit calculation for two vectors

        Inputs :
            di0: the integer number of samples to shift x
            x: independent variable, (n,)
            y: dependent variable (n,)
            x2, y2 : squares of x and y
            mask_x, mask_y: elements that should be included in the calculation
            N: number of elements in each array
        outputs:
            di_f: the fractional shift needed to minimize the interpolated residual
            A: the scaling of x needed to match y in a least-squares sense
            R: the RMS difference between A*x and y
    """

    cdef double xx=0
    cdef double xy0=0
    cdef double xy1=0
    cdef double y0y0=0
    cdef double y0y1=0
    cdef double y1y1=0
    cdef double di_f =0.
    cdef double A = 0.
    cdef int i0
    cdef int i1
    cdef int count=0
    cdef int first_samp
    cdef int last_samp

    # x is to be shifted by -di0
    first_samp = N-2 + di0
    if first_samp > N-2:
        first_samp = N-2
    last_samp = di0
    if last_samp < 0:
        last_samp = 0

    # calculate dot products
    for i0 in range(first_samp, last_samp-1, -1):
        i1 = i0 + 1
        ix = i0 - di0
        if mask_x[ix] > 0 and mask_y[i0] > 0 and mask_y[i1] > 0:
            count += 1
            xx  += x2[ix]
            xy0 += x[ix] * y[i0]
            xy1 += x[ix] * y[i1]
            y0y0 += y2[i0]
            y0y1 += y[i0] * y[i1]
            y1y1 += y2[i1]

    di_f = (xx*y0y0 - xx*y0y1 - xy0*xy0 + xy0*xy1) / \
        (xx*y0y0 - 2*xx*y0y1 + xx*y1y1 - xy0*xy0 + 2*xy0*xy1 - xy1**2)

    A = (xy0 - di_f*xy0 + di_f*xy1)/xx
    R2 = xx*A**2 + y0y0*(1-di_f)**2 + y1y1*di_f**2 -2*xy0*A*(1-di_f) \
        - 2*xy1*A*di_f + 2*y0y1*di_f*(1-di_f)

    # for very small misfits, roundoff may make R2 less than zero
    if R2 < 0:
        R2 = 0.

    return di_f, A, sqrt(R2/(count-1))

