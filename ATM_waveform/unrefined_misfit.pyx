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

def unrefined_misfit(int di0, np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] y, \
            np.ndarray[double, ndim=1] x2, np.ndarray[double, ndim=1] y2,
               np.ndarray[int, ndim=1] mask_x, np.ndarray[int, ndim=1] mask_y, \
               int N):

    #def refined_misfit(di0, x, y,mask_x, mask_y, N):

    """
        Efficient misfit calculation for two vectors

        Inputs :
            di0: the integer number of samples to shift x
            x: independent variable, (n,)
            y: dependent variable (n,)
            mask_x, mask_y: elements that should be included in the calculation
            N: number of elements in each array
        outputs:
            R: the RMS difference between A*x and y
    """

    cdef double xx=0
    cdef double xy=0
    cdef double yy=0
    cdef double R2 = 0.
    cdef int ix
    cdef int iy
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
    for iy in range(first_samp, last_samp-1, -1):
        ix = iy - di0
        if mask_x[ix] > 0 and mask_y[iy] > 0:
            count += 1
            xx += x2[ix]
            yy += y2[iy]
            xy += x[ix]*y[iy]

    R2 = yy-xy*xy/xx

    # for very small misfits, roundoff may make R2 less than zero
    if R2 < 0:
        R2 = 0.

    return sqrt(R2/(count-1))

