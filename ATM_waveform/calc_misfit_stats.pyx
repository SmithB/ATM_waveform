#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 16:47:33 2022

@author: ben
"""
from __future__ import division
cimport numpy as np
import numpy as np
cimport cython
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.initializedcheck(False)
#def calc_misfit_stats(int ix_start, int iy_start, int N,  \
#            np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] y, \
#            np.ndarray[double, ndim=1] x2, np.ndarray[double, ndim=1] y2, \
#            np.ndarray[int, ndim=1] mask_x, np.ndarray[int, ndim=1] mask_y):
def calc_misfit_stats(int ix_start, int iy_start, int N,  \
            double[:] x, double[:] y, \
            double[:] x2, double[:] y2, \
            int[:] mask_x, int[:] mask_y):
    cdef int ii
    cdef int ix
    cdef int iy
    cdef int iy0
    cdef int iy1
    cdef double xx=0
    cdef double xy0=0
    cdef double xy1=0
    cdef double y0y0=0
    cdef double y0y1=0
    cdef double y1y1=0
    cdef double dt_f
    cdef int count=0
    cdef double R2

    # calculate dot products
    for ii in range(0, N):
        iy0 = iy_start+ii
        iy1 = iy0 + 1
        ix = ix_start + ii
        if mask_x[ix] > 0 and mask_y[iy0] > 0 and mask_y[iy1] > 0:
            count += 1
            xx  += x2[ix]
            xy0 += x[ix] * y[iy0]
            xy1 += x[ix] * y[iy1]
            y0y0 += y2[iy0]
            y0y1 += y[iy0] * y[iy1]
            y1y1 += y2[iy1]
    di_f = (xy0*y0y1-xy1*y0y0)/(xy0*(y0y1-y1y1) + xy1*(y0y1-y0y0))
    #di_f = (xx*y0y0 - xx*y0y1 - xy0*xy0 + xy0*xy1) / \
    #    (xx*y0y0 - 2*xx*y0y1 + xx*y1y1 - xy0*xy0 + 2*xy0*xy1 - xy1**2)
    if di_f < 0:
        di_f = 0.
    if di_f > 1:
        di_f = 1.

    A  = (xy0*(y1y1-y0y1) + xy1*(y0y0-y0y1)) / (y0y0*y1y1-y0y1**2)
    # calc R2: need y1y1, y0y1, y0y0, xy0, xy1
    R2 = A*A *(di_f**2*y1y1 + 2*(1-di_f)*di_f*y0y1 +  (1-di_f)**2*y0y0) -2*A*(di_f*xy1+(1-di_f)*xy0) + xx

    if R2 < 0:
        R2 = 0.
    #return sqrt(R2/(count-1)), di_f, A, count
    return R2, di_f, A, count