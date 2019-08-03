#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:16:30 2019

@author: ben
"""

from corr_no_mean_cython import corr_no_mean_cython
from fit_waveforms import amp_misfit
import numpy as np
from time import time

N_iterations=5000

x=np.arange(255, dtype=np.float64)
y=2*x+np.random.randn(255)
x2=x**2
els= (x>5) & (x < 200)
x[~els]=np.NaN

R,  A, ii = amp_misfit(x, y, els=els, x_squared=x2)

A1, R1 = corr_no_mean_cython(x, y, x2, els.astype(np.int32), 255)

print("python: A=%f, R=%f" % (A, R))
print("cython: A=%f, R=%f" % (A1, R1))



t0=time()
for ii in range(N_iterations):
    #R,  A, ii = amp_misfit(x, y,  x_squared=x2)
    R,  A, ii = amp_misfit(x, y, els=els, x_squared=x2)
dt=time()-t0
print("python: time for %d iterations = %3.3f" % (N_iterations,   dt))
els=els.astype(np.int8)
t0=time()
for ii in range(N_iterations):
    els=np.isfinite(x) & np.isfinite(y)
    A1, R1 = corr_no_mean_cython(x, y, x2, els.astype(np.int32), 255)
dt1=time()-t0
print("cython: time for %d iterations = %3.3f" % (N_iterations,   dt1))

print("python / cython = %3.2f" % (dt/dt1))