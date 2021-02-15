#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:51:30 2020

@author: ben
"""

import scipy.stats as ss
import numpy as np

def RDE(x):
    xs=x.copy()
    xs=np.isfinite(xs)   # this changes xs from values to a boolean
    if np.sum(xs)<2 :
        return np.nan
    ind=np.arange(0.5, np.sum(xs))
    LH=np.interp(np.array([0.16, 0.84])*np.sum(xs), ind, np.sort(x[xs]))
    return (LH[1]-LH[0])/2.  # trying to get some kind of a width of the data ~variance


def three_sigma_edit_fit(G0, d0, n_iterations=5, sigma_min=0):
    good0=np.all(np.isfinite(G0), axis=1) & np.isfinite(d0)
    good=good0.copy()
    m=np.zeros(G0.shape[1])
    r=np.zeros(d0.shape)
    sigma=np.NaN
    good_last=np.zeros_like(d0, dtype=bool)
    for ii in range(n_iterations+1):
        G=G0[good,:]
        if (G.shape[0]<G.shape[1]) or np.all(good_last==good):
            break
        try:
            m=np.linalg.solve(G.T.dot(G), G.T.dot(d0[good]))
        except Exception as E:
            m=np.zeros(G.shape[1])+np.NaN
            r=np.zeros_like(d0)+np.NaN
            sigma=np.NaN
            good=np.zeros_like(d0, dtype=bool)
            return m, r, sigma, good
        r=d0[good0]-G0[good0,:].dot(m)
        sigma=RDE(r[good])
        good = good0 & ( np.abs(r) < 3*np.maximum(sigma, sigma_min) )
        
    return m, r, sigma, good