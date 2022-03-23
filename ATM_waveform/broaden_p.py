#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 10:46:57 2022

@author: ben
"""

import numpy as np

def broaden_p(wf, sigma):
    if sigma==0:
        return wf.p
    nK = np.minimum(np.floor(wf.p.size/2)-1,3*np.ceil(sigma/wf.dt))
    tK = np.arange(-nK, nK+1)*wf.dt
    K = gaussian(tK, 0, sigma)
    K /= np.sum(K)
    return np.convolve(wf.p.ravel(), K,'same')


def gaussian(x, ctr, sigma):
    """
        return a normalized gaussian kernel centered on 'ctr' with width 'sigma'
    """
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-ctr)**2/2/sigma**2)
