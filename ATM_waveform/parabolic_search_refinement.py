#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 14:11:59 2022

@author: ben
"""

import numpy as np

def parabolic_search_refinement(x, R):
    """
    Fit a parabola to search results (x, R), return the refined values
    """
    ind=np.argsort(x)
    xs=x[ind]
    Rs=R[ind]
    best=np.argmin(R)
    if best==0:
        p_ind=[0, 1, 2]
    elif best==len(R)-1:
        p_ind=np.array([-2, -1, 0])+best
    else:
        p_ind=best+np.array([-1, 0, 1])
    G=np.c_[np.ones((3,1)), (xs[p_ind]-xs[p_ind[1]]), (xs[p_ind]-xs[p_ind[1]])**2]

    m=np.linalg.solve(G, Rs[p_ind])
    x_optimum= -m[1]/2/m[2]
    R_opt= m[0]  + m[1]*x_optimum + m[2]*x_optimum**2
    return x_optimum + xs[p_ind[1]], R_opt
