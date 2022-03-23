#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 14:08:39 2022

@author: ben
"""
import numpy as np
from ATM_waveform.parabolic_search_refinement import parabolic_search_refinement


def golden_section_search(f, x0, delta_x, bnds=[-np.Inf, np.Inf], \
                          integer_steps=False, tol=0.01, max_count=100, \
                              refine_parabolic=False, search_hist=None):
    """
    iterative search using the golden-section algorithm (more or less)

    Search the points in x0 for the minimum value of f, then either broaden the
    search range by delta_x, or refine the search spacing around the minumum x.
    Repeat until the minimum interval between searched values falls below tol
    If boundaries (bnds) are provided and the search strays outside them, refine
    the spacing around the nearest boundary until the stopping criterion is reached
    """
    if search_hist is None:
        search_hist={}

    # phi is the golden ratio.
    if integer_steps is False:
        phi=(1+np.sqrt(5))/2
    else:
        phi=2
    b=1.-1./phi
    a=1./phi
    # if x0 is not a list or an array, wrap it in a list so we can iterate
    if not hasattr(x0,'__iter__'):
        x0=[x0]
    R_dict=dict()
    it_count=0
    largest_gap=np.Inf
    while (len(R_dict)==0) or (largest_gap > tol):
        for x in x0:
            if x in R_dict:
                continue
            R_dict[x]=f(x)
        searched=np.array(sorted(R_dict.keys()))
        # make a list of R_vals searched
        R_vals=np.array([R_dict[x] for x in searched])
        # find the minimum of the R vals
        iR=np.argmin(R_vals)

        # choose the next search point.  If the searched picked the minimum or maximum of the
        # time offsets, take a step  of delta_x to the left or right
        if iR==0:
            x0 = [searched[0] - delta_x]
            if x0[0] < bnds[0]:
                # if we're out of bounds, search between the minimum searched value and the left boundary
                x0=[a*bnds[0]+b*np.min(searched[searched > bnds[0]])]
                largest_gap=searched[1]-searched[0]
            else:
                largest_gap=np.Inf
        elif iR==len(searched)-1:
            x0 = [searched[-1] + delta_x ]
            if x0[0] > bnds[1]:
                x0=[a*bnds[1]+b*np.max(searched[searched < bnds[1]])]
                largest_gap=searched[-1]-searched[-2]
            else:
                largest_gap=np.Inf
        else:
            # if the minimum was in the interior, find the largest gap in the x values
            # around the minimum, and add a point using a golden-rule search
            if searched[iR+1]-searched[iR] > searched[iR]-searched[iR-1]:
                # the gap to the right of the minimum is largest: put the new point there
                x0 = [ a*searched[iR] + b*searched[iR+1] ]
                largest_gap=searched[iR+1]-searched[iR]
            else:
                # the gap to the left of the minimum is largest: put the new point there
                x0 = [ a*searched[iR] + b*searched[iR-1] ]
                largest_gap=searched[iR]-searched[iR-1]
        if integer_steps is True:
            if np.floor(x0) not in searched:
                x0=np.floor(x0).astype(int)
            elif np.ceil(x0) not in searched:
                x0=np.ceil(x0).astype(int)
            else:
                break
        # need to make delta_t a list so that it is iterable
        it_count+=1
        if it_count > max_count:
            print("WARNING: too many shifts")
            break
    search_hist.update({'x':searched, 'R':R_vals})
    if refine_parabolic and (len(searched) >= 3):
        x_ref, R_ref = parabolic_search_refinement(searched, R_vals)
        if (x_ref < bnds[0]) or (x_ref > bnds[1]):
            return searched[iR], R_vals[iR]
        else:
            return x_ref, R_ref
    else:
        return searched[iR], R_vals[iR]
