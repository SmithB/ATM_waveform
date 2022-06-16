# -*- coding: utf-8 -*-
"""
Created on Tue Jun  14 20:41:00 2022

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

def golden_section_search(f, x0, delta_x, bnds=[-np.Inf, np.Inf], \
                          integer_steps=False, step_size=1, tol=0.01, \
                              max_count=100, refine_parabolic=False,
                              search_hist={}, fn_args=None, fn_kwargs=None):
    """
    iterative search using the golden-section algorithm (more or less)

    Search the points in x0 for the minimum value of f, then either broaden the
    search range by delta_x, or refine the search spacing around the minumum x.
    Repeat until the minimum interval between searched values falls below tol
    If boundaries (bnds) are provided and the search strays outside them, refine
    the spacing around the nearest boundary until the stopping criterion is reached
    """
    if fn_args is None:
        fn_args=[]
    if fn_kwargs is None:
        fn_kwargs={}

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
    searched=np.array([])
    R_dict=dict()
    it_count=0
    largest_gap=np.Inf
    while (len(searched)==0) or (largest_gap > tol):
        for x in x0:
            # Calculate the residuals here:
            R_dict[x]=f(x, *fn_args, **fn_kwargs)
            searched=np.union1d(searched, [x])
        # make a list of R_vals searched
        R_vals=np.array([R_dict[x] for x in searched])
        # find the minimum of the R vals
        try:
            iR=np.argmin(R_vals)
        except TypeError:
            print("HERE!")
        # choose the next search point.  If the search picked the minimum or maximum of the
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
            # convert x0 to an integer step if it isn't already
            if np.floor(np.array(x0)/step_size).astype(int)*step_size not in searched:
                x0=np.floor(np.array(x0)/step_size).astype(int)*step_size
            elif np.ceil(np.array(x0)/step_size).astype(int)*step_size not in searched:
                x0=np.ceil(np.array(x0)/step_size).astype(int)*step_size
            else:
                # exit if we have searched both the integer below and the integer above x0
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
