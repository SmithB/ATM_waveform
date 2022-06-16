#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 16:28:31 2022

@author: ben
"""

import numpy as np
from ATM_waveform.calc_misfit_stats import calc_misfit_stats

def find_matching_indices(tx, ty):

    # y is interpolated at the time values in tx -> need one value before the
    # first value in tx, and one after the last

    Nx=tx.size
    Ny=ty.size
    di0f = 0.
    di1f = 0.
    ix0 = 0
    iy0 = 0
    ixN = Nx
    N = Nx
    di_f = 0.
    dt = tx[1]-tx[0]

    # check if the first value in tx is in ty
    di0f = (tx[0] - ty[0])/dt
    if di0f < 0:
        # if the first value in tx is less than the first value in ty,
        # calculate the first value in x to use
        #ix0 = ceil((ty0[0] - tx0[0])/dt)
        ix0 = int(np.ceil(-di0f))
    else:
        iy0 = int(np.floor(di0f))

    # calculate the difference between the last sample in y and the last
    # sample in x
    #di1f = ( (tx0 + dt*(Nx-1) - (ty0 + dt*(Ny-1) )) /dt
    #     = (tx0/dt + Nx-1 - (ty0/dt + Ny-1))
    di1f = (tx[0]-ty[0])/dt + Nx - Ny
    if di1f > 0:
        # last sample in tx is after last sample in ty
        ixN = Nx - int(np.ceil(di1f))

    N = ixN-ix0
    di_f = di0f-np.floor(di0f)

    return ix0, iy0, N, di_f

def calc_R_and_tshift(t_shift, WFd, WFm, fit_history=None, return_R_only=False):

    """
        Efficient misfit calculation for two vectors

        Inputs :
            t_shift: time by which WFm should be shifted initially
            WFd: the measured waveform, to whose time values the model will be interpolated
            WFm: the model waveform, which will be scaled, shifted and interpolated to match the measured waveform
            fit_history: dict, optional
                dictionary to contain the history results of searches (R, A, t_shift, count)
            return_R_only: bool, optional
                If true, return only R (other values may be stored in fit_history)
        outputs:
            R: the RMS difference between A*WFm.p amd WF.p
            t_shift_refined: the refined shift value needed to make the best match between the waveforms (applied to WFm)
            A: the scaling of WFm
            count: the number of samples used in the match

    """
    # make sure all the right fields are there
    for WF in WFd, WFm:
        if WF.p_squared is None:
            WF.p_squared =WF.p**2
        try:
            if WF.mask is None:
                WF.mask = np.isfinite(WF.p).astype(np.int32)
            elif WF.mask.dtype != np.int32:
                WF.mask=WF.mask.astype(np.int32)
        except AttributeError:
            WF.mask = np.isfinite(WF.p).astype(np.int32)

    dt = WFd.t[1] - WFd.t[0]

    id_start, im_start, N, di0_f = find_matching_indices(WFd.t.ravel(), WFm.t.ravel()+t_shift)

    # call the cython routines that calculate the dot products
    R2, di_f, A, count = calc_misfit_stats(id_start, im_start, N,  \
            WFd.p.ravel(), WFm.p.ravel(), \
            WFd.p_squared.ravel(), WFm.p_squared.ravel(), \
            WFd.mask.ravel(), WFm.mask.ravel())

    t_shift_refined = (WFd.t[id_start]-di_f*dt)-WFm.t[im_start]
    R=np.sqrt(R2/(count-1))

    if fit_history is not None:
        fit_history[t_shift]={'R': R, 'A':A, \
                              't_shift_refined':t_shift_refined, \
                                  'count':count}
    if return_R_only:
        return R
    else:
        return R, t_shift_refined, A, count
