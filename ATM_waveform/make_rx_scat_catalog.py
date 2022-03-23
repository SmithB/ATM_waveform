#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 14:01:44 2018

@author: ben
"""

import numpy as np
import h5py
from ATM_waveform.waveform import waveform
import scipy.interpolate as si

def make_rx_scat_catalog(TX0, h5_file=None, reduce_res=False):
    """
    make a dictionary of waveform templates by convolving the transmit pulse with
    subsurface-scattering SRFs

    TBD: shift the transmit pulse by a fraction of a sample so that its centroid is exactly on zero time
    """
    if h5_file is None:
        h5_file='/Users/ben/Dropbox/ATM_red_green/subsurface_srf_no_BC.h5'


    # make a shifted, sample-aligned version of the transmit pulse:
    t0 = TX0.t.ravel()-TX0.t[np.argmin(np.abs(TX0.t))]
    p0 = si.interp1d(TX0.t.ravel()-TX0.nSigmaMean()[0], TX0.p.ravel(),'cubic',
                   bounds_error=False, fill_value = 'extrapolate')(t0)
    TX = waveform(t0, p0)


    with h5py.File(h5_file,'r') as h5f:
        t0=np.array(h5f['t'])*1.e9;
        z=np.zeros_like(t0)
        z[np.argmin(abs(t0))]=1;
        TXc=np.convolve(TX.p.ravel(), z, 'full')
        TX.p[~np.isfinite(TX.p)]=0.
        t_full=np.arange(TXc.size)*0.25
        t_full -= waveform(t_full, TXc).nSigmaMean()[0]
        RX=dict()
        # the zero value is the same as the transmit pulse
        this_p =  si.interp1d( t_full.ravel(), TXc.ravel(),\
                              'cubic', bounds_error=False, \
                            fill_value='extrapolate')(TX.t.ravel())
        RX[0.]=waveform(TX.t,this_p.reshape(TX.t.shape))
        r_vals = np.array(h5f['r_eff'])
        if reduce_res:
            r_vals=np.concatenate([r_vals[r_vals< 1e-4][::10], r_vals[(r_vals >= 1e-4) &  (r_vals < 5e-3)][::2], r_vals[r_vals >= 5e-3] ])
        for row, r_val in enumerate(h5f['r_eff']):
            if r_val not in r_vals:
                continue
            rx0=h5f['p'][row,:]
            temp=np.convolve(TX.p.ravel(), rx0, 'full')*0.25e-9
            this_p = si.interp1d(t_full.ravel(), temp.ravel(),
                                 'cubic',bounds_error=False, \
                          fill_value='extrapolate')(TX.t.ravel())
            RX[r_val]=waveform(TX.t, this_p.reshape(TX.t.shape))
    for r_val in RX.keys():
        RX[r_val].t0=0.
        RX[r_val].tc=RX[r_val].nSigmaMean()[0]
        RX[r_val].mask = np.isfinite(RX[r_val].p).astype(np.int32)
    return RX

