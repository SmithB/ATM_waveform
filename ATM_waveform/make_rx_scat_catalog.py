#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 14:01:44 2018

@author: ben
"""

import numpy as np
import h5py
from ATM_waveform.waveform import waveform

def make_rx_scat_catalog(TX, h5_file=None, reduce_res=False):
    """
    make a dictionary of waveform templates by convolving the transmit pulse with
    subsurface-scattering SRFs
    """
    # assume that TX time units are in ns
    dt=TX.t[1]-TX.t[0]
    if h5_file is None:
        h5_file='/Users/ben/Dropbox/ATM_red_green/subsurface_srf_no_BC.h5'
    with h5py.File(h5_file,'r') as h5f:
        t0=np.array(h5f['t'])*1.e9;
        z=np.zeros_like(t0)
        z[np.argmin(abs(t0))]=1;
        TXc=np.convolve(TX.p.ravel(), z, 'full')
        TX.p[~np.isfinite(TX.p)]=0.
        t_full=np.arange(TXc.size)*dt
        t_full -= waveform(t_full, TXc).nSigmaMean()[0]
        RX=dict()
        
        if 'r_eff' in h5f:
            key_vals = np.array(h5f['r_eff'])
            key_field='r_eff'
            if reduce_res:
                key_vals=np.concatenate([key_vals[key_vals< 1e-4][::10],
                                         key_vals[(key_vals >= 1e-4) &  (key_vals < 5e-3)][::2], key_vals[key_vals >= 5e-3] ])

        else:
            key_vals = np.array(h5f['K'])
            key_field='K'
        for key_val in sorted(h5f[key_field]):
            if key_val not in key_vals:
                continue
            row=np.flatnonzero(key_vals==key_val)[0]
            rx0=h5f['p'][row,:]
            # again, assume that TX.t units are ns
            temp=np.convolve(TX.p.ravel(), rx0, 'full')*dt*1e-9
            RX[key_val]=waveform(TX.t, np.interp(TX.t.ravel(), t_full.ravel(), temp).reshape(TX.t.shape))
            RX[key_val].t0=0.
            RX[key_val].tc=RX[key_val].nSigmaMean()[0]
    return RX

