#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:12:28 2022

@author: ben
"""

import numpy as np
from ATM_waveform.waveform import waveform

class WFcatalog(object):
    def __init__(self, n_samps, dt, t=None, blocksize=1024):
        self.data=[]
        self.index={}
        self.data_count=0
        self.nSamps=n_samps
        self.blocksize=blocksize
        self.block_count=0
        self.dt=dt
        if t is None:
            self.t=np.arange(n_samps)*dt
        else:
            self.t=t.copy()

    def __contains__(self, key):
        if isinstance(key, (int, float)):
            return key < self.data_count
        else:
            return key in self.index
    def __getitem__(self, key):
        if isinstance(key, (int, float)):
            N=key
        else:
            try:
                N=self.index[key]
            except KeyError:
                self.data_count += 1
                self.index[key] = self.data_count
                N=self.data_count
        block_ind=int(np.floor(N/self.blocksize))
        col=int(N-self.blocksize*block_ind)
        if block_ind > self.block_count-1:
            self.__add_block__()
        return WFref(col, self.data[block_ind])

    def __add_block__(self):
        self.data += [waveform(self.t, np.NaN+np.zeros((self.nSamps, self.blocksize), order='F'))]
        self.data[-1].p_squared=np.NaN+np.zeros((self.nSamps, self.blocksize), order='F')
        self.data[-1].mask=np.ones(self.data[-1].p.shape, dtype=np.int32, order='F')
        self.data[-1].FWHM=np.zeros_like(self.data[-1].t0)+np.NaN
        self.block_count += 1

    def update(self, key, p=None, p_squared=None, mask=None, t0=None, tc=None):
        temp=self[key]
        if p is not None:
            temp.p[:]=p.copy().ravel()
            if p_squared is None:
                temp.p_squared[:] = (p*p).ravel()
            else:
                temp.p_squared[:] = p_squared.ravel()
        if mask is not None:
            temp.mask[:]=mask.ravel()
        if t0 is not None:
            temp.t0[:]=t0
        if tc is not None:
            temp.tc[:]=tc


class WFref(object):
    __slots__=['t','t0','tc','p','p_squared','mask', 'nSamps','FWHM', 'dt',\
               'parent', 'col']
    def __init__(self, col, parent):
        self.parent=parent
        self.col=col
        self.t=parent.t
        self.nSamps=parent.nSamps
        self.p=parent.p[:,col]
        self.p_squared=parent.p_squared[:,col]
        self.nSamps=parent.nSamps
        self.mask=parent.mask[:,col]
        self.t0=parent.t0[col:col+1]
        self.tc=parent.tc[col:col+1]
        self.dt=parent.dt
        self.FWHM=parent.FWHM[col:col+1]
    def fwhm(self):
        """
        Calculate the full width at half max of a distribution
        """
        FWHM=self.parent.FWHM[self.col]
        if np.isfinite(FWHM):
            return [FWHM]
        p=self.p[:]
        t=self.parent.t
        dt=self.parent.dt
        p50=np.nanmax(p)/2
        # find the elements that have power values greater than 50% of the max
        ii=np.where(p>p50)[0]
        i50=ii[0]
        # linear interpolation between the first p>50 value and the last p<50
        dp=(p50 - p[i50-1]) / (p[i50] - p[i50-1])
        temp = t[i50-1] + dp*dt
        # linear interpolation between the last p>50 value and the next value
        i50=ii[-1]+1
        if np.isfinite(p[i50]):
            dp=(p50 - p[i50-1]) / (p[i50] - p[i50-1])
        else:
            dp=0
        FWHM = t[i50-1] + dp*dt - temp
        self.parent.FWHM[self.col]=FWHM
        return FWHM