#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 09:06:47 2018

@author: ben
"""
import numpy as np
import scipy.integrate as sciint
import matplotlib.pyplot as plt

def gaussian(x, ctr, sigma):
    """
        return a normalized gaussian kernel centered on 'ctr' with width 'sigma'
    """
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-ctr)**2/2/sigma**2)

class waveform(object):
    __slots__=['p','t','t0', 'dt', 'tc', 't_shift', 'size', 'error_flag', 'nSamps', 'nPeaks','shots','params','noise_RMS', 'p_squared','FWHM','seconds_of_day']
    """
        Waveform class contains tools to work data that give power as a function of time

        Attributes include:
            size: Number of traces in the waveform object
            nSamps: number of samples per waveform
            p: power (nSamps, size) or (nSamps,)
            t: relative time (nSamps,)
            t0: Offset time added to t (size)
            dt: time spacing of t
            tc: center time of the trace
            t_shift: any time shfits that have been applied to the waveform
            nPeaks: number of peaks in each trace
            shots: shot number of each trace
            noise_RMS: background noise estimate for each trace
            p_squared: the square of the p values
            FWHM: full width at half maximum of each trace
            seconds_of_day: time of day for each trace
    """
    def __init__(self, t, p, t0=0., tc=0., t_shift=0,  nPeaks=1, shots=None, noise_RMS=None, p_squared=None, FWHM=None, seconds_of_day=None):
        self.t=t
        self.t.shape=[t.size,1]
        self.dt=t[1]-t[0]
        self.p=p
        if p.ndim == 1:
            self.p.shape=[p.size,1]
        self.size=self.p.shape[1]
        self.nSamps=self.p.shape[0]
        self.params=dict()
        self.p_squared=p_squared
        self.error_flag = np.zeros(self.size, dtype=int)
        # turn keyword arguments into 1-d arrays of size (self.size,)
        kw_dict={'t0':t0, 'tc':tc, 'nPeaks':nPeaks,'shots':shots, 'noise_RMS':noise_RMS, 'seconds_of_day':seconds_of_day, 'FWHM': FWHM}
        for key, val in kw_dict.items():
            if val is None:
                setattr(self, key, None)
                continue
            if hasattr(val, '__len__'):
                this_N = val.size
            else:
                this_N = 1
            if this_N < self.size:
                setattr(self, key, np.zeros(self.size, dtype=np.array(val).dtype)+val)
            else:
                if isinstance(val, np.ndarray):
                    if val.shape==(self.size,):
                        setattr(self, key, val)
                    elif val.ndim==0:
                        setattr(self, key, np.array([val]))
                    else:
                        setattr(self, key, np.array(val))
                else:
                    setattr(self, key, np.array([val]))

    def __getitem__(self, key):
        result=waveform(self.t, self.p[:,key])
        for field in ['t0','tc','nPeaks', 'shots','noise_RMS','seconds_of_day','FWHM','error_flag']:
            temp=getattr(self, field)
            if temp is not None:
                if isinstance(key, np.ndarray):
                    setattr(result, field, temp[key])
                else:
                    setattr(result, field, temp[[key]])
        return result
        #return waveform(self.t, self.p[:,key], t0=self.t0[key], tc=self.tc[key], nPeaks=self.nPeaks[key], shots=self.shots[key],
        #                noise_RMS=self.noise_RMS[key], FWHM=fwhm, seconds_of_day=self.seconds_of_day[key])

    def centroid(self, els=None, threshold=None):
        """
        Calculate the centroid of a distribution, optionally for the subset specified by "els"
        """
        if els is not None:
            return np.sum(self.t[els]*self.p[els])/self.p[els].sum()
        if threshold is not None:
            p=self.p.copy()
            p[p<threshold]=0
            p[~np.isfinite(p)]=0
            return np.sum(self.t*p, axis=0)/np.sum(self.p, axis=0)
        return  np.sum(self.t*self.p, axis=0)/np.sum(self.p, axis=0)

    def sigma(self, els=None, C=None):
        """
        Calculate the standard deviation of the energy in a distribution,  optionally for the subset specified by "els"
        """
        if els is None:
            els=np.ones_like(self.t, dtype=bool)
        if C is None:
            C=self.centroid(els)
        return np.sqrt(np.sum(((self.t[els]-C)**2)*self.p[els])/self.p[els].sum())

    def percentile(self, P, els=None):
        """
        Calculate the specified percentiles of a distribution,  optionally for the subset specified by "els"
        """
        if els is not None:
            #C=np.cumsum(np.concatenate([[0],self.p[els][:-1].ravel()]))
            C=sciint.cumtrapz(self.p[els].ravel(), self.t[els].ravel(), initial=0)
            return np.interp(P, C/C[-1], self.t[els])
        else:
            C=np.cumsum(self.p)
            return np.interp(P, C/C[-1], self.t.ravel())

    def robust_spread(self, els=None):
        """
        Calculate half the difference bewteen the 16th and 84th percentiles of a distribution
        """
        lowHigh=self.percentile(np.array([0.16, 0.84]), els=els)
        return (lowHigh[1]-lowHigh[0])/2.

    def count_peaks(self, threshold=0.25, W=3, return_indices=False):
        """
        Count the peaks in a distribution
        """
        K=gaussian(np.arange(-3*W, 3*W+1), 0, W)
        N=np.zeros(self.size)
        if return_indices:
            peak_list=list()
        for col in range(self.size):
            pS=np.convolve(self.p[:,col], K,'same')
            peaks=(pS[1:-1] > pS[0:-2]) & (pS[1:-1] > pS[2:]) & (pS[1:-1] > np.nanmax(pS)*threshold)
            N[col]=peaks.sum()
            if return_indices:
                peak_list.append(np.where(peaks)[0]+1)
        if return_indices:
            return N, peak_list
        else:
            return N


    def nSigmaMean(self, N=3, els=None, tol=None, maxCount=20):
        """
            Calculate the iterative N-sigma edit, using the robust spread to measure sigma
        """
        if tol is None:
            tol=0.1*(self.t[1]-self.t[0])
        if els is None:
            els=self.p>0
        else:
            els = els & (self.p > 0)
        t_last=self.t[0]
        tc=self.centroid(els)
        sigma=self.robust_spread(els)
        count=0
        while (np.abs(t_last-tc) > tol) and (count<maxCount):
            count+=1
            these=(self.p > 0) & (np.abs(self.t-tc) < N*sigma)
            t_last=tc;
            tc=self.centroid(els=these)
            sigma=self.robust_spread(els=these)
        return tc, sigma

    def subBG(self, bg_samps=np.arange(0,30, dtype=int), t50_minus=None):
        """ subtract a background estimate from each trace

        For each individual waveform, calculate an estimate of the bacground estimate.
        Two options allowed are:
            -specify samples with bg_samps (default = first 30 samples of the trace)
            -specify t50_minus: samples earlier than the trace's t50() minus
                t50_minus are used in the background calculation
        """
        if t50_minus is not None:
            t50=self.t50()
            bgEst=np.zeros(self.size)
            noiseEst=np.zeros(self.size)
            for ii in range(self.size):
                bgind=np.where(self.t < t50[ii]-t50_minus)[0]
                if len(bgind) > 1:
                    bgEst[ii]=np.nanmean(self.p[bgind, ii])
                    noiseEst[ii]=np.nanstd(self.p[bgind, ii])
        else:
            bgEst=np.nanmean(self.p[bg_samps, :], axis=0)
            noiseEst=np.nanstd(self.p[bg_samps,:], axis=0)
        self.p=self.p-bgEst
        self.noise_RMS=noiseEst
        return self

    def normalize(self):
        """
        Normalize a distribution by its background-corrected maximum
        """
        self.subBG()
        self.p=self.p/np.nanmax(self.p, axis=0)
        return self

    def t50(self):
        """
        Find the first 50%-max threshold crossing of a waveform
        """
        t50=np.zeros(self.size)
        for col in np.arange(self.size):
            p=self.p[:,col]
            p50=np.nanmax(p)/2
            i50=np.where(p>p50)[0][0]
            dp=(p50 - p[i50-1]) / (p[i50] - p[i50-1])
            t50[col] = self.t[i50-1] + dp*self.dt
        return t50

    def fwhm(self):
        """
        Calculate the full width at half max of a distribution
        """
        if self.FWHM is not None:
            return self.FWHM
        FWHM=np.zeros(self.size)
        for col in np.arange(self.size):
            p=self.p[:,col]
            p50=np.nanmax(p)/2
            # find the elements that have power values greater than 50% of the max
            ii=np.where(p>p50)[0]
            i50=ii[0]
            # linear interpolation between the first p>50 value and the last p<50
            dp=(p50 - p[i50-1]) / (p[i50] - p[i50-1])
            temp = self.t[i50-1] + dp*self.dt
            # linear interpolation between the last p>50 value and the next value
            i50=ii[-1]+1
            if np.isfinite(p[i50]):
                dp=(p50 - p[i50-1]) / (p[i50] - p[i50-1])
            else:
                dp=0
            FWHM[col] = self.t[i50-1] + dp*self.dt - temp
        self.FWHM=FWHM
        return FWHM

    def calc_mean(self, threshold=255, normalize=True):
        """
        Calculate the centroid of a distribution relative to a thresohold
        """
        good=np.sum( (~np.isfinite(self.p)) & (self.p < threshold), axis=0) < 2
        if normalize:
            WF_mean=waveform(self.t, np.nanmean(self[good].normalize().p, axis=1))
        else:
            WF_mean=waveform(self.t, np.nanmean(self[good].p, axis=1))
        WF_mean.p.shape=[WF_mean.p.size, 1]
        return WF_mean

    def  threshold_centroid(self, fraction=0.38):
        """
        Calculate the centroid of the energy of a waveform where the amplitude is more than a fraction of the maximum
        """
        C=np.zeros(self.size)+np.NaN
        t=self.t.ravel()
        for col in np.arange(self.size):
            p=self.p[:,col]
            bins = (p>np.nanmax(p)*fraction) & np.isfinite(p)
            if bins.sum() >= 2:
                C[col] = np.sum(p[bins]*t[bins])/np.sum(p[bins])
        return C

    def broaden(self, sigma):
        if sigma==0:
            return self
        nK = np.minimum(np.floor(self.p.size/2)-1,3*np.ceil(sigma/self.dt))
        tK = np.arange(-nK, nK+1)*self.dt
        K = gaussian(tK, 0, sigma)
        K /= np.sum(K)
        self.p=np.convolve(self.p.ravel(), K,'same')
        self.p.shape=[self.p.size,1]
        return self