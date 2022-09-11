#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 18:55:51 2022

@author: ben
"""

import numpy as np
from time import time
import scipy.stats as sps
import copy
from ATM_waveform.read_ATM_wfs import read_ATM_file
from ATM_waveform.fit_waveforms import fit_catalog
import argparse
#import matplotlib.pyplot as plt
import h5py

def get_tx_est(filename, nShots=np.Inf, TX0=None, source='TX', pMin=50, skip_n_tx=None, skip_make_TX=False, verbose=False):
    if source == 'TX':
        # get the transmit pulse mean
        D=read_ATM_file(filename, nShots=nShots, readTX=True, readRX=False)
        TX=D['TX']
    if source == 'RX':
        D=read_ATM_file(filename, nShots=nShots, readRX=True, readTX=False)
        TX=D['RX']
    if TX0 is None:
        # select waveforms that are not clipped and have adequate amplitude
        valid=np.where((np.nanmax(TX.p,axis=0) < 255) & (np.nanmax(TX.p,axis=0) > pMin) & (D['calrng'] > 5))[0]
        TX1=TX[valid]
        t50=TX1.t50()
        ti=TX1.t.ravel()-np.mean(TX1.t)
        # align the pulses on their 50% threshold
        for ii in range(TX1.size):
            TX1.p[:,ii]=np.interp(ti, TX1.t.ravel()-t50[ii], TX1.p[:,ii])
        TXm=TX1.calc_mean().subBG().normalize()
        TXn=TX1[np.arange(0, TX1.size, dtype=int)].subBG().normalize()
        misfit=np.sqrt(np.nanmean((TXn.p-TXm.p)**2, axis=0))

        # calculate the mean of the WFs most similar to the mean
        TX0 = TXn[misfit<2*np.median(misfit)].calc_mean()
        txC, txSigma=TX0.nSigmaMean()
        TX0.t=TX0.t-txC
        TX0.tc=0

    # Prepare the input txdata for fitting
    # Use a subsetting operation to copy the data and remove clipped WFs
    calrng=D['calrng']
    if skip_n_tx is not None:
        ind=np.arange(0, TX.size, skip_n_tx, dtype=int)
        TX=TX[ind]
        calrng=calrng[ind]
    valid=np.where((np.nanmax(TX.p,axis=0) < 255) & (np.nanmax(TX.p,axis=0) > pMin) & (calrng>1))[0]
    txData = TX[valid]
    txData.t=txData.t-txData.t.mean()
    thresholdMask = txData.p >= 255
    txData.subBG()
    txData.tc=np.array([txData[ii].nSigmaMean()[0] for ii in range(txData.size)])
    txData.p[thresholdMask]=np.NaN
    txData.nPeaks=np.ones(txData.size)

    t_old=time()
    deltas = np.arange(-2.5, 2.5, 1)
    sigmas = np.arange(0, 1.5, 0.25)
    # minimize the shifted misfit between each transmit pulse and the waveform mean
    txP=fit_catalog(txData, {0.:TX0}, sigmas, deltas)
    print("     time to fit start pulse=%3.3f" % (time()-t_old))

    if skip_make_TX is True:
        return dict(), txP

    # evaluate the fit and find the waveforms that best match the mean transmitted pulse
    RR = txP['R'] / txP['A']
    error_tol=sps.scoreatpercentile(RR, 68)
    all_shifted_TX=list()
    for ii in range(len(txP['R'])):
        temp=np.interp(TX0.t.ravel(), txData.t.ravel() - txP['delta_t'][ii], txData[ii].p.ravel())
        if RR[ii] < error_tol and txP['A'][ii] < 250 and txP['sigma'][ii] <= sigmas[2]:
            if np.isfinite( txP['B'][ii]):
                temp = (temp - txP['B'][ii]) / txP['A'][ii]
            else:
                temp = temp  / txP['A'][ii]
            all_shifted_TX.append(temp)
    # put together the shifted transmit pulses that passed muster
    all_shifted_TX = np.c_[all_shifted_TX]
    TX = copy.copy(TX0)
    all_shifted_TX[~np.isfinite(all_shifted_TX)]=0.
    TX.p = np.nanmean(all_shifted_TX, axis=0).reshape(TX0.t.shape)
    TX.p = np.interp(TX.t.ravel(), TX.t.ravel()-TX.nSigmaMean()[0], TX.p.ravel()).reshape(TX.t.shape)
    TX.tc = np.array(TX.nSigmaMean()[0])
    TX.normalize()
    return TX, txP

def main():
    parser = argparse.ArgumentParser(description='Fit the waveforms from an ATM file with a set of scattering parameters')
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--N_shots', '-N', type=int, default=1000)
    args=parser.parse_args()
    TX, TXp = get_tx_est(args.input_file, source='RX', nShots=args.N_shots)

    with h5py.File(args.output_file,'w') as h5f:
        h5f.create_dataset('TX/t', data=TX.t.ravel())
        h5f.create_dataset('TX/p', data=TX.p.ravel())


if __name__=="__main__":
    main()

