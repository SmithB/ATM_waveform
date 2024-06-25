#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 18:31:16 2018

@author: ben
"""
import h5py
import numpy as np
#import matplotlib
#matplotlib.use('nbagg')
from ATM_waveform.waveform import waveform


## Changes:   Removed the peak count ('/pulse/count')
##            Changed variables that were in /waveforms/twv/foo/bar to /Waveforms/twv/foo_bar
## Will need to modify dependent functions to ignore the missing peak count

def _read_wf(D, shot, starting_sample=0, read_tx=False, read_rx=False, read_all=False):
    """read transmit and receive pulses for a shot from data extracted from an h5 file
    """
    gate0=D['/Waveforms/twv/shot_gate_start'][shot]-1
    gateN=gate0+D['/Waveforms/twv/shot_gate_count'][shot]
    result=list()
    result=dict()
    if read_tx:
        result['tx']={'gate':gate0+D['/laser/gate_xmt'][shot]-1}
    if read_rx:
        result['rx']={'gate':gate0+D['/laser/gate_rcv'][shot]-1}
    if read_all:
        for ii in np.arange(gateN-gate0+1, dtype=int):
            result[ii]={'gate':gate0+ii}
    for key in result:
        gate=result[key]['gate']
        samp0=D['/Waveforms/twv/gate_wvfm_start'][gate]-1-starting_sample
        sampN=samp0+D['/Waveforms/twv/gate_wvfm_length'][gate]
        result[key].update({'pos':D['/Waveforms/twv/gates_position'][gate],\
              'P':D['/Waveforms/twv/wvfm_amplitude'][samp0:sampN]})
    return result

def read_nonstandard_file(fname, getCountAndReturn=False, shot0=0, nShots=np.Inf, readTX=True, readRX=True):
    """
    Read data from an ATM file
    """
    with h5py.File(fname,'r') as h5f:

        # figure out what shots to read
        #shotMax=h5f['/Waveforms/twv/shot_gate_start'].size
        shotMax=h5f['/laser/calrng'].size
        if getCountAndReturn:
            return shotMax

        nShots=np.minimum(shotMax-shot0, nShots)
        shotN = int(shot0+nShots)
        shot0 = int(shot0)
        # read in some of the data fields
        D_in=dict()

        # read the waveform starts, stops, and lengths for all shots in the file (inefficient, but hard to avoid)
        for key in ('/Waveforms/twv/gate_wvfm_start', '/Waveforms/twv/gate_wvfm_length', '/Waveforms/twv/gates_position'):
            D_in[key]=np.array(h5f[key], dtype=int)
        # read in the gate info for the shots we want to read
        for key in( '/Waveforms/twv/shot_gate_start', '/Waveforms/twv/shot_gate_count'):
            D_in[key]=np.array(h5f[key][shot0:shotN], dtype=int)

        #read in the geolocation and time
        try:
            for key in ('/footprint/latitude','/footprint/longitude','/footprint/elevation','/laser/scan_azimuth', '/laser/calrng'):
                D_in[key]=np.array(h5f[key][shot0:shotN])
        except KeyError:
            print("failed to read " + key)
            pass
        D_in['/Waveforms/twv/shot_seconds_of_day']=np.array(h5f['/Waveforms/twv/shot_seconds_of_day'][shot0:shotN])
        # read the sampling interval, convert to ns
        dt=np.float64(h5f['/Waveforms/twv/sampleRate'])*1.e9
        # guess that gate_xmit is one if there are two returns, two if there are 3
        gate_xmt = np.ones_like(D_in['/Waveforms/twv/shot_gate_count'])
        gate_xmt[D_in['/Waveforms/twv/shot_gate_count']==3] = 2
        D_in['/laser/gate_xmt'] = gate_xmt
        
        gate_rcv = D_in['/Waveforms/twv/shot_gate_count']
        D_in['/laser/gate_rcv'] = gate_rcv
        
        # figure out what samples to read from the 'amplitude' dataset
        gate0=D_in['/Waveforms/twv/shot_gate_start'][0]-1 + gate_xmt[0]-1
        sample_start = D_in['/Waveforms/twv/gate_wvfm_start'][gate0]-1
        gateN = D_in['/Waveforms/twv/shot_gate_start'][-1]-1 + gate_rcv[-1]-1
        sample_end =  D_in['/Waveforms/twv/gate_wvfm_start'][gateN] + D_in['/Waveforms/twv/gate_wvfm_length'][gateN]
        # ... and read the amplitude.  The sample_start variable will get subtracted off
        # subsequent indexes into the amplitude array
        key='/Waveforms/twv/wvfm_amplitude'
        D_in[key]=np.array(h5f[key][sample_start:sample_end+1], dtype=int)

        TX=list()
        NTX=[]
        RX=list()
        tx_samp0=list()
        rx_samp0=list()
        RX=list()
        nPeaks=list()
        rxBuffer=np.zeros(192)+np.NaN
        txBuffer=np.zeros(192)+np.NaN
        for shot in range(int(nShots)):
            wfd=_read_wf(D_in, shot, starting_sample=sample_start, read_tx=readTX, read_rx=readRX)
            if readTX:
                nTX=np.minimum(190, wfd['tx']['P'].size)
                txBuffer[0:nTX]=wfd['tx']['P'][0:nTX]
                txBuffer[nTX+1:]=np.NaN
                TX.append(txBuffer.copy())
                #TX.append(wfd['tx']['P'][0:160])
                tx_samp0.append(wfd['tx']['pos'])
            if readRX:
                nRX=np.minimum(190, wfd['rx']['P'].size)
                rxBuffer[0:nRX]=wfd['rx']['P'][0:nRX]
                rxBuffer[nRX+1:-1]=np.NaN
                RX.append(rxBuffer.copy())
                rx_samp0.append(wfd['rx']['pos'])
        shots=np.arange(shot0, shotN, dtype=int)
        result={ 'az':D_in['/laser/scan_azimuth'],'dt':dt}
        for field in ['elevation','latitude','longitude']:
            try:
                result[field]=D_in['/footprint/'+field]
            except KeyError:
                print(f'\t read_ATM_WFs.py: field footprint/{field} not readable')
                pass
        result['calrng']=D_in['/laser/calrng']
        result['seconds_of_day']=D_in['/Waveforms/twv/shot_seconds_of_day']

        if readTX:
            TX=np.c_[TX].transpose()
            result['TX']=waveform(np.arange(TX.shape[0])*dt, TX, shots=shots, \
                                t0=tx_samp0*dt, \
                                seconds_of_day=D_in['/Waveforms/twv/shot_seconds_of_day'])

        if readRX:
            RX=np.c_[RX].transpose()
            nPeaks=np.ones(RX.shape[1], dtype=int)
            result['RX']=waveform(np.arange(RX.shape[0])*dt, RX, shots=shots, nPeaks=nPeaks, t0=rx_samp0*dt, seconds_of_day=D_in['/Waveforms/twv/shot_seconds_of_day'])
            result['rx_samp0']=rx_samp0
        result['shots']=shots
    return result

def normalize_wf(wf, noise_samps=[0, 30]):
    """
    normalize an input waveform to have a peak of 1 and a pre-trigger mean of zero
    """
    P=wf.astype(np.float64)
    bg=np.mean(P[noise_samps[0]:noise_samps[1]])
    A=P.max()-bg
    P=(P-bg)/A
    return P

def est_mean_wf(P):
    """
    Estimate the mean waveform from a collection
    """
    notSaturated=np.where(((P==255).sum(axis=0)<2) & (np.all(np.isfinite(P), axis=0)))[0]
    P1=np.zeros((P.shape[0], notSaturated.size))
    for col_out, col in enumerate(notSaturated):
        P1[:,col_out]=normalize_wf(P[:,col])
    good=np.mean(P1[115:125,:], axis=0)<0.1
    wf_bar=np.mean(P1[:,good], axis=1)
    wf_sigma=np.std(P1[:,good], axis=1)
    return wf_bar, wf_sigma

# test code:
TEST=False
if TEST:
    dd=read_nonstandard_file('/Volumes/ice3/ben/ATM_WF/Weddell/20161017_192819.atm5bT5.h5', nShots=100)
