# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 09:47:58 2019

@author: ben
"""
import numpy as np
from IS2_calval.waveform import waveform
from IS2_calval.fit_waveforms  import gaussian
from IS2_calval.fit_ATM_scat import proc_RX, make_rx_scat_catalog
import h5py
import sys
import os
import scipy.stats as sps
import matplotlib.pyplot as plt

def plot_scat_fit_error(in_file, A_val=200, sigma_val=0, ax=None):
    if ax is None:
        plt.figure();
        plt.clf()
        ax=plt.subplot(111)
        no_show=False
    else:
        plt.sca(ax)
        no_show=True
    with h5py.File(in_file,'r') as h5f:
        A=np.array(h5f['A'])
        sigma=np.array(h5f['sigma'])
        ii=((A==A_val) & (sigma==sigma_val) & (np.array(h5f['K0'])>0))
        nData=np.sum(ii)
        #print(nData)
        xx=np.zeros((nData,2))
        for kk in [0, 1]:
            xx[:,kk]=h5f['K0'][ii]
        yy=np.zeros((nData,2))
        yy[:,0]=h5f['K16'][ii]
        yy[:,1]=h5f['K84'][ii]
        uX=np.unique(xx[:,0])
        for xi in uX:
            E0=np.min(yy[xx[:,0]==xi,0])
            E1=np.max(yy[xx[:,0]==xi,1])
            plt.loglog(np.array([xi, xi]), np.array([E0, E1]),'r', marker='.', linewidth=3)
    ax.loglog(xx[:,kk], xx[:,kk])
    ax.set_xlabel('$r_{eff}$ input, m')
    ax.set_ylabel('$r_{eff}$ recovered, m')
    ax.set_title('A=%d, $\sigma$=%1.1f ns' % (A_val, sigma_val))
    if not no_show:
        plt.show()

def broadened_WF(TX, sigma):
    nK=3*np.ceil(sigma/TX.dt)
    tK=np.arange(-nK, nK+1)*TX.dt
    K=gaussian(tK, 0, sigma)
    K=K/np.sum(K)
    return waveform(TX.t, np.convolve(TX.p.ravel(), K,'same'))

def errors_for_one_scat_file(scat_file, TX_file, out_file=None):

    N_WFs=256
    sigma_vals=[0, 0.25, 0.5, 1, 2]
    A_vals=[ 25., 50., 100., 200.]

    with h5py.File(TX_file) as h5f:
        TX=waveform(np.array(h5f['/TX/t']),np.array(h5f['/TX/p']))
    TX.t -= TX.nSigmaMean()[0]
    TX.tc = 0
    WF_library = dict()
    WF_library.update({0.:TX})
    if scat_file is not None:
        WF_library.update(make_rx_scat_catalog(TX, h5_file=scat_file))
    K0_vals=np.sort(list(WF_library))[::5]
    #K0_vals=np.sort(list(WF_library))[-2:]

    N_out=len(sigma_vals)*len(A_vals)*len(K0_vals)
    Dstats={field:np.zeros(N_out)+np.NaN for field in {'K16','K84', 'sigma16', 'sigma84', 'sigma','A','K0','Kmed', 'Ksigma','Ksigma_est'}}

    catalogBuffer=None
    ii=0

    for key in K0_vals:
        for sigma in sigma_vals:
            for A in A_vals:
                if sigma > 0:
                    BW=broadened_WF(WF_library[key], sigma)
                else:
                    BW=waveform(WF_library[key].t, WF_library[key].p)
                BW.normalize()
                WFs=waveform(TX.t, np.tile(A*BW.p, [1, N_WFs])+np.random.randn(BW.p.size*N_WFs).reshape(BW.p.size, N_WFs))
                WFs.shots=np.arange(WFs.size)
                D_out, rxData, D, catalogBuffer = proc_RX(None, np.arange(N_WFs), rxData=WFs, sigmas=np.array([0., 1.]), deltas=np.arange(-0.5, 1, 0.5), TX=TX, WF_library=WF_library, catalogBuffer=catalogBuffer)
                sR=sps.scoreatpercentile(D_out['sigma'], [16, 84])
                Dstats['A'][ii]=A
                Dstats['sigma'][ii]=sigma
                Dstats['K0'][ii]=key
                Dstats['sigma16'][ii]=sR[0]
                Dstats['sigma84'][ii]=sR[1]
                KR=sps.scoreatpercentile(D_out['K0'], [16, 84])
                Dstats['K16'][ii]=KR[0]
                Dstats['K84'][ii]=KR[1]
                Dstats['Kmed'][ii]=np.nanmedian(D_out['K0'])
                Dstats['Ksigma_est'][ii]=np.nanmedian(D_out['Kmax']-D_out['Kmin'])
                Dstats['Ksigma'][ii]=np.nanstd(D_out['K0'])
                print([key, sigma, A, KR-key, Dstats['Ksigma'][ii]])
                ii += 1

    print("yep")
    if out_file is not None:
        if os.path.isfile(out_file):
            os.remove(out_file)
        out_h5=h5py.File(out_file,'w')
        for kk in Dstats:
            out_h5.create_dataset(kk, data=Dstats[kk])
        out_h5.close()
    return Dstats


#if len(sys.argv)==4:
#    scat_file=sys.argv[1]
#    TX_file=sys.argv[2]
#    out_file=sys.argv[3]
#    Dstats=errors_for_one_scat_file(scat_file, TX_file, out_file=out_file)
#else:
#    plot_scat_fit_error(sys.argv[1], A_val=200, sigma_val=0)
#    plot_scat_fit_error(sys.argv[1], A_val=50, sigma_val=0)
#
#    plot_scat_fit_error(sys.argv[1], A_val=200, sigma_val=1)
#    plot_scat_fit_error(sys.argv[1], A_val=50, sigma_val=1)
#
#
#    plot_scat_fit_error(sys.argv[1], A_val=200, sigma_val=2)
#    plot_scat_fit_error(sys.argv[1], A_val=50, sigma_val=2)
