# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 09:47:58 2019

@author: ben
"""
import numpy as np
from ATM_waveform.waveform import waveform
from ATM_waveform.fit_waveforms  import gaussian, listDict
from ATM_waveform.fit_ATM_scat import make_rx_scat_catalog
from ATM_waveform.fit_ATM_scat_2color import fit_catalogs
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

def errors_for_one_scat_file(scat_files, TX_files, out_file=None):

    sigmas=np.arange(0, 5, 0.25)
    # choose a set of delta t values
    delta_ts=np.arange(-1., 1.5, 0.5)

    N_WFs=256
    channels=['r','g']
    TX={}
    # get the transmit pulse
    for ind, ch in enumerate(channels):
        with h5py.File(args.TXfiles[ind],'r') as fh:
            TX[ch]=waveform(np.array(fh['/TX/t']), np.array(fh['/TX/p']) )
        TX[ch].t -= TX[ch].nSigmaMean()[0]
        TX[ch].tc = 0
        TX[ch].normalize()

    # initialize the library of templates for the transmit waveforms
    TX_library={}
    for ind, ch in enumerate(channels):
        TX_library[ch] = listDict()
        TX_library[ch].update({0.:TX[ch]})

    # initialize the library of templates for the received waveforms
    WF_library=dict()
    for ind, ch in enumerate(channels):
        WF_library[ch] = dict()
        WF_library[ch].update({0.:TX[ch]})
        WF_library[ch].update(make_rx_scat_catalog(TX[ch], h5_file=args.scat_files[ind]))

    sigma_vals=[0, 0.25, 1., 2.]
    A_vals=[50., 100., 200.]
    K0_vals=10**np.arange(-4.5, -1.5, 0.125)
    N_out=len(sigma_vals)*len(A_vals)*len(K0_vals)
    Dstats={field:np.zeros(N_out)+np.NaN for field in {'K16','K84', 'sigma16', 'sigma84', 'sigma','A','K0','Kmed', 'Ksigma','Ksigma_est'}}

    ii=0
    noise_RMS={'g':1, 'r':5}
    catalog_buffers={'g':{}, 'r':{}}
    for key in K0_vals:
        for sigma in sigma_vals:
            for A in A_vals:
                WFs={}
                for ch in channels:
                    if sigma > 0:
                        BW=broadened_WF(WF_library[ch][key], sigma)
                    else:
                        BW=waveform(WF_library[ch][key].t, WF_library[key].p)
                    BW.normalize()
                    WFs[ch]=waveform(TX.t, np.tile(A*BW.p, [1, N_WFs])+\
                       noise_RMS[ch]*np.random.randn(BW.p.size*N_WFs).reshape(BW.p.size, N_WFs))
                    WFs[ch].shots=np.arange(WFs[ch].size)


                D_out, catalog_buffers= fit_catalogs(wf_data, WF_library, sigmas, delta_ts, \
                                            t_tol=0.25, sigma_tol=0.25)
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

if __name__=="__main__":

    import argparse
    n_chan=int(sys.argv[1])
    del(sys.argv[1])
    parser = argparse.ArgumentParser(description='Calculate error ranges for synthetic data based on a set of input pulses.  The first argument gives the number of channels')
    parser.add_argument('output_file', type=str)
    parser.add_argument('--scat_files', '-f', type=str, nargs=n_chan, default=None)
    parser.add_argument('--TXfiles', '-T', type=str, nargs=n_chan, default=None)
    args=parser.parse_args()
    errors_for_one_scat_file(args.scat_files, args.TXfiles, args.output_file)




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
