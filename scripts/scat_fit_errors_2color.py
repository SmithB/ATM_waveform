# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 09:47:58 2019

@author: ben
"""
import numpy as np
import ATM_waveform as aw
import h5py
import sys
import os
import time
import scipy.stats as sps
import matplotlib.pyplot as plt

def plot_scat_fit_error(in_file, A_val=1, sigma_val=0, ax=None, line_color='r'):

    if ax is None:
        plt.figure();
        plt.clf()
        ax=plt.subplot(111)
        no_show=False
    else:
        plt.sca(ax)
        no_show=True
    with h5py.File(in_file,'r') as h5f:
        A=np.array(h5f['A_scale'])
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
            plt.loglog(np.array([xi, xi]), np.array([E0, E1]), line_color, marker='*', linewidth=3)
    ax.loglog(xx[:,kk], xx[:,kk])
    ax.set_xlabel('$r_{eff}$ input, m')
    ax.set_ylabel('$r_{eff}$ recovered, m')
    ax.set_title('A=%2.1f, $\sigma$=%1.1f ns' % (A_val, sigma_val))
    if not no_show:
        plt.show()

def broadened_WF(WF, sigma):
    """
    Generate a version of WF broadened by a Gaussian of width sigma
    """
    return aw.waveform(WF.t, aw.broaden_p(WF, sigma))

def make_sim_WFs(N_WFs, WF_library, key, sigma, noise_RMS, amp_scale):
    """
    Generate simulated waveforms for a specified grain size (K0)

    Inputs:
        N_WFs: number of pulses in each waveform object
        WF_library: a dict of waveform catalogs (keys specify different channels)
        key: entry within the WF_library from whcih to generate the waveforms
        sigma: pulse spreading time
        noise_RMS: dict giving the RMS noise values for each channel
        amp_scale: scale by which each channel's ampitude is multiplied
    outputs:
        WFs: dict giving waveform data for each channel.
    """

    WFs={}
    for ch in WF_library.keys():
        if sigma > 0:
            BW=broadened_WF(WF_library[ch][key], sigma)
        else:
            BW=aw.waveform(WF_library[ch][key].t, WF_library[ch][key].p)
        #BW.normalize()
        WFs[ch]=aw.waveform(BW.t, np.tile(amp_scale[ch]*BW.p, [1, N_WFs])+\
           noise_RMS[ch]*np.random.randn(BW.p.size*N_WFs).reshape(BW.p.size, N_WFs))
        WFs[ch].noise_RMS=noise_RMS[ch]+np.zeros(WFs[ch].size)
        WFs[ch].shots=np.arange(WFs[ch].size)
    return WFs



def errors_for_one_scat_file(test_scat_files, fit_scat_files, TX_files, channels, N_WFs=256, noise_amp=1, out_file=None):
    
    sigmas=np.arange(0, 5, 0.25)
    # choose a set of delta t values
    delta_ts=np.arange(-1., 1.5, 0.5)

    TX={}
    # get the transmit pulse
    for ind, ch in enumerate(channels):
        with h5py.File(TX_files[ind],'r') as fh:
            TX[ch]=aw.waveform(np.array(fh['/TX/t']), np.array(fh['/TX/p']) )
        TX[ch].t -= TX[ch].nSigmaMean()[0]
        TX[ch].tc = 0
        TX[ch].normalize()

    # initialize the library of templates for the transmit waveforms
    TX_library={}
    for ind, ch in enumerate(channels):
        TX_library[ch] = aw.listDict()
        TX_library[ch].update({0.:TX[ch]})

    # initialize the library of templates for the test_waveforms
    test_WF_library=dict()
    for ind, ch in enumerate(channels):
        test_WF_library[ch] = dict()
        test_WF_library[ch].update({0.:TX[ch]})
        test_WF_library[ch].update(aw.make_rx_scat_catalog(TX[ch], h5_file=test_scat_files[ind]))

        
    # initialize the library of templates for the received waveforms
    WF_library=dict()
    for ind, ch in enumerate(channels):
        WF_library[ch] = dict()
        WF_library[ch].update({0.:TX[ch]})
        WF_library[ch].update(aw.make_rx_scat_catalog(TX[ch], h5_file=fit_scat_files[ind]))

    out_fields=['K16','K84','K5','K95','sigma16', 'sigma84', 'sigma','A_scale','K0','Kmed', 'Ksigma','Ksigma_est', 'N','fitting_time']
    for ch in channels:
        out_fields.append('A_'+ch)

    noise_RMS={'G':noise_amp, 'IR':0.25*noise_amp}
    unit_amp_target={'G':180, 'IR':140}
    sigma_vals=[0, 0.5, 1, 2]
    A_scale=[0.5, 0.75, 1, 1.25]
    K0_vals=np.array(list(test_WF_library['G'].keys()))
    N_out=len(sigma_vals)*len(A_scale)*len(K0_vals)
    Dstats={field:np.zeros(N_out)+np.NaN for field in out_fields}
    # calculate waveforms with no scaling applied
    WF_unscaled=make_sim_WFs(1, test_WF_library, list(test_WF_library['G'].keys())[1], 0, {'G':0, 'IR':0}, {'G':1, 'IR':1})
    ii=0
    for key in K0_vals:
        for sigma in sigma_vals:
            for A in A_scale:
                # calculate scaling to apply to the waveforms to achieve the
                # target peak amplitude (could be done outside the loop)
                amp_scale={ch:A*unit_amp_target[ch]/WF_unscaled[ch].p.max() for ch in channels}

                #Calculate the noise-free waveforms
                WF_expected=make_sim_WFs(1, test_WF_library, key, sigma,\
                                         {ch:0. for ch in channels}, amp_scale)
                if np.min([WF_expected[ch].p.max()/noise_RMS[ch] for ch in channels]) < 3:
                    continue
                # calculate scaled and broadened waveforms
                WFs=make_sim_WFs(N_WFs, test_WF_library, key, sigma, noise_RMS, amp_scale)
                # fit the waveforms
                tic=time.time()
                D_out= aw.fit_catalogs(WFs, WF_library, sigmas, delta_ts, \
                                            t_tol=0.25, sigma_tol=0.25)
                Dstats['fitting_time'][ii]=time.time()-tic
                for ch in channels:
                    Dstats['A_'+ch][ii]=WF_expected[ch].p.max()
                sR=sps.scoreatpercentile(D_out['both']['sigma'], [16, 84])
                Dstats['A_scale'][ii]=A
                Dstats['sigma'][ii]=sigma
                Dstats['K0'][ii]=key
                Dstats['sigma16'][ii]=sR[0]
                Dstats['sigma84'][ii]=sR[1]
                KR=sps.scoreatpercentile(D_out['both']['K0'], [16, 84])
                Dstats['K16'][ii]=KR[0]
                Dstats['K84'][ii]=KR[1]
                KR=sps.scoreatpercentile(D_out['both']['K0'], [5, 95])
                Dstats['K5'][ii]=KR[0]
                Dstats['K95'][ii]=KR[1]
                Dstats['Kmed'][ii]=np.nanmedian(D_out['both']['K0'])
                Dstats['Ksigma_est'][ii]=np.nanmedian(D_out['both']['Kmax']-D_out['both']['Kmin'])
                Dstats['Ksigma'][ii]=np.nanstd(D_out['both']['K0'])
                Dstats['N'][ii]=np.sum(np.isfinite(D_out['both']['K0']))
                print('K0=%2.2g, sigma=%2.2f, A=%2.2f, ER=[%2.2g, %2.2g], E=%2.2g' %(key, sigma, A, KR[0]-key, KR[1]-key, Dstats['Ksigma'][ii]))
                ii += 1

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
    parser.add_argument('-c','--colors', type=str, nargs=n_chan, default=['IR','G'])
    parser.add_argument('--scat_files', '-f', type=str, nargs=n_chan, default=None)
    parser.add_argument('--test_scat_files', type=str, nargs=n_chan, default=None)
    parser.add_argument('--noise_amp', type=float, default=1)
    parser.add_argument('--TXfiles', '-T', type=str, nargs=n_chan, default=None)
    args=parser.parse_args()
    
    if args.test_scat_files is None:
        args.test_scat_files=args.scat_files

    Dstats=errors_for_one_scat_file(args.test_scat_files, args.scat_files, args.TXfiles, args.colors,  out_file=args.output_file)

# Example command lines:
#2 2color_test.h5  -f SRF_IR_full.h5 SRF_green_full.h5  -T TX_IR.h5 TX_green.h5
#1 green_only_test.h5  -f SRF_green_full.h5  -T TX_green.h5 -c G
