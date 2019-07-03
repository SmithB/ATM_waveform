#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:51:42 2018

@author: ben
"""
#  --startShot 283000
import numpy as np
import matplotlib.pyplot as plt
from ATM_waveform.read_ATM_wfs import read_ATM_file
from ATM_waveform.fit_waveforms import listDict
#from fit_waveforms import waveform
from ATM_waveform.make_rx_scat_catalog import make_rx_scat_catalog
from ATM_waveform.fit_waveforms import fit_catalog

from ATM_waveform.fit_2color_waveforms import fit_catalogs
from ATM_waveform.waveform import waveform
from time import time
import argparse
import h5py
import os

np.seterr(invalid='ignore')
os.environ["MKL_NUM_THREADS"]="1"  # multiple threads don't help that much

def get_overlapping_shots(input_files):
    # get the overlapping shots from the input files
    times={}
    for key in input_files.keys():
        with h5py.File(input_files[key],'r') as h5f:
            times[key]=5.e-5*np.round(np.array(h5f['/waveforms/twv/shot/seconds_of_day'])/5.e-5)
    times_both=np.intersect1d(*[times[key] for key in times.keys()])

    return {key:np.where(np.in1d(times[key], times_both))[0] for key in times.keys()}


def main(args):
    input_files={'IR':args.input_files[0],'G':args.input_files[1]}
    channels = list(input_files.keys())
    # get the waveform count from the output file
    shots= get_overlapping_shots(input_files)
    nWFs=np.minimum(args.nShots, shots[channels[0]].size)
    lastShot=args.startShot+nWFs

    # make the output file
    if os.path.isfile(args.output_file):
        os.remove(args.output_file)
    outDS={}
    for ch in channels:
        outDS[ch]=['R', 'A', 'B', 'delta_t', 't0','tc', 'noise_RMS','shot']
    outDS['both']=['R', 'K0', 'Kmin', 'Kmax', 'sigma']
    outDS['location']=['latitude', 'longitude', 'elevation']
    out_h5 = h5py.File(args.output_file,'w')
    for grp in outDS:
        out_h5.create_group('/'+grp)
        for DS in outDS[grp]:
            out_h5.create_dataset('/'+grp+'/'+DS, (nWFs,), dtype='f8')

    # make groups in the file for transmit data
    for ch in channels:
        for field in ['t0','A','R','shot','sigma']:
            out_h5.create_dataset('/TX/%s/%s' % (ch, field), (nWFs,))

    if args.waveforms:
        for ch in channels:
            out_h5.create_dataset('RX/%s/p' % ch, (192, nWFs))
            out_h5.create_dataset('RX/%s/p_fit' % ch, (192, nWFs))

    TX={}
    # get the transmit pulse
    for file in args.TXfiles:
        for ind, ch in enumerate(channels):
            with h5py.File(args.TXfiles[ind],'r') as fh:
                TX[ch]=waveform(np.array(fh['/TX/t']), np.array(fh['/TX/p']) )
            TX[ch].t -= TX[ch].nSigmaMean()[0]
            TX[ch].tc = 0
            TX[ch].normalize()
    # write the transmit pulse to the file
    for ch in channels:
        out_h5.create_dataset("TX/%s/t" % ch, data=TX[ch].t.ravel())
        out_h5.create_dataset("TX/%s/p" % ch, data=TX[ch].p.ravel())

    # make the library of templates
    WF_library=dict()
    for ind, ch in enumerate(channels):
        WF_library[ch] = dict()
        WF_library[ch].update({0.:TX[ch]})
        WF_library[ch].update(make_rx_scat_catalog(TX[ch], h5_file=args.scat_files[ind]))

    print("Returns:")
    # loop over start vals (one block at a time...)
    # choose how to divide the output
    blocksize=10
    start_vals=args.startShot+np.arange(0, nWFs, blocksize, dtype=int)

    catalog_buffers={ch:listDict() for ch in channels}
    time_old=time()

    sigmas=np.arange(0, 5, 0.25)
    # choose a set of delta t values
    delta_ts=np.arange(-1.5, 2, 0.5)

    D={}
    for shot0 in start_vals:
        outShot0=shot0-args.startShot
        these_shots=np.arange(shot0, np.minimum(shot0+blocksize, lastShot))
        tic=time()
        wf_data={}
        for ch in channels:
            ch_shots=shots[ch][np.in1d( shots[ch], these_shots)]
            # make the return waveform structure
            D=read_ATM_file(input_files[ch], shot0=ch_shots[0], nShots=ch_shots[-1]-ch_shots[0])

            # fit the transmit data for this channel and these pulses

            D['TX']=D['TX'][np.in1d(D['TX'].shots, ch_shots)]
            t_wf_ctr = np.nanmean(D['TX'].t)
            D['TX'].t0 += t_wf_ctr
            D['TX'].t -= t_wf_ctr
            D['TX'].subBG(t50_minus=3)
            D['TX'].tc = D['TX'].threshold_centroid(fraction=0.38)
            #D_out_TX, catalog_buffers[ch]=fit_catalog(D['TX'],  {0:TX[ch]}, np.arange(0, 2, 0.25), np.arange(-2, 3, 0.25), t_tol=0.25, sigma_tol=0.125, return_catalog=True, catalog=catalog_buffers[ch])
            #N_out=len(D_out_TX['A'])
            #for field in ['t0','A','R','shot','sigma']:
            #    out_h5['/TX/%s/%s' % (ch, field)][outShot0:outShot0+N_out]=D_out_TX[field]

            wf_data[ch]=D['RX']
            wf_data[ch]=wf_data[ch][np.in1d(wf_data[ch].shots, ch_shots)]
            t_wf_ctr = np.nanmean(wf_data[ch].t)
            wf_data[ch].t -= t_wf_ctr
            wf_data[ch].t0 += t_wf_ctr
            wf_data[ch].subBG(t50_minus=3)

            if 'latitude' in D:
                # only one channel has geolocation information.  Write it out now.
                these_shots=np.in1d(wf_data[ch].shots, ch_shots)
                for field in outDS['location']:
                    out_h5['location'][field][outShot0:outShot0+len(these_shots)]=D[field][these_shots]

            wf_data[ch].tc = wf_data[ch].threshold_centroid(fraction=0.38)
        D_out, catalog_buffers= fit_catalogs(wf_data, WF_library, sigmas, delta_ts, \
                                            t_tol=0.25, sigma_tol=0.25, return_data_est=args.waveforms, \
                                            return_catalogs=True,  catalogs=catalog_buffers, params=outDS)
               
        delta_time=time()-tic
        N_out=D_out['both']['R'].size

        for ch in ['both']+channels:
            for key in outDS[ch]:
                try:
                    out_h5[ch][key][outShot0:outShot0+N_out]=D_out[ch][key].ravel()
                except OSError:
                    print("OSError for key=%s, outshot0=%d, outshotN=%d, nDS=%d"% (key, outShot0, outShot0+N_out, out_h5[key].size))

        if args.waveforms:
            for ch in channels:
                out_h5['RX/'+ch+'/p_fit'][:, outShot0:outShot0+N_out] = D_out[ch]['wf_est']
                out_h5['RX/'+ch+'/p'][:, outShot0:outShot0+N_out] = wf_data[ch].p

        print("  shot=%d out of %d, N_keys=%d, dt=%5.1f" % (shot0+blocksize, start_vals[-1]+blocksize, len(catalog_buffers['G'].keys()), delta_time))
    print("   time to fit RX=%3.2f" % (time()-time_old))

    if args.waveforms:
        out_h5.create_dataset('RX/t', data=D_out.t.ravel())

    out_h5.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Fit the waveforms from an ATM file with a set of scattering parameters')
    parser.add_argument('input_files', type=str, nargs=2)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--startShot', '-s', type=int, default=0)
    parser.add_argument('--scat_files', '-f', type=str, nargs=2, default=None)
    parser.add_argument('--nShots', '-n', type=int, default=np.Inf)
    parser.add_argument('--DOPLOT', '-P', action='store_true')
    parser.add_argument('--skipRX', action='store_true', default=False)
    parser.add_argument('--everyNTX', type=int, default=100)
    parser.add_argument('--TXfiles', '-T', type=str, nargs=2, default=None)
    parser.add_argument('--waveforms', '-w', action='store_true', default=False)
    args=parser.parse_args()
    main(args)