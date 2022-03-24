#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:51:42 2018

@author: ben
"""
#  2 ./IR/ILNIRW1B_20190906_132900.atm6CT7.h5 ./green/ILNSAW1B_20190906_132900.atm6DT7.h5 20190906_132900.out.h5 -f data/SRF_IR_full.h5 data/SRF_green_full.h5 -T TX_IR.h5 TX_green.h5 -c IR G
#  1 ./green/ILNSAW1Bprelim_20181028_013800.atm6DT7.filt.h5 20181028_013800_G_out.h5 -f SRF_green_full.h5 -T  TX_green.h5 -c G

import os
os.environ["MKL_NUM_THREADS"]="1"  # multiple threads don't help that much
import numpy as np
#import matplotlib.pyplot as plt
from ATM_waveform.read_ATM_wfs import read_ATM_file
#from fit_waveforms import waveform
from ATM_waveform.make_rx_scat_catalog import make_rx_scat_catalog

from ATM_waveform.fit_2color_waveforms import fit_catalogs
from ATM_waveform import waveform, WFcatalog
from time import time
import argparse
import h5py

import sys

np.seterr(invalid='ignore')


def choose_shots(input_files, skip=None):
    # get the overlapping shots from the input files
    #
    # matches the shots from input files by their "seconds_of_day_field,' using a default
    # resolution of 5e-5 s
    times={}
    for key in input_files.keys():
        with h5py.File(input_files[key],'r') as h5f:
            times[key]=5.e-5*np.round(np.array(h5f['/waveforms/twv/shot/seconds_of_day'])/5.e-5)
    if len(input_files) > 1:
        times_both=np.intersect1d(*[times[key] for key in times.keys()])
    else:
        key=list(input_files.keys())[0]
        times_both=times[key]
    if skip is not None:
        times_both=times_both[::skip]
    return {key:np.flatnonzero(np.in1d(times[key], times_both)) for key in times.keys()}

def main(args):
    # main method : open the input files, create output files, process waveforms

    input_files={}
    for ii, ch in enumerate(args.ch_names):
        if ch != 'None':
            input_files[ch]=args.input_files[ii]
    channels = list(input_files.keys())

    # get the waveform count from the output file
    shots= choose_shots(input_files, args.reduce_by)
    nWFs=np.minimum(args.nShots, shots[channels[0]].size)
    lastShot=np.minimum(args.startShot+args.nShots, len(shots[channels[0]]))
    nWFs = lastShot-args.startShot+1

    # make the output file
    if os.path.isfile(args.output_file):
        os.remove(args.output_file)
    # define the output datasets
    outDS={}
    outDS['ch']=['R', 'A', 'B', 'delta_t', 't0','tc', 't_shift', 'noise_RMS','shot','Amax','seconds_of_day','nPeaks']
    outDS['both']=['R', 'K0', 'Kmin', 'Kmax', 'sigma']
    outDS['location']=['latitude', 'longitude', 'elevation']
    out_h5 = h5py.File(args.output_file,'w')
    for grp in ['both','location']:
        out_h5.create_group('/'+grp)
        for DS in outDS[grp]:
            out_h5.create_dataset('/'+grp+'/'+DS, (nWFs,), dtype='f8')
    for ch in channels:
        out_h5.create_group('/'+ch)
        for DS in outDS['ch']:
            out_h5.create_dataset('/'+ch+'/'+DS, (nWFs,), dtype='f8')

    # make groups in the file for transmit data
    for ch in channels:
        for field in ['t0','A','R','shot','sigma']:
            out_h5.create_dataset('/TX/%s/%s' % (ch, field), (nWFs,))

    if args.waveforms:
        for ch in channels:
            out_h5.create_dataset('RX/%s/p' % ch, (192, nWFs))
            out_h5.create_dataset('RX/%s/p_fit' % ch, (192, nWFs))
            out_h5.create_dataset('RX/%s/t_shift' % ch, (nWFs,))
    TX={}
    # get the transmit pulse
    for ind, ch in enumerate(channels):
        with h5py.File(args.TXfiles[ind],'r') as fh:
            TX[ch]=waveform(np.array(fh['/TX/t']), np.array(fh['/TX/p']) )
        TX[ch].t -= TX[ch].nSigmaMean()[0]
        TX[ch].tc = 0
        TX[ch].p_squared=TX[ch].p*TX[ch].p
        TX[ch].mask=np.isfinite(TX[ch].p)
        TX[ch].normalize()
    # write the transmit pulse to the file
    for ch in channels:
        out_h5.create_dataset("/TX/%s/t" % ch, data=TX[ch].t.ravel())
        out_h5.create_dataset("/TX/%s/p" % ch, data=TX[ch].p.ravel())

    # initialize the library of templates for the transmit waveforms
    TX_library={}
    for ind, ch in enumerate(channels):
        TX_library[ch] ={}
        TX_library[ch].update({0.:TX[ch]})

    # initialize the library of templates for the received waveforms
    WF_library=dict()
    for ind, ch in enumerate(channels):
        WF_library[ch] = dict()
        #WF_library[ch].update({0.:TX[ch]})
        WF_library[ch].update(make_rx_scat_catalog(TX[ch], h5_file=args.scat_files[ind]))


    print("Returns:")
    # loop over start vals (one block at a time...)
    # choose how to divide the output
    blocksize=1000
    start_vals=args.startShot+np.arange(0, nWFs, blocksize, dtype=int)


    catalog_buffers={ch:WFcatalog(TX[ch].nSamps, TX[ch].dt[0], t=TX[ch].t) for ch in channels}
    TX_catalog_buffers={ch:WFcatalog(TX[ch].nSamps, TX[ch].dt[0], t=TX[ch].t) for ch in channels}
    time_old=time()

    sigmas=np.arange(0, 5, 0.25)
    # choose a set of delta t values
    delta_ts=np.arange(-1., 1.5, 0.5)

    D={}
    for shot0 in start_vals:
        outShot0=shot0-args.startShot
        these_shots=np.arange(shot0, np.minimum(shot0+blocksize, lastShot), dtype=int)
        if len(these_shots) < 1:
            continue
        #tic=time()
        wf_data={}
        for ch in channels:
            ch_shots=shots[ch][these_shots]
            # make the return waveform structure
            try:
                D=read_ATM_file(input_files[ch], shot0=ch_shots[0], nShots=ch_shots[-1]-ch_shots[0]+1)
            except Exception as e:
                print(f"caught exception for channel {ch} for shots {ch_shots[0]} to {ch_shots[1]}:")
                print(e)
                continue
            # fit the transmit data for this channel and these pulses
            D['TX']=D['TX'][np.in1d(D['TX'].shots, ch_shots)]
            # set t0 to the center of the waveform
            t_wf_ctr = np.nanmean(D['TX'].t)
            D['TX'].t0 += t_wf_ctr
            D['TX'].t -= t_wf_ctr
            # subtract the background noise
            D['TX'].subBG(t50_minus=3)
            # calculate tc (centroid time relative to t0)
            D['TX'].tc = D['TX'].threshold_centroid(fraction=0.38)
            #D_out_TX, catalog_buffers= fit_catalogs({ch:D['TX']}, TX_library, sigmas, delta_ts, \
            #                            t_tol=0.25, sigma_tol=0.25,  \
            #                            return_catalogs=True,  catalogs=catalog_buffers)
            D_out_TX = fit_catalogs({ch:D['TX']}, TX_library, sigmas, delta_ts, \
                                        t_tol=0.25, sigma_tol=0.25,  \
                                        return_catalogs=False,  catalogs=TX_catalog_buffers, params=outDS)
            N_out=len(D_out_TX[ch]['A'])
            for field in ['t0','A','R','shot']:
                out_h5['/TX/%s/%s' % (ch, field)][outShot0:outShot0+N_out]=D_out_TX[ch][field].ravel()
            out_h5['/TX/%s/%s' % (ch, 'sigma')][outShot0:outShot0+N_out]=D_out_TX['both']['sigma'].ravel()

            wf_data[ch]=D['RX']
            wf_data[ch]=wf_data[ch][np.in1d(wf_data[ch].shots, ch_shots)]
            # identify the samples that have clipped amplitudes:
            clipped=wf_data[ch].p >= 255
            t_wf_ctr = np.nanmean(wf_data[ch].t)
            wf_data[ch].t -= t_wf_ctr
            wf_data[ch].t0 += t_wf_ctr
            wf_data[ch].subBG(t50_minus=3)

            if 'latitude' in D:
                # only one channel has geolocation information. Copy it, will use the 'shot' field to match it to the output data
                loc_info={ff:D[ff] for ff in outDS['location']}
                loc_info['channel']=ch
                loc_info['shot']=D['shots']

            wf_data[ch].tc = wf_data[ch].threshold_centroid(fraction=0.38)
            wf_data[ch].p[clipped]=np.NaN
        # now fit the returns with the waveform model
        tic=time()
        D_out, catalog_buffers= fit_catalogs(wf_data, WF_library, sigmas, delta_ts, \
                                            t_tol=0.25, sigma_tol=0.25, return_data_est=args.waveforms, \
                                            return_catalogs=True,  catalogs=catalog_buffers, params=outDS)

        delta_time=time()-tic

        # write out the fit information
        N_out=D_out['both']['R'].size
        for ch in channels:
            for key in outDS['ch']:
                try:
                    out_h5[ch][key][outShot0:outShot0+N_out]=D_out[ch][key].ravel()
                except OSError:
                    print("OSError for channel %s,  key=%s, outshot0=%d, outshotN=%d, nDS=%d"% (ch, key, outShot0, outShot0+N_out, out_h5[key].size))
        for key in outDS['both']:
            try:
                out_h5['both'][key][outShot0:outShot0+N_out]=D_out['both'][key].ravel()
            except OSError:
                print("OSError for both channels, key=%s, outshot0=%d, outshotN=%d, nDS=%d"% (key, outShot0, outShot0+N_out, out_h5[key].size))

        # write out the location info
        loc_ind=np.flatnonzero(np.in1d(loc_info['shot'], D_out[loc_info['channel']]['shot']))
        for field in outDS['location']:
            out_h5['location'][field][outShot0:outShot0+N_out]=loc_info[field][loc_ind]

        # write out the waveforms
        if args.waveforms:
            for ch in channels:
                out_h5['RX/'+ch+'/p_fit'][:, outShot0:outShot0+N_out] = np.squeeze(D_out[ch]['wf_est']).T
                out_h5['RX/'+ch+'/p'][:, outShot0:outShot0+N_out] = wf_data[ch].p
                out_h5['RX/'+ch+'/t_shift'][outShot0:outShot0+N_out] = D_out[ch]['t_shift'].ravel()

        print("  shot=%d out of %d, N_keys=%d, dt=%5.1f" % (shot0+blocksize, start_vals[-1]+blocksize, len(catalog_buffers['G'].index.keys()), delta_time))
    print("   time to fit RX=%3.2f" % (time()-time_old))

    if args.waveforms:
        for ch in channels:
            out_h5.create_dataset('RX/'+ch+'/t', data=wf_data[ch].t.ravel())

    out_h5.close()

# parse the input arguments
if __name__=="__main__":

    n_chan=int(sys.argv[1])
    del(sys.argv[1])
    parser = argparse.ArgumentParser(description='Fit the waveforms from an ATM file with a set of scattering parameters.\n The first argument gives the number of channels')
    parser.add_argument('input_files', type=str, nargs=n_chan)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--startShot', '-s', type=int, default=0)
    parser.add_argument('--scat_files', '-f', type=str, nargs=n_chan, default=None)
    parser.add_argument('--nShots', '-n', type=int, default=np.Inf)
    parser.add_argument('--DOPLOT', '-P', action='store_true')
    parser.add_argument('--skipRX', action='store_true', default=False)
    parser.add_argument('--everyNTX', type=int, default=100)
    parser.add_argument('--reduce_by', '-r', type=int, default=1)
    parser.add_argument('--TXfiles', '-T', type=str, nargs=n_chan, default=None)
    parser.add_argument('--waveforms', '-w', action='store_true', default=False)
    parser.add_argument('--ch_names','-c', type=str, nargs=n_chan, default=['IR','G'])
    args=parser.parse_args()
    main(args)
#2 IR/ILNIRW1B_20190906_132900.atm6CT7.h5 green/ILNSAW1B_20190906_132900.atm6DT7.h5 -o 20190906_132900.out.h5 -f data/SRF_IR_full.h5 data/SRF_green_full.h5 -T TX_IR.h5 TX_green.h5 -s 432544 -n 100 -c IR G
