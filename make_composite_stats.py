#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 22:28:01 2020

@author: ben
"""
import numpy as np
from make_rx_scat_catalog import make_rx_scat_catalog
import h5py
from ATM_waveform.waveform import waveform
import pointCollection as pc
from ATL11 import RDE
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import scipy.integrate as sciint
from scipy.interpolate import interp1d
import os

def interp_inv(fi, t0, f0):
    ii_plus =np.min(np.flatnonzero(f0>=fi))
    ii_minus=np.max(np.flatnonzero(f0<=fi))
    if f0[ii_plus]==f0[ii_minus]:
        return (t0[ii_plus]+t0[ii_minus])/2
    return t0[ii_minus]+(fi-f0[ii_minus])*(t0[ii_plus]-t0[ii_minus])/(f0[ii_plus]-f0[ii_minus])

def TX_corr(TX, T_win, sigma, TX_sigma=None):
    if TX_sigma is None:
        TX_sigma=TX.robust_spread()
    sigma_extra=np.sqrt(np.max([0, sigma**2-TX_sigma**2]))
    TXb=waveform(TX.t, TX.p.copy()).broaden(sigma_extra)
    return wf_med_bar(TXb, T_win, t_tol=0.01*0.25)

def wf_med_bar(WF, T_win, t_tol=0.01):
    IPT=sciint.cumtrapz((WF.p*WF.t).ravel(), WF.t.ravel(), initial=0)
    IP=sciint.cumtrapz(WF.p.ravel(), WF.t.ravel(), initial=0)
    C=lambda tt, W:np.diff(np.interp(tt+np.array([-W/2,W/2]), WF.t.ravel(), IPT))/np.diff(np.interp(tt+np.array([-W/2,W/2]), WF.t.ravel(), IP))
    tc=C(0, T_win)
    tc_last=tc-10*t_tol
    while np.abs(tc-tc_last) > t_tol:
        tc_last=tc.copy()
        tc=C(tc, T_win)
    # find the integrand of IP at Tc-T_win/2, Tc+T_win/2
    IPW=np.interp(np.array([tc-T_win/2, tc+T_win/2]), WF.t.ravel(), IP)
    # find the integrand value at the median
    t_med=interp_inv(np.mean(IPW), WF.t, IP)
    #if plot:
    #    plt.plot(WF.t, (IP-IPW[0])/(IPW[1]-IPW[0]))
    #    plt.plot(t_med, (np.mean(IPW)-IPW[0])/(IPW[1]-IPW[0]),'*')
    #    plt.gca().vlines([tc-T_win/2, tc+T_win/2], 0, 1)
    
    return tc, t_med

def make_composite_wf(k0s, catalog):
    catalog_k0=np.array([key for key in catalog])
    WF_list=[]
    for k0 in k0s:
        ii=np.argmin(np.abs(catalog_k0-k0))
        WF_list.append(catalog[(catalog_k0[ii])])
    WFs=waveform(catalog[0].t, np.concatenate([WFi.p for WFi in WF_list], axis=1))
    WF_mean=WFs.calc_mean(normalize=False, threshold=5000)
    return WF_mean, WFs

def composite_stats(WF, t_window):
    # iterate to find the median:
    ctr_last=-999
    els=np.ones_like(WF.p, dtype=bool)
    ctr=WF.centroid()
    count=0
    while np.abs(ctr-ctr_last) > 0.001 and count < 100:
        sigma_r=WF.robust_spread(els=els)
        els=np.abs(WF.t-ctr) <= np.max([3*sigma_r, t_window/2])
        ctr_last=ctr
        ctr=WF.centroid(els=els)
        count += 1
    sigma_r=WF.robust_spread(els=els)
    sigma=WF.sigma(els=els)
    med=WF.percentile(0.5, els=els)
    return ctr, med, sigma, sigma_r

def my_lsfit(G, d):
    try:
        # this version of the inversion is much faster than linalg.lstsq
        m=np.linalg.solve(G.T.dot(G), G.T.dot(d))#, rcond=None)
        #m=m0[0]
    except ValueError:
        print("ValueError in LSq")
        return np.NaN+np.zeros(G.shape[1]), np.NaN, np.NaN
    except np.linalg.LinAlgError:
        print("LinalgError in LSq")
        return np.NaN+np.zeros(G.shape[1]), np.NaN, np.NaN
    r=d-G.dot(m)
    R=np.sqrt(np.sum(r**2)/(d.size-G.shape[1]))
    sigma_hat=RDE(r)
    return m, R, sigma_hat

def plane_fit_R(D, ind):
    G=np.c_[np.ones(ind.size), D.x[ind].ravel()-D.x[ind].mean(), D.y[ind].ravel()-D.y[ind].mean()]
    m, R, sigma_hat = my_lsfit(G, D.elevation[ind])
    return R, np.sqrt(np.sum(m[1:2]**2))

def setup(scat_file, impulse_file):

    with h5py.File(impulse_file,'r') as h5f:
        TX=waveform(np.array(h5f['/TX/t']),np.array(h5f['/TX/p']))

    TX.t *= 1e9
    TX.t -= TX.nSigmaMean()[0]
    TX.tc = 0
    TR=np.round((np.max(TX.t)-np.min(TX.t))/0.25)*0.25;
    t_i=np.arange(-TR/2, TR/2+0.25, 0.25)
    TX.p=np.interp(t_i, TX.t.ravel(), TX.p.ravel())
    TX.t=t_i
    TX.p.shape=[TX.p.size,1]
    TX.normalize()

    # make the library of templates

    catalog = dict()
    catalog.update({0.:TX})
    catalog.update(make_rx_scat_catalog(TX, h5_file=scat_file))
    return catalog

def make_composite_stats(k0_file, scat_file, impulse_file,  res=40, \
                         t_window=20, dt_fine=0.05):
    k0_field_dict={'location':['latitude','longitude', 'elevation'],
            'both':['K0']}
    catalog=setup(scat_file, impulse_file)
    D_in=pc.data().from_h5(k0_file, field_dict=k0_field_dict)
    D_in.get_xy(EPSG=3031)
    # remove bogus points (lat=lon=0)
    D_in.index(np.abs(D_in.x)>0)

    pt_dict=pc.bin_rows(np.c_[np.round(D_in.x/res)*res, np.round(D_in.y/res)*res])
    D_out=pc.data().from_dict(
        {field:np.zeros(len(pt_dict))+np.NaN for field in \
         ['t_ctr', 't_med','t_sigma', 't_sigma_r','N','x','y', 'z_sigma', 'z_slope_mag']})
    
    TX=resample_wf(catalog[0], dt_fine)
    TX_mean, TX_med = composite_stats(TX, t_window)[0:2]

    for ii, ctr in enumerate(pt_dict):
        N=len(pt_dict[ctr])
        if N < 10:
            continue
        WF0 = make_composite_wf(D_in.K0[pt_dict[ctr]], catalog)[0]
        WF0 = resample_wf(WF0, 0.05)
        D_out.t_ctr[ii], D_out.t_med[ii], D_out.t_sigma[ii], D_out.t_sigma_r[ii]=composite_stats(WF0, t_window)
        D_out.z_sigma[ii], D_out.z_slope_mag[ii] = plane_fit_R(D_in, pt_dict[ctr])
        D_out.x[ii] = ctr[0]
        D_out.y[ii] = ctr[1]
        D_out.N[ii] = len(pt_dict[ctr])
        
    return D_out, TX_mean, TX_med

def resample_wf(WF, dt):
    ti=np.arange(WF.t[0], WF.t[-1]+dt, dt)
    sz=[ti.size, 1]
    fi=interp1d(WF.t.ravel(), WF.p.ravel(), kind='cubic', fill_value=0, bounds_error=False)
    return waveform(ti.reshape(sz), fi(ti).reshape(sz))

def make_plot(k0_file, scat_file, impulse_file,  xy0=None, res=40, t_window=20):

    #res_test(impulse_file)

    k0_field_dict={'location':['latitude','longitude'],
            'both':['K0']}
    catalog=setup(scat_file, impulse_file)
    D_in=pc.data().from_h5(k0_file, field_dict=k0_field_dict)
    D_in.get_xy(EPSG=3031)
    pt_dict=pc.bin_rows(np.c_[np.round(D_in.x/res)*res, np.round(D_in.y/res)*res])
    xy_fits=np.concatenate([np.array(key).reshape([1,2]) for key in pt_dict], axis=0)
    best=np.argmin(np.abs((xy_fits[:,0]-xy0[0])+1j*(xy_fits[:,1]-xy0[1])))
    WF_composite, WFs=make_composite_wf(D_in.K0[pt_dict[tuple(xy_fits[best,:])]], catalog)
    TX=resample_wf(catalog[0], 0.05)
    WF_composite=resample_wf(WF_composite, 0.05)
    
    bar0, med0, sigma0, sigmar_0=composite_stats(TX, t_window)
    barc, medc, sigmac, sigmar_c=composite_stats(WF_composite, t_window)

    TX_corr_mean_c, TX_corr_med_c=TX_corr(TX, np.max([6*sigmar_c, t_window]), sigmar_c, TX_sigma=sigma0)
    TX_corr_mean_0, TX_corr_med_0=TX_corr(TX, np.max([6*sigmar_0, t_window]), sigmar_0, TX_sigma=sigma0)

    plt.figure()
    z0 = TX.centroid()*-0.15

    for WF in WFs:
        #WFi = resample_wf(WF, 0.01)
        plt.plot(WF.p, WF.t*-.15-z0, linewidth=0.5, color='gray')
    plt.plot(TX.p, TX.t*-.15-z0, 'k', linewidth=2)
    plt.plot(WF_composite.p.ravel(), WF_composite.t.ravel()*-.15-z0, 'b', linewidth=2)
    plt.gca().axhline(-bar0*.15-z0, color='k', linewidth=2, label='TX mean')
    plt.gca().axhline(-barc*.15-z0, color='b', linewidth=2, label='RX mean')
    plt.gca().axhline(-med0*.15-z0, linestyle='--', color='k', linewidth=2, label='TX med')
    plt.gca().axhline(-medc*.15-z0, linestyle='--', color='b', linewidth=2, label='RX med')
    plt.legend()
    plt.xlabel('power')
    plt.ylabel('elevation WRT surface, m')

def res_test(impulse_file):

    with h5py.File(impulse_file,'r') as h5f:
        TX=waveform(np.array(h5f['/TX/t'])*1.e9,np.array(h5f['/TX/p']))
    dt_vals=[0.25, 0.1, 0.05, 0.025, 0.0125, 0.0125/2, 0.0125/4]
    ctr_vals=np.zeros_like(dt_vals)+np.NaN
    med_vals=np.zeros_like(dt_vals)+np.NaN
    wctr_vals=np.zeros_like(dt_vals)+np.NaN
    wmed_vals=np.zeros_like(dt_vals)+np.NaN
    sigma_vals=np.zeros_like(dt_vals)+np.NaN
    for ii, dt in enumerate(dt_vals):
        ti=np.arange(TX.t[0], TX.t[-1]+dt, dt)
        ti=ti.reshape([ti.size, 1])
        #TXi=waveform(ti, np.interp(ti, TX.t.ravel(), TX.p.ravel()).reshape(ti.shape))
        TXi=resample_wf(TX, dt)
        ctr_vals[ii], med_vals[ii] = wf_med_bar(TXi, 20., t_tol=0.001)       
        wctr_vals[ii], wmed_vals[ii], _, sigma_vals[ii] = composite_stats(TXi, 20)
        plt.title(dt)

    plt.figure(11); plt.clf()
    plt.semilogx(dt_vals, ctr_vals, marker='o', label='ctr from int')
    plt.semilogx(dt_vals, med_vals, marker='x', label='med from int')
    plt.semilogx(dt_vals, wctr_vals, marker='o', label='ctr from wf')
    plt.semilogx(dt_vals, wmed_vals, marker='x', label='med from wf')
    plt.legend()


def main():
    parser=ArgumentParser("add simulated waveforms based on k0 values in files, report IECsat-2 time delays")
    parser.add_argument('fit_file', type=str, help='output fit file from fit_ATM_scat_2color')
    parser.add_argument('--scat_file', '-s', type=str, help='scattering file giving expected surface response functions as a function of k0')
    parser.add_argument('--impulse_file', '-i', type=str, help='file giving the transmit-pulse shape.  Must have fields t and p in group TX')
    parser.add_argument('--out_file', '-o', type=str, help='output file, h5 format')
    parser.add_argument('--width', '-w', type=float, default=40, help='window over which to collect waveforms in x and y.')
    parser.add_argument('--t_window','-t', type=float, default=20, help='time window for elevation calculations, default of 20 ns matches ATL06')
    parser.add_argument('--xy', type=float, nargs=2, help='make plot for location x, y')
    args=parser.parse_args()

    if args.xy is not None:
        make_plot(args.fit_file, args.scat_file, args.impulse_file, xy0=args.xy,  res=args.width, t_window=args.t_window)
        return
    if args.out_file is not None:
        out_file=args.out_file
    else:
        out_dir=os.path.join(os.path.dirname(args.fit_file), 'composite_wf')
        if not os.path.isdir(out_dir):
            print(f"making directory:{out_dir}")
            os.mkdir(out_dir)
        out_file=os.path.join(out_dir, os.path.basename(args.fit_file).replace('out.h5','out_composite_WF.h5'))
        print(f"out_file is {out_file}")
    D_out, TX_mean, TX_median = make_composite_stats(args.fit_file, args.scat_file, args.impulse_file, res=args.width, t_window=args.t_window)

    D_out.to_h5(out_file)
    with h5py.File(out_file,'r+') as h5f:
        h5f.attrs['tx_mean']=TX_mean
        h5f.attrs['tx_median']=TX_median

if __name__=='__main__':
    main()


# note: these input arguments:
    #ILNSAW1B_20181110_181100.atm6DT7.h5_q4_out.h5 -s SRF_green_full.h5 -i TEP.h5 -w 40 --xy -511599.5 302234.5
