#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 20:20:14 2019

@author: ben
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

from IS2_calval.waveform import waveform
from IS2_calval.fit_waveforms import broadened_misfit, listDict, gaussian
from IS2_calval.fit_2color_waveforms import fit_broadened
from IS2_calval.make_rx_scat_catalog import make_rx_scat_catalog
SRF_green='/Users/ben/git_repos/IS2_calval/SRF_green_full.h5'
SRF_IR='/Users/ben/git_repos/IS2_calval/SRF_IR_full.h5'
with h5py.File('/Data/ATM_WF/GroundTest/Spring18/NarrowSwathTx.h5','r') as h5f:
    TX_spring_green=waveform(np.array(h5f['/TX/t']), np.array(h5f['TX/p']))
with h5py.File('/Data/ATM_WF/GroundTest/IR/TX_IR_20181002.h5','r') as h5f:
    TX_spring_IR=waveform(np.array(h5f['/TX/t']), np.array(h5f['TX/p']))

impulse_catalogs={'G':make_rx_scat_catalog(TX_spring_green, h5_file=SRF_green),
    'IR':make_rx_scat_catalog(TX_spring_IR, h5_file=SRF_IR)}

channels=list(impulse_catalogs.keys())

r_vals=[X for X in impulse_catalogs[channels[0]].keys()]
twomm=np.argmin(np.abs(np.array(r_vals)-0.002))
tK=np.arange(-20, 20, 0.25);
sigma_true=2
K=gaussian(tK, 0, sigma_true); K /= K.sum()

impulses={'G':{0:TX_spring_green}, 'IR':{0:TX_spring_IR}}
catalogs={'G':listDict(), 'IR':listDict()}
for ch in channels:
    # set up the waveform catalog
    for key in impulse_catalogs[ch]:
        catalogs[ch][[key]]=waveform(impulse_catalogs[ch][key].t.copy(), impulse_catalogs[ch][key].p.copy())
    catalogs[ch].update(impulses[ch])

model_WFs={key:waveform(impulse_catalog[r_vals[-1]].t, np.convolve(impulse_catalog[r_vals[twomm]].p.ravel(), K.ravel(),'same')).normalize() for key, impulse_catalog in impulse_catalogs.items()}

for key, WF in model_WFs.items():
    WF.p += np.random.randn(*WF.p.shape)*WF.p.max()/50.
    WF.subBG(t50_minus=5)


# define a plotting function
def t_misfit_plot(sigma, reff, model_WFs, catalogs, name, color):
    channels=list(model_WFs.keys())
    delta_ts=np.arange(-1.25, 1.5, 0.125)
    Rcurve={(ii):[] for ii in delta_ts}
    for ch in channels:
        M=listDict()
        _=broadened_misfit(delta_ts, sigma, model_WFs[ch], catalogs[ch], M, [reff],  t_tol=0.5, refine_parabolic=False)
        for X in M.keys():
            if 'delta_t' in M[X]:
                if (M[X]['delta_t']) not in Rcurve:
                    (M[X]['delta_t'])=[]
                Rcurve[(M[X]['delta_t'])] += [M[X]['R']]
    R_array=np.c_[[np.array(Rcurve[(ii)]) for ii in delta_ts]]
    for col, ch in enumerate(channels):
        R_array[:,col] /= model_WFs[ch].noise_RMS

    plt.plot(delta_ts, np.sum(R_array, axis=1),marker='o', label=name, color=color)


def sigma_misfit_plot(sigmas, reff, model_WFs, catalogs, name, color):
    channels=list(model_WFs.keys())
    #channels=['G']
    delta_ts=np.arange(-1.5, 2, 0.5)
    R_best=np.zeros([len(sigmas),2])
    R_best_P=np.zeros([len(sigmas), 2])
    for ii, sigma in enumerate(sigmas):
        for col, ch in enumerate(channels):
            M=listDict()
            R_best[ii, col] = broadened_misfit(delta_ts, sigma, model_WFs[ch], catalogs[ch], M, [reff],  t_tol=0.25, refine_parabolic=False)/model_WFs[ch].noise_RMS
            M=listDict()
            R_best_P[ii, col] = broadened_misfit(delta_ts, sigma, model_WFs[ch], catalogs[ch], M, [reff],  t_tol=0.25, refine_parabolic=True)/model_WFs[ch].noise_RMS

    plt.plot(sigmas, np.sum(R_best, axis=1),marker='o', label=name, color=color)
    plt.plot(sigmas, np.sum(R_best_P, axis=1),marker='x', label=name, color=color)


def r_misfit_plot(r_eff_vals, model_WFs, catalogs, name, color, t_tol=.125):

    #channels=['G']
    sigmas=None;#np.array([0, 0.5, 1, 2, 4])
    delta_ts=np.arange(-1.7, 2, 1)
    R_best=np.zeros([len(r_eff_vals),])
    for ii, rr in enumerate(r_eff_vals):
        Ms={key:listDict() for key in model_WFs}
        R_best[ii]=fit_broadened(delta_ts, sigmas,  model_WFs, catalogs,  Ms, [rr], sigma_tol=0.125, sigma_max=5., t_tol=t_tol)

    plt.semilogx(r_eff_vals, R_best,marker='o', label=name, color=color)

if True:
    plt.figure(); plt.plot(model_WFs['G'].t, model_WFs['G'].p,'g.')
    plt.plot(model_WFs['IR'].t, model_WFs['IR'].p,'r.')
    plt.xlabel('time, ns')
    plt.ylabel('amplitude')

if False:
    plt.figure()
    t_misfit_plot(np.maximum(sigma_true-1, 0), r_vals[twomm], model_WFs, catalogs, 'right r, $\sigma$ too small','r')
    t_misfit_plot(sigma_true+1,  r_vals[twomm], model_WFs, catalogs, 'right r, $\sigma$ too large','b')
    t_misfit_plot(sigma_true, r_vals[twomm-10], model_WFs, catalogs, 'r too small, right $\sigma$','m')
    t_misfit_plot(sigma_true, r_vals[twomm+10], model_WFs, catalogs, 'r too large, right $\sigma$','g')
    t_misfit_plot(sigma_true, r_vals[twomm], model_WFs, catalogs, 'bang on!','k')
    plt.legend()
    plt.xlabel('time offset, ns')
    plt.ylabel('misfit')

if False:
    sigmas=np.arange(0, .5, 0.1)
    plt.figure()
    sigma_misfit_plot(sigmas, r_vals[twomm-10], model_WFs, catalogs, 'r too small','m')
    sigma_misfit_plot(sigmas, r_vals[twomm+10], model_WFs, catalogs, 'r too large','g')
    sigma_misfit_plot(sigmas, r_vals[twomm], model_WFs, catalogs, 'bang on!','k')



r_eff_vals=np.array(r_vals)[np.maximum(0,twomm-10):np.minimum(len(r_vals)-1, twomm+10)]
plt.figure()
r_misfit_plot(r_eff_vals, model_WFs, catalogs, 'the whole bananna', 'k', t_tol=.125/2)
r_misfit_plot(r_eff_vals, model_WFs, catalogs, 'the whole bananna', 'b', t_tol=.125)
r_misfit_plot(r_eff_vals, model_WFs, catalogs, 'the whole bananna', 'm', t_tol=.125*2)
r_misfit_plot(r_eff_vals, model_WFs, catalogs, 'the whole bananna', 'r', t_tol=.125*4)



