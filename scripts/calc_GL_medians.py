#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 20:32:40 2020

@author: ben
"""

import pointCollection as pc
import glob
import numpy as np
from pointCollection.points_to_grid import apply_bin_fn
import os
from ATM_waveform.three_sigma_edit_fit import three_sigma_edit_fit
import matplotlib.pyplot as plt

mosaic=pc.grid.data().from_geotif('/Volumes/ice1/ben/MOG/2005/mog_2005_1km.tif')

K0_med = lambda D,E : np.nanmedian(D.K0[E])
K0_threshold_sigma = lambda D,E : np.nanstd(D.K0[E]>0.0003)
A_bar = lambda D, E : np.nanmean(D.A[E])
R_rms = lambda D, E : np.nanmean(D.R[E]**2)**0.5

def plane_misfit(D, E):
    G0=np.c_[D.x[E].ravel(), D.y[E].ravel(), np.ones_like(D.y[E]).ravel()]
    z0=D.elevation[E]
    m, r, sigma, good=three_sigma_edit_fit(G0, z0, n_iterations=3)
    return sigma

field_dict={'location':['latitude','longitude'], 'both':['K0'], 'G':['A','R']}

def write_data(in_file, data=None):
    in_dir=os.path.dirname(in_file)
    thumb_dir=in_dir+'/thumbs'
    out_file=thumb_dir+'/'+os.path.basename(in_file)
    if data is None:
        return out_file
    for count, key in enumerate(data.keys()):
        data[key].to_h5(out_file, group=key, replace=(count==0))

def read_data(file):
    groups=['10m','200m','D_std_threshold_200m', 'block_std_threshold_200m']
    D={group:pc.data().from_h5(file, group=group) for group in groups}
    return D

D_1km_all={}

dir_list=glob.glob('/Volumes/ice2/ben/ATM_WF/GL_18/fits/2018.*')
problems={}

for thedir in dir_list:
    print("working on "+thedir)
    thumb_dir=thedir+'/thumbs'
    if not os.path.isdir(thumb_dir):
        os.mkdir(thumb_dir)

    #if os.path.isfile(thedir+'/thumbs/all_1km_med_k0'):
    #    continue
    #if not os.path.isdir(thedir+'/thumbs_1km')
    files=glob.glob(thedir+'/*.h5')
    if len(files)==0:
        continue
    D_list=[]
    for file in files:
        out_file=write_data(file)
        if os.path.isfile(out_file):
            continue
        try:
            D=pc.data().from_h5(file, field_dict=field_dict).get_xy(EPSG=3413)
            Db={}
            Db['10m']=apply_bin_fn(D, 10, fn=K0_med, field='K0')
            Db['200m']=apply_bin_fn(D, 200, fn=K0_med, field='K0')
            Db['Abar_200m']=apply_bin_fn(D, 200, fn=A_bar, field='Abar')
            Db['Rrms_200m']=apply_bin_fn(D, 200, fn=R_rms, field='Rrms')

            Db['D_std_threshold_200m']=apply_bin_fn(D, 200, K0_threshold_sigma, field='K0_sigma_30um')
            Db['block_std_threshold_200m']=apply_bin_fn(Db['10m'], 200, K0_threshold_sigma, field='K0_sigma_30um')
            D_list.append(D)
            write_data(file, data=Db)
        except (AttributeError, OSError)  as e:
            problems[file]=e
    if len(D_list) > 0:
        D_all=pc.data().from_list(D_list)
        D_1km_all[thedir]=apply_bin_fn(D_all, 1000, fn=K0_med, field='K0')    
        D_1km_all[thedir].to_h5(thedir+'/thumbs/all_1km_med_k0')
Dm=pc.data().from_list([D_1km_all[key] for key in D_1km_all])
Dm.K0[Dm.K0<1e-6] = 1e-6

mosaic.show(cmap='gray', vmin=14000, vmax=17000)
plt.scatter(Dm.x, Dm.y, 4, c=np.log10(Dm.K0), vmin=-4.5, vmax=-3); plt.colorbar()

