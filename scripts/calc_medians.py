#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 20:32:40 2020

@author: ben
"""

import pointCollection as pc
import glob
import numpy as np
import sys

import os
from ATM_waveform.three_sigma_edit_fit import three_sigma_edit_fit
import pointCollection as pc

K0_med = lambda D,E : np.nanmedian(D.K0[E])

K0_threshold_sigma = lambda D,E : np.nanstd(D.K0[E]>0.0003)

def plane_misfit(D, E):
    G0=np.c_[D.x[E].ravel(), D.y[E].ravel(), np.ones_like(D.y[E]).ravel()]
    z0=D.elevation[E]
    m, r, sigma, good=three_sigma_edit_fit(G0, z0, n_iterations=3)
    return sigma

field_dict={'location':['latitude','longitude'], 
            'both':['K0']}

def write_data(in_file, Db):
    in_dir=os.path.dirname(in_file)
    thumb_dir=in_dir+'/thumbs'
    out_file=thumb_dir+'/'+os.path.basename(in_file)
    for count, key in enumerate(Db.keys()):
        Db[key].to_h5(out_file, group=key, replace=(count==0))

D_1km_all={}

dir_list=glob.glob('/home/besmith4/nobackup/ATM_WF/AA_18/fits/2018*')

for thedir in dir_list:
    thumb_dir=thedir+'/thumbs'
    if not os.path.isdir(thumb_dir):
        os.mkdir(thumb_dir)

    #if not os.path.isdir(thedir+'/thumbs_1km')
    files=glob.glob(thedir+'/*.h5')
    if len(files)==0:
        continue
    D_list=[]
    for file in files:
        D=pc.data().from_h5(file, field_dict=field_dict).get_xy(EPSG=3031)
        Db={}
        Db['10m']=apply_bin_fn(D, 10, fn=K0_med, field='K0')
        Db['100m']=apply_bin_fn(D, 100, fn=K0_med, field='K0')
        Db['z_misfit_10m']=apply_bin_fn(D, 10, fn=plane_misfit, field='elevation')
        D_list.append(D)
        write_data(file, Db)
    D_all=pc.data().from_list(D_list)
    D_1km_all[thedir]=apply_bin_fn(D_all, 1000, fn=K0_med, field='K0')    
    D_1km_all[thedir].to_h5(thedir+'/thumbs/all_1km_med_k0')
Dm=pc.data().from_list([D_1km_all[key] for key in D_1km_all])
Dm.K0[Dm.K0<1e-6] = 1e-6
