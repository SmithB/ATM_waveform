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
import argparse
from ATM_waveform.three_sigma_edit_fit import three_sigma_edit_fit
import pointCollection as pc

K0_stats = lambda D,E : [np.nanmedian(D.K0[E]), np.nanstd(D.K0[E])]

def plane_fit(D, E):
    G0=np.c_[D.x[E].ravel()-np.nanmean(D.x[E]), D.y[E].ravel()-np.nanmean(D.y[E]), np.ones_like(D.y[E]).ravel()]
    z0=D.elevation[E]   
    m, r, sigma, good=three_sigma_edit_fit(G0, z0, n_iterations=3, sigma_min=0.1)
    return [m[0], m[1], m[2], sigma]


field_dict={'location':['latitude','longitude', 'elevation'], 
            'both':['K0']}

def write_data(in_file, out_file,  Db, thumb_dir):
    for count, key in enumerate(Db.keys()):
        Db[key].to_h5(out_file, group=key, replace=(count==0))


parser = argparse.ArgumentParser(description='Fit the waveforms from an ATM file with a set of scattering parameters')
parser.add_argument('input_dir', type=str)
parser.add_argument('--input_file', '-f', type=str)
parser.add_argument('--EPSG','-E', type=str)
parser.add_argument('--queue','-q', action='store_true')

args=parser.parse_args()

thedir=args.input_dir
thumb_dir=thedir+'/thumbs'
if not(os.path.isdir(thumb_dir)):
    os.mkdir(thumb_dir)

EPSG=args.EPSG

if args.input_file is None:
    fnames=glob.glob(thedir+'/I*.h5')
else:
    if os.path.isfile(args.input_file):
        fnames=[args.input_file]
        args.input_dir='.'
    if not os.path.isfile(args.input_file):
        fnames=[os.path.join(args.input_dir, args.input_file)]    

fnames.sort()
for fname in fnames:
    
    out_file=thumb_dir+'/'+os.path.basename(fname)
    if os.path.isfile(out_file):
        continue
    if args.queue:
        thestr=f"calc_medians.py {thedir} -f {fname} --EPSG {args.EPSG}"
        print(thestr)
        continue

    print("working on "+ fname)
    try:
        D=pc.data().from_h5(fname, field_dict=field_dict).get_xy(EPSG=int(EPSG)) 
        D.index(D.latitude!=0)
        Db={}
        Db['100m']=pc.apply_bin_fn(D, 100, fn=K0_stats, fields=['K0_med', 'K0_std'])
        Db['10m']=pc.apply_bin_fn(D, 10, fn=K0_stats, fields=['K0_med', 'K0_std'])   
        temp = pc.apply_bin_fn(D, 10, fn=plane_fit, fields=['slope_x', 'slope_y', 'h0', 'R'])
        Db['10m'].assign({field:getattr(temp, field) for field in ['slope_x', 'slope_y', 'h0', 'R']})
        write_data(fname, out_file, Db, thumb_dir)
    except Exception as e:
        print(f"problem with {fname}:")
        print(e)
