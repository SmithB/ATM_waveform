#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 09:29:24 2020

@author: ben
"""

import h5py
import glob
import numpy as np
import matplotlib.pyplot as plt
import pointCollection as pc
import os
thedir='/Users/ben/temp/AA_wf_fits/'

field_dict={None:['x','y','t_ctr','t_med','t_sigma','z_sigma']}
D=[]
for file in glob.glob(os.path.join(thedir, '2*/composite_WF/*.h5')):
    D += [pc.data().from_h5(file, field_dict=field_dict)]

with h5py.File(file,'r') as ff:
    t_tx_med=ff.attrs['tx_median']
    t_tx_mean=ff.attrs['tx_mean']

D=pc.data().from_list(D)

D.index(np.isfinite(D.x+D.y+D.t_ctr))
MOA_1km=pc.grid.data().from_geotif('/Data/MOA/2009/moa1000_2009_hp1_v1.1.tif', \
            bounds=[(np.min(D.x)-1.e5, np.max(D.x)+1.e5), (np.min(D.y)-1.e5, np.max(D.y)+1.e5)])


bds_PIG=(array([-1757250., -1407479.]), array([-504892., -106748.]))
bds_REC=(array([-707937., -350724.]), array([ 75579., 395582.]))

pt_ctr_fine=[-1597798.0, -23800.0]
pt_ctr_coarse=[-511599.5, 302234.5]