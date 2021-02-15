#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:42:09 2020

@author: ben
"""

import pointCollection as pc
import glob
import numpy as np
import matplotlib.pyplot as plt
import fastkml
import shapely.geometry as geo

thedir='/Volumes/ice2/ben/ATM_WF/RampPasses/fits/2017.12.04/'
#thedir='/Data/ATM_WF/RampPasses/fits/2017.12.04/'

field_dict={'both':['K0'], 'location':['latitude','longitude','elevation'],'G':['nPeaks', 'R','shot','noise_RMS','A']}


D_list=[];
files=glob.glob(thedir+'/*.h5')
last_shot=0
for count, file in enumerate(files[0:]):
    try:
        D_list += [pc.data().from_h5(file, field_dict=field_dict)]
        D_list[-1].assign({'file_num':np.ones(D_list[-1].size)+count,
                           'shot_all':D_list[-1].shot+last_shot})
        last_shot=np.max(D_list[-1].shot_all)
    except Exception:
        print("Error in %s" % file)
        pass
D=pc.data().from_list(D_list)
Re=6378e6
d2r=np.pi/180.

lat0=np.nanmedian(D.latitude)
lon0=np.nanmedian(D.longitude)
D.assign({'y':(D.latitude-lat0)*d2r*Re, 'x': np.cos(lat0*d2r)*d2r*Re*(D.longitude-lon0)})

D.index(D.nPeaks==1)

good_dZ=np.zeros_like(D.elevation)
good_dZ[0:-1]=np.abs(np.diff(D.elevation))<0.25
good_dZ=np.convolve(good_dZ.astype(float), np.ones(3), mode='same')==3

xx=np.unique(np.round(D.longitude/.001)+1j*np.round(D.latitude/.001))
D1=pc.data().from_dict({'latitude': np.imag(xx)/1000, 'longitude':np.real(xx)/1000-360})

D_filt=D[good_dZ]
D_nonscat=D_filt[D_filt.K0==0]
D_scat=D_filt[D_filt.K0>0]

geoms={}
for file in ['Airstrip_outer.kml', 'Grassy_1.kml', 'Grassy_2.kml' ]:
    with open(file,'rb') as ff:
        doc=ff.read()

    K=fastkml.KML()
    K.from_string(doc)
    geoms[file.replace('.kml','')] =K._features[0]._features[0].geometry

inside = [geo.Point(item[0], item[1]).within(geoms['Airstrip_outer']) for item in zip(D_filt.longitude-360, D_filt.latitude)]

D_inside = D_filt[np.array(inside)]
grassy_1 = [geo.Point(item[0], item[1]).within(geoms['Grassy_1']) for item in zip(D_inside.longitude-360, D_inside.latitude)]
grassy_2 = [geo.Point(item[0], item[1]).within(geoms['Grassy_2']) for item in zip(D_inside.longitude-360, D_inside.latitude)]
D_airstrip = D_inside[(np.array(grassy_1)==0) & (np.array(grassy_2)==0)]

plt.figure(1);
plt.clf()
plt.plot(D_nonscat.x, D_nonscat.y,'k.', markersize=2, zorder=0)
ii=np.argsort(D_scat.K0)
plt.scatter(D_scat.x[ii], D_scat.y[ii], 2, c=np.log10(D_scat.K0[ii]), vmin=-4.5, vmax=-3, zorder=1);
hb=plt.colorbar()
hb.set_label('log $r_{eff}$, m')

plt.figure(2)
plt.clf()
ii=np.flatnonzero((D.K0==0)& good_dZ)
plt.plot(D.x[ii], D.y[ii],'k.', zorder=0)
ii=np.flatnonzero((D.K0>0) & good_dZ)
plt.scatter(D.x[ii], D.y[ii], c=np.log10(D.K0[ii]), vmin=-4.5, vmax=-3, zorder=1); hb=plt.colorbar()
hb.set_label('log $r_{eff}$')

plt.figure(3); plt.clf()
plt.plot(D.shot_all, np.log10(D.K0),'.')
plt.plot(D.shot_all[good_dZ], np.log10(D.K0[good_dZ]),'.')

plt.figure(4); plt.clf()
ii=np.flatnonzero(good_dZ)
plt.scatter(D.longitude[ii]-360, D.latitude[ii], c=D.elevation[ii])
plt.xlabel('longitude'); plt.ylabel('latitude')
hb=plt.colorbar()
hb.set_label('elevation, m')
#plt.figure(5); plt.clf()
#plt.plot(D.shot_all[good_dZ], D.elevation[good_dZ],'.')
