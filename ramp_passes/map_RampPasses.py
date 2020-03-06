#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:28:13 2020

@author: ben
"""

import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import pointCollection as pc
import numpy as np

D=pc.data().from_h5('ramp_pass_ll.h5')
latR=np.array([np.nanmin(D.latitude), np.nanmax(D.latitude)])
lonR=np.array([np.nanmin(D.longitude), np.nanmax(D.longitude)])


if True:
    # Create a Stamen terrain background instance.
    bg_image = cimgt.GoogleTiles(style='satellite'); #Stamen('terrain-background')

    fig = plt.figure(figsize=(10, 10))

    # Create a GeoAxes in the tile's projection.
    ax = fig.add_subplot(1, 1, 1, projection=bg_image.crs)

    # Limit the extent of the map to a small longitude/latitude range.
    ax.set_extent([lonR[0], lonR[1], latR[0], latR[1]], crs=ccrs.Geodetic())

    # Add the Stamen data at zoom level 8.
    ax.add_image(bg_image, 10)
    #plt.plot(D.longitude.reshape([1, -1]), D.latitude.reshape([1, -1]), color='k', marker='.', transform=ccrs.Geodetic())
    plt.scatter(D.longitude, D.latitude, 2, c=D.latitude, marker='.', transform=ccrs.Geodetic())

    #for lati, loni in zip(D.latitude, D.longitude):
    #    plt.plot(loni, lati, 'k.', transform=ccrs.Geodetic())
