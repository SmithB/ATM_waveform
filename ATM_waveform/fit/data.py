#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 10:18:03 2022

pointCollection class for reading ATM Waveform fit data


@author: ben
"""
import numpy as np
import pointCollection as pc


class data(pc.data):
    np.seterr(invalid='ignore')

    def __default_field_dict__(self):
        return {
                'G':['A', 'Amax', 'B', 'R','delta_t', 'nPeaks', 'noise_RMS',\

                     'shot','t0','tc', 'seconds_of_day'],
                'both':['K0','R','sigma'],
                'location':['elevation', 'latitude', 'longitude']}
    def from_h5(self, filename, **kwargs):
        if 'field_dict' not in kwargs or kwargs['field_dict'] is None:
            kwargs['field_dict']=self.__default_field_dict__()
        # call pc.data() to read the file,
        D0=pc.data().from_h5(filename, **kwargs)
        if D0 is None:
            return D0
        if 'latitude' in D0.fields:
            D0.index( ~((D0.latitude==0) & (D0.longitude==0)))
        return D0
