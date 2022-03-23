#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 20:19:13 2022

@author: ben
"""
import numpy as np
from ATM_waveform import broaden_p

def make_broadened_WF(sigma, K0_top, WF_top, catalog):
    this_key=(K0_top, sigma)
    if this_key not in catalog:
        # if we haven't already broadened the WF to sigma, try it now:
        if sigma==0:
            catalog.update(this_key, p=WF_top.p, t0=WF_top.t0, tc=WF_top.tc)
            return(catalog[this_key])
        else:
            try:
                temp=broaden_p(WF_top, sigma)
                catalog.update(this_key, p=temp, p_squared=temp*temp, \
                               t0=WF_top.t0, tc=WF_top.tc,)
                return catalog[this_key]
            except ValueError:
                print("Convolution failed")
    return catalog[this_key]
