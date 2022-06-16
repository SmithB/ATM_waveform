#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:04:29 2022

@author: ben
"""
import numpy as np
from ATM_waveform.fit_waveforms import wf_misfit, listDict
from ATM_waveform.golden_section_search import golden_section_search

def fit_shifted(delta_t_list, sigma, catalog, WF, M, key_top,  t_tol=None, refine_parabolic=True):
    """
    Find the shift value that minimizes the misfit between a template and a waveform

    performs a golden-section search along the time dimension
    """
    #G=np.ones((WF.p.size, 2))
    if t_tol is None:
        t_tol=WF['t_samp']/10.
    delta_t_spacing=delta_t_list[1]-delta_t_list[0]
    bnds=[np.min(delta_t_list)-5, np.max(delta_t_list)+5]
    fDelta = lambda deltaTval: wf_misfit(deltaTval, sigma, WF, catalog, M,  key_top)
    if key_top in M and 'best' in M[key_top]:
        this_delta_t_list = delta_t_list[np.argsort(np.abs(delta_t_list-M[key_top]['best']['delta_t']))[0:2]]
        this_delta_t_list = np.concatenate([this_delta_t_list, [this_delta_t_list.mean()]])
    else:
        this_delta_t_list=delta_t_list
    #K_last=list(M.keys())
    delta_t_best, R_best = golden_section_search(fDelta, this_delta_t_list, delta_t_spacing, bnds=bnds, tol=t_tol, max_count=100, refine_parabolic=refine_parabolic)
    #K_new=[kk for kk in M.keys() if (kk not in K_last)]
    this_key=key_top+[sigma]+[delta_t_best]
    if this_key not in M:
        # make temporary catalog with only this entry and the top entry (b/c this entry won't get reused)
        temp_catalog=listDict()
        temp_catalog[key_top] = catalog[key_top]
        temp_catalog[key_top+[sigma]] = catalog[key_top+[sigma]]
        R_best = wf_misfit(delta_t_best, sigma, WF, temp_catalog, M,  key_top)
    #M[this_key]={'key':this_key, 'R':R_best, 'sigma':sigma, 'delta_t':delta_t_best}
    M[key_top+[sigma]]['best'] = {'key':this_key, 'R':R_best, 'delta_t':delta_t_best}
    if 'best' not in M[key_top]:
        M[key_top]['best'] = {'key':this_key, 'R':R_best, 'delta_t':delta_t_best}
    elif R_best < M[key_top]['best']['R']:
        M[key_top]['best'] = {'key':this_key, 'R':R_best, 'delta_t':delta_t_best}
    return R_best