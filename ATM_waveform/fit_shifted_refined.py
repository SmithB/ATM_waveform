#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 07:48:36 2022

@author: ben
"""
import numpy as np
from ATM_waveform.calc_R_and_tshift import calc_R_and_tshift
from ATM_waveform.golden_section_search import golden_section_search

def fit_shifted_refined(delta_t_list, sigma, catalog, WF, M, key_top,  t_tol=None, refine_parabolic=True):
    """
    Find the shift value that minimizes the misfit between a template and a waveform

    performs a golden-section search along the time dimension
    """
    #G=np.ones((WF.p.size, 2))
    if t_tol is None:
        t_tol=WF['t_samp']/10.
    delta_t_spacing=delta_t_list[1]-delta_t_list[0]
    bnds=[np.min(delta_t_list)-5, np.max(delta_t_list)+5]

    if key_top in M and 'best' in M[key_top]:
        this_delta_t_list = delta_t_list[np.argsort(np.abs(delta_t_list-M[key_top]['best']['delta_t']))[0:2]]
        this_delta_t_list = np.concatenate([this_delta_t_list, [this_delta_t_list.mean()]])
    else:
        this_delta_t_list=delta_t_list
    #K_last=list(M.keys())
    delta_t_history={}
    this_key = key_top + [sigma]
    delta_t_best, R_best = golden_section_search(calc_R_and_tshift,
                                                 this_delta_t_list,
                                                 delta_t_spacing, bnds=bnds,
                                                 fn_args=[WF, catalog[this_key]],
                                                 fn_kwargs={'fit_history':delta_t_history,'return_R_only':True},
                                                 tol=WF.dt, integer_steps=True,
                                                 step_size=WF.dt)
    delta_t_refined = delta_t_history[delta_t_best]['t_shift_refined']
    A=delta_t_history[delta_t_best]['A']
    #M[this_key]={'key':this_key, 'R':R_best, 'sigma':sigma, 'delta_t':delta_t_best}
    M[this_key] = {'key':this_key, 'R':R_best, 'delta_t':delta_t_refined,'A':A,'sigma':sigma}
    if 'best' not in M[key_top]:
        M[key_top]['best'] = {'key':this_key, 'R':R_best, 'delta_t':delta_t_refined,'A':A}
    elif R_best < M[key_top]['best']['R']:
        M[key_top]['best'] = {'key':this_key, 'R':R_best, 'delta_t':delta_t_refined,'A':A}
    return R_best