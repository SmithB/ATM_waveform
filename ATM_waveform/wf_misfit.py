#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:21:53 2022

@author: ben
"""

import numpy as np
from ATM_waveform import make_broadened_WF
from ATM_waveform import corr_no_mean


def amp_misfit(x, y, els=None, A=None, x_squared=None):
    """
    misfit for the best-fitting scaled template model.
    """
    if els is None:
        ii=np.isfinite(x) & np.isfinite(y)
    else:
        ii=els
    if A is None:
        xi=x[ii]
        yi=y[ii]
        if x_squared is not None:
            A = np.dot(xi, yi) / np.sum(x_squared[ii])
        else:
            A=np.dot(xi, yi)/np.dot(xi, xi)
    r=yi-xi*A
    #r=y[ii]-x[ii]*A
    R=np.sqrt(np.sum((r**2)/(ii.sum()-2)))
    return R, A, ii

def lin_fit_misfit(x, y, G=None, m=None, Ginv=None, good_old=None):
    """
    Misfit for the the best-fitting background + scaled template model
    """
    if G is None:
        G=np.ones((x.size, 2))
    G[:,0]=x.ravel()
    good=np.isfinite(G[:,0]) & np.isfinite(y.ravel())
    if good_old is not None and Ginv is not None and np.all(good_old==good):
        # use the previously calculated version of Ginv
        m=Ginv.dot(y[good])
        R=R=np.sqrt(np.sum((y[good]-G[good,:].dot(m))**2.)/(good.sum()-2))
    else:
        # need at least three good values to calculate a misfit
        if good.sum() < 3:
            m=np.zeros(2)
            R=np.sqrt(np.sum(y**2)/(y.size-2))
            return R, m
        G1=G[good,:]
        try:
            #m=np.linalg.solve(G1.transpose().dot(G1), G1.transpose().dot(y[good]))
            Ginv=np.linalg.solve(G1.transpose().dot(G1), G1.transpose())
            m=Ginv.dot(y[good])
            R=np.sqrt(np.sum((y[good]-G1.dot(m))**2.)/(good.sum()-2))
            #R=np.sqrt(m_all[1][0])
        except np.linalg.LinAlgError:
            m=np.zeros(2)
            R=np.sqrt(np.sum(y**2.)/(y.size-2))
    return R, m, Ginv, good


def wf_misfit(delta_t, sigma, WF, catalog, M, K0_top, WF_top,  G=None, update_catalog=True,
              return_data_est=False, fit_BG=False, parent_WF=None):
    """
        Find the misfit between a scaled and shifted template and a waveform
    """
    if G is None and fit_BG:
        G=np.ones((WF.nSamps, 2))
    this_key = (K0_top, sigma, delta_t)
    if (this_key in M) and (return_data_est is False):
        return M[this_key]['R']
    else:
        # check if the broadened but unshifted version of this key is in the catalog
        if parent_WF is None:
            broadened_key = (K0_top, sigma)
            if broadened_key in catalog:
                parent_WF = catalog[broadened_key]
            else:
                parent_WF = make_broadened_WF(sigma, K0_top, WF_top, catalog)
        broadened_p = parent_WF.p.copy()
        # check if the shifted version of the broadened waveform is in the catalog
        if this_key in catalog:
            this_entry=catalog[this_key]
            this_p = this_entry.p
            this_p_squared = this_entry.p_squared
            mask = this_entry.mask
        else:
            # if not, make it.
            M[this_key]={}
            this_p = np.interp(WF.t.ravel(), (parent_WF.t-parent_WF.tc+delta_t).ravel(), \
                               broadened_p.ravel(), left=np.NaN, right=np.NaN)
            this_p_squared = this_p * this_p
            # Note that argmax on a binary array returns the first nonzero index (faster than where)
            ii=np.argmax(this_p > 0.01*np.nanmax(this_p))
            mask=np.ones_like(this_p, dtype=bool)
            mask[0:ii-4] = False
            if update_catalog:
                catalog.update(this_key,  p=this_p, p_squared=this_p_squared,\
                        tc=parent_WF.tc, t0=parent_WF.t0, mask=mask)
        if fit_BG:
            this_entry=catalog[this_key]
            # solve for the background and the amplitude
            R, m, Ginv, good = lin_fit_misfit(catalog[this_key].p, WF.p, G=G,\
                Ginv=this_entry.params['Ginv'], good_old=this_entry.mask)
            M[this_key] = {'K0':K0_top, 'R':R, 'A':np.float64(m[0]), 'B':np.float64(m[1]), 'delta_t':delta_t, 'sigma':sigma}
        else:
            # solve for the amplitude only
            G=this_p
            good=np.isfinite(G).ravel() & np.isfinite(WF.p).ravel() & mask
            m, R = corr_no_mean(G.ravel(), WF.p.ravel(), this_p_squared.ravel(), good.astype(np.int32).ravel(), G.size)
            M[this_key] = {'K0':K0_top, 'R':R, 'A':m, 'B':0., 'delta_t':delta_t, 'sigma':sigma}
        if return_data_est:
            return R, G.dot(m)
        else:
            return R
