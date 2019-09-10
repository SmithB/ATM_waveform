# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:56:31 2018

@author: ben
"""
import numpy as np
#from numpy.linalg import inv
import sys
import matplotlib.pyplot as plt
#import bisect
from ATM_waveform.waveform import waveform
from copy import deepcopy
from ATM_waveform.fit_waveforms import listDict, integer_shift, broadened_misfit, wf_misfit, golden_section_search


DOPLOT=False
KEY_SEARCH_PLOT=False

def fit_broadened(delta_ts, sigmas,  WFs, catalogs,  Ms, key_top, sigma_tol=0.125, sigma_max=5., t_tol=None, refine_sigma=False, sigma_last=None):
    """
    Find the best broadening value that minimizes the misfit between a template and a waveform
    """
    channels=list(WFs.keys())
    for _,M in Ms.items():
        #print(key_top)
        if key_top not in M:
            M[key_top]=listDict()
    fSigma = lambda sigma: np.sum([broadened_misfit(delta_ts, sigma, WFs[ch], catalogs[ch], Ms[ch], key_top, t_tol=t_tol, refine_parabolic=True)/WFs[ch].noise_RMS for ch in channels])
    sigma_step=2*sigma_tol
    FWHM2sigma=2.355

    if sigmas is None:
        # Choose a sigma range to search.  Both channels would have been broadened from
        # the template by some unknown roughness.  Solving for that roughness for each
        # channel gives the maximum that might be needed for either (most conservative range)
        sigma0=0
        for ch in channels:
            sigma_template=catalogs[ch][key_top].fwhm()[0]/FWHM2sigma
            sigma_WF=WFs[ch].fwhm()[0]/FWHM2sigma
            #estimate broadening from template WF to measured WF
            sigma_extra=np.sqrt(np.maximum(0,  sigma_WF**2-sigma_template**2))
            sigma0=np.maximum(sigma0, sigma_step*np.ceil(sigma_extra/sigma_step))
        dSigma=np.maximum(sigma_step, np.ceil(sigma0/4.))
        sigmas=np.unique([0., np.maximum(sigma_step, sigma0-dSigma), sigma0, np.maximum(sigma_step, sigma0+dSigma)])
    else:
        dSigma=np.max(sigmas)/4.
    if np.any(~np.isfinite(sigmas)):
        print("NaN in sigma for shot %d " % WFs[channels[0]].shots)

    if sigma_last is not None:
        i1=np.maximum(1, np.argmin(np.abs(sigmas-sigma_last)))
    else:
        i1=len(sigmas)-1
    sigma_list=[sigmas[0], sigmas[i1]]
    search_hist={}
    sigma_best, R_best = golden_section_search(fSigma, sigma_list, dSigma, bnds=[0, sigma_max], tol=sigma_tol, max_count=20, refine_parabolic=refine_sigma, search_hist=search_hist)

    if refine_sigma:
        # calculate the misfit at this exact sigma value
        R_best=0
        for ch in channels:
            # pass in a temporary catalog so that the top-level catalogs don't
            # get populated with entries that won't get reused
            this_catalog=listDict()
            this_catalog[(key_top)]=catalogs[ch][key_top]
            R_best += broadened_misfit(delta_ts, sigma_best, WFs[ch], this_catalog, Ms[ch], key_top, t_tol=t_tol, refine_parabolic=True)/WFs[ch].noise_RMS
    return R_best

def fit_catalogs(WFs, catalogs_in, sigmas, delta_ts, t_tol=None, sigma_tol=None, return_data_est=False, return_catalogs=False, catalogs=None, params=None, M_list=None):
    """
    Search a library of waveforms for the best match between the broadened, shifted library waveform
    and the target waveforms

    Inputs:
        WFs: a dict of waveform objects, whose entries include:
            't': the waveform's time vector
            'p': the power samples of the waveform
            'tc': a center time relative to which the waveform's time is shifted
        catalog_in: A dictionary containing waveform objects that will be broadened and
                    shifted to match the waveforms in 'WFs'
        sigmas: a list of spread values that will be searched for each template and waveform
                The search over sigmas terminates when a minimum is found
        delta_ts: a list of time-shift values that will be searched for each template and
                waveform.  All of these will be searched, then the results will be refined
                to a tolerance of t_tol
        'params' : a dict giving a list of parameters to return in D_out
        keyword arguments:
            return_data_est:  set to 'true' if the algorithm should return the best-matching
                shifted and broadened template for each input
            t_tol: tolerance for the time search, defaults to WF.t_samp/10
    Outputs:
        WFp: a set of best-fitting waveform parameters that give:
            delta_t: the time-shift required to align the template and measured waveforms
            sigma: the broadening applied to the measured waveform
            k0: the key into the waveform catalog for the best-fitting waveform

    """
    # set a sensible tolerance for delta_t if none is specified
    if t_tol is None:
        t_tol=WFs.dt
    if sigma_tol is None:
        sigma_tol=0.25

    channels=list(WFs.keys())

    N_shots=WFs[channels[0]].size
    N_samps=WFs[channels[0]].t.size

    # make an empty output_dictionary
    if params is None:
        params={'ch':['R','A','B','noise_RMS', 'delta_t', 'shot', 't0','tc','t_shift'], 'both':['K0','R','sigma','Kmin','Kmax']}
    WFp_empty={}
    for ch in channels:
        WFp_empty[ch]={f:np.NaN for f in params['ch']}
    WFp_empty['both']={f:np.NaN for f in params['both']}
    if return_data_est:
        for ch in channels:
            WFp_empty[ch]['wf_est']=np.zeros_like(WFs[ch].t)+np.NaN
            WFp_empty[ch]['t_shift']=np.NaN

    if catalogs is None:
        catalogs={ch:listDict() for ch in channels}

    # make a container for the pulse widths for the catalogs, and copy the input catalogs into the buffer catalogs
    W_catalogs={}
    for ch in channels:
        k_vals=np.sort(list(catalogs_in[ch]))
        W_catalogs[ch]=np.zeros(k_vals.shape)
        # loop over the library of templates
        for ii, kk in enumerate(k_vals):
            # record the width of the waveform
            W_catalogs[ch][ii]=catalogs_in[ch][kk].fwhm()[0]
            # check if we've searched this template before, otherwise copy it into
            # the library of checked templates
            if [kk] not in catalogs[ch]:
                # make a copy of the current template
                temp=catalogs_in[ch][kk]
                catalogs[ch][[kk]]=waveform(temp.t, temp.p, t0=temp.t0, tc=temp.tc)

    fit_param_list=[]
    sigma_last = None
    t_center={ch:WFs[ch].t.mean() for ch in channels}
    last_keys={ch:[] for ch in channels}

    # loop over input waveforms
    for WF_count in range(N_shots):
        fit_params=deepcopy(WFp_empty)
        WF={ch:WFs[ch][WF_count] for ch in channels}
        # skip multi-peak returns:
        #n_peaks=np.array([WF[ch].nPeaks for ch in channels])
        #if np.any(n_peaks > 1):
        #    continue
        # shift the waveforms to put their tcs at the center of the time vector
        # doing this means that we have to store fewer shifted catalogs
        for ch in channels:
            delta_samp=np.round((WF[ch].tc-t_center[ch])/WF[ch].dt)
            WF[ch].p = integer_shift(WF[ch].p, -delta_samp)
            WF[ch].t0 += delta_samp*WF[ch].dt
            WF[ch].tc -= delta_samp*WF[ch].dt
            WF[ch].t_shift = delta_samp*WF[ch].dt

        # set up a matching dictionary (contains keys of waveforms and their misfits)
        Ms={ch:listDict() for ch in channels}
        # this is the bulk of the work, and it's where problems happen.  Wrap it in a try:
        # and write out errors to be examined later
        try:
            if len(k_vals)>1:
                 # find the best misfit between this template and the waveform
                fB=lambda ind:fit_broadened(delta_ts, None, WF, catalogs, Ms, [k_vals[ind]], sigma_tol=sigma_tol, t_tol=t_tol, sigma_last=sigma_last, refine_sigma=True)
                W_broad_ind=0
                # find the first catalog entry that's broader than the waveform (check both cnannels, pick the broader one)
                for ch in channels:
                    this_broad_ind=np.flatnonzero(W_catalogs[ch] >= WF[ch].fwhm()[0])
                    if len(this_broad_ind)==0:
                        this_broad_ind=len(W_catalogs[ch])
                    else:
                        this_broad_ind=this_broad_ind[0]
                    W_broad_ind=np.maximum(W_broad_ind, this_broad_ind)
                # search two steps on either side of the broadness-matched waveform, as well as zero (all broadening due to roughness)
                key_search_ind=np.array(sorted(tuple(set([0, W_broad_ind-2, W_broad_ind+2]))))
                key_search_ind=key_search_ind[(key_search_ind>=0) & (key_search_ind<len(k_vals))]
                search_hist={}
                iBest, Rbest = golden_section_search(fB, key_search_ind, delta_x=2, bnds=[0, len(k_vals)-1], integer_steps=True, tol=1, refine_parabolic=False, search_hist=search_hist)
                iBest=int(iBest)
            else:
                _=fit_broadened(delta_ts, None, WF, catalogs, Ms, [k_vals[0]], sigma_tol=sigma_tol, t_tol=t_tol, sigma_last=sigma_last)
                iBest=0
                Rbest=Ms[ch][[k_vals[0]]]['best']['R']
            this_kval=[k_vals[iBest]]
            fit_params['both']['R']=Rbest
            fit_params['both']['K0']=this_kval
            R_dict={}
            sigma_last=0
            for ch in channels:
                M=Ms[ch]
                M['best']={'key':this_kval, 'R':M[this_kval]['best']['R']}
                for ki in [this_key for this_key in k_vals if [this_key] in M]:
                    if ki in R_dict:
                        R_dict[ki] += M[[ki]]['best']['R']/WF[ch].noise_RMS
                    else:
                        R_dict[ki] = M[[ki]]['best']['R']/WF[ch].noise_RMS

                # recursively traverse the M dict for the best match.  The lowest-level match
                # will not have a 'best' entry
                this_key=this_kval
                while 'best' in M[this_key]:
                    this_key=M[this_key]['best']['key']
                # write out the best model information
                fit_params[ch].update(M[this_key])
                fit_params[ch]['noise_RMS']=WF[ch].noise_RMS[0]
                fit_params[ch]['tc']=WF[ch].tc[0]
                fit_params[ch]['Amax']=np.nanmax(WF[ch].p)
                fit_params[ch]['seconds_of_day']=WF[ch].seconds_of_day
                #fit_params[ch][WF_count]['delta_t'] -= WF[ch].t0
                fit_params['both']['shot'] = WF[ch].shots[0]
                sigma_last = np.maximum(sigma_last, M[this_key]['sigma'])
            fit_params['both']['sigma']=M[this_key]['sigma']
            searched_k_vals = np.array(sorted(R_dict.keys()))
            R = np.array([R_dict[ki] for ki in searched_k_vals]).ravel()

            R_max=fit_params['both']['R']*(1.+1./np.sqrt(N_samps))
            if np.sum(searched_k_vals>0)>=3:
                these=np.flatnonzero(searched_k_vals>0)
                if len(these) > 3:
                     ind_k_vals=np.argsort(R[these])
                     these=these[ind_k_vals[0:4]]
                E_roots=np.polynomial.polynomial.Polynomial.fit(np.log10(searched_k_vals[these]), R[these]-R_max, 2).roots()
                if np.any(np.imag(E_roots)!=0):
                    fit_params['both']['Kmax']=10**np.minimum(3,np.polynomial.polynomial.Polynomial.fit(np.log10(searched_k_vals[these]), R[these]-R_max, 1).roots()[0])
                    fit_params['both']['Kmin']=np.min(searched_k_vals[R<R_max])
                else:
                    fit_params['both']['Kmin']=10**np.min(E_roots)
                    fit_params['both']['Kmax']=10**np.max(E_roots)
            if (0. in searched_k_vals) and R[searched_k_vals==0]<R_max:
                fit_params['both']['Kmin']=0.
            #copy remaining waveform parameters to the output data structure
            for ch in channels:
                fit_params[ch]['shot']=WF[ch].shots[0]
                fit_params[ch]['t0']=WF[ch].t0[0]
                fit_params[ch]['t_shift']=WF[ch].t_shift[0]
                fit_params[ch]['noise_RMS']=WF[ch].noise_RMS[0]
            #print(this_key+[R[iR][0]])
            if return_data_est or DOPLOT:
                # call WF_misfit for each channel
                wf_est={}
                for ch, WFi in WF.items():
                    R0, wf_est=wf_misfit(fit_params[ch]['delta_t'], fit_params[ch]['sigma'], WF[ch], catalogs[ch], Ms[ch], [this_key[0]], return_data_est=True)
                    fit_params[ch]['wf_est']=wf_est
                    fit_params[ch]['t_shift']=WF[ch].t_shift

            if KEY_SEARCH_PLOT:
                ch_keys={}
                new_keys={}
                fig=plt.gcf();
                fig.clf()
                for ind, ch in enumerate(channels):
                    fig.add_subplot(2, 1, ind+1)
                    ch_keys[ch]=[[key[0:2]] for key in catalogs[ch].keys() if len(key) > 1]
                    new_keys[ch]=[key for key in ch_keys[ch] if key not in last_keys[ch]]
                    kxy=np.concatenate(ch_keys[ch], axis=0)
                    plt.plot(np.log10(kxy[:,0]), kxy[:,1],'k.')
                    if len(new_keys[ch]) > 0:
                        kxy_new=np.concatenate(new_keys[ch], axis=0)
                        plt.plot(np.log10(kxy_new[:,0]), kxy_new[:,1], 'ro')
                last_keys={ch:ch_keys[ch].copy() for ch in ch_keys.keys()}
            # report
            fit_param_list += [fit_params]
            if DOPLOT:
                plt.figure();
                colors={'IR':'r','G':'g'}
                this_title=''
                for ch in channels:
                    #plt.plot(WF[ch].t, integer_shift(WF[ch].p, delta_samp),'.', color=colors[ch])
                    plt.plot(WF[ch].t,  WF[ch].p, 'x', color=colors[ch])
                    plt.plot(WF[ch].t,  wf_est[ch], color=colors[ch])
                    this_title+='%s: K=%3.2g, dt=%3.2g, $\sigma$=%3.2g, R=%3.2f\n' % (ch, this_key[0], fit_params[ch]['delta_t'], fit_params[ch]['sigma'], fit_params[ch]['R'])
                plt.title(this_title[0:-2])
                print(WF_count)
            if M_list is not None:
                M_list += [Ms]
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            print("Exception thrown for shot %d" % WF[channels[0]].shots)
            print(e)
            pass
        if np.mod(WF_count, 1000)==0 and WF_count > 0:
            print('    N=%d, N_keys=%d, %d' % (WF_count, len(list(catalogs[channels[0]])), len(list(catalogs[channels[1]]))))

    result={}
    for ch in channels+['both']:
        result[ch]={}
        for field in WFp_empty[ch].keys():
            try:
                result[ch][field]=np.array([ii[ch][field] for ii in fit_param_list])
            except ValueError:
                print("problem with channel %s, field %s" %(ch, field))
    if return_catalogs:
        return result, catalogs
    else:
        return result
