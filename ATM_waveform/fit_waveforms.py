# -*- coding: utf-8 -*-0
"""
Created on Mon Nov  5 16:56:31 2018

@author: ben
"""
import numpy as np
#from numpy.linalg import inv
import matplotlib.pyplot as plt
#import bisect
from ATM_waveform.waveform import waveform
#import
#from ATM_waveform.corr_no_mean import corr_no_mean
from ATM_waveform.unrefined_misfit import unrefined_misfit
from ATM_waveform.refined_misfit import refined_misfit
from ATM_waveform.WFcatalog import WFcatalog
from ATM_waveform.golden_section_search import golden_section_search
from ATM_waveform.parabolic_search_refinement import parabolic_search_refinement
from ATM_waveform.wf_misfit import wf_misfit
from ATM_waveform.make_broadened_WF import make_broadened_WF
from ATM_waveform.broaden_p import broaden_p
DOPLOT=False


def integer_shift(p, delta, fill_value=np.NaN):
    result = np.empty_like(p)
    delta=np.int(delta)
    if delta > 0:
        result[:delta] = fill_value
        result[delta:] = p[:-delta]
    elif delta < 0:
        result[delta:] = fill_value
        result[:delta] = p[-delta:]
    else:
        result[:] = p
    return result


def gaussian(x, ctr, sigma):
    """
        return a normalized gaussian kernel centered on 'ctr' with width 'sigma'
    """
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-ctr)**2/2/sigma**2)




def wf_unrefined_misfit(delta_ind, sigma, WF, M, K0_top,  parent_WF=None):
    """
        Find the misfit between a scaled and shifted template and a waveform
    """
    delta_t=delta_ind*WF.dt[0]
    this_key = (K0_top, sigma, delta_t)
    if (this_key in M):
        return M[this_key]['R']
    else:
        broadened_p = parent_WF.p.copy()
        broadened_p2 = parent_WF.p_squared.copy()

        # check if the shifted version of the broadened waveform is in the catalog
        R = unrefined_misfit(int(delta_ind), broadened_p.ravel(), WF.p.ravel(),
                             broadened_p2.ravel(), WF.p_squared.ravel(),
                             parent_WF.mask.ravel(), WF.mask.ravel(),
                             len(broadened_p))
        M[this_key] = {'K0':K0_top, 'R':R, 'A':None, 'B':0., 'delta_t':delta_t, 'sigma':sigma}
    return R


def fit_shifted(delta_t_list, sigma, catalog, WF, M, K0_top, WF_top,
                t_tol=None, broadened_WF=None):
    """
    Find the shift value that minimizes the misfit between a template and a waveform

    performs a golden-section search along the time dimension
    """
    #G=np.ones((WF.p.size, 2))
    if t_tol is None:
        t_tol=WF['t_samp']/10.
    delta_t_spacing=delta_t_list[1]-delta_t_list[0]
    bnds=[np.min(delta_t_list)-5, np.max(delta_t_list)+5]
    if broadened_WF is None:
        broadened_WF = make_broadened_WF(sigma, K0_top, WF_top, catalog)

    dt=WF_top.dt
    if not isinstance(dt, (int, float)):
        dt=dt[0]
    #fDelta = lambda deltaTval: wf_misfit(deltaTval, sigma, WF, catalog, M,  K0_top, WF_top, parent_WF=broadened_WF)
    # if a 'best' value has been found for this K0_top, (because a different sigma has been searched already)
    # search the delta-t values around it
    if (K0_top,) in M and 'best' in M[(K0_top,)]:
        this_delta_t_list = delta_t_list[np.argsort(np.abs(delta_t_list-M[(K0_top,)]['best']['delta_t']))[0:2]]
        this_delta_t_list = np.concatenate([this_delta_t_list, [this_delta_t_list.mean()]])
    else:
        this_delta_t_list=delta_t_list

    delta_ind_list = np.unique(np.round(this_delta_t_list/dt).astype(int))
    ind_bnds = [bnds[0]/dt, bnds[1]/dt]

    # this is the index shift for which the time-zero values for the two waveforms
    # line up.
    delta_ind_ctr = ((WF.t[0] -( broadened_WF.t[0]))/dt)[0]

    #(np.round((WF.t[0]-broadened_WF.t[0])/WF.dt[0]))

    f_ind = lambda ind: wf_unrefined_misfit(int(ind-delta_ind_ctr), sigma, WF, M, K0_top, parent_WF=broadened_WF)

    golden_search_hist={}
    delta_ind_best, R_best = golden_section_search(f_ind, delta_ind_list,
                            delta_t_spacing/dt, bnds=ind_bnds, tol=1,
                            integer_steps=True, max_count=100,
                            search_hist=golden_search_hist,
                            refine_parabolic=False)

    # select a range of delta values to search for an optimal shift
    refine_search_ind = np.arange(delta_ind_list[0], delta_ind_list[-1]+1)
    refine_search_ind = refine_search_ind[np.argsort(np.abs(refine_search_ind-.01))].astype(int)
    for this_ind in delta_ind_best + refine_search_ind:
        delta_ind_refined, A, R = refined_misfit(this_ind-delta_ind_ctr, broadened_WF.p.ravel(), WF.p.ravel(),\
                                                broadened_WF.p_squared.ravel(), WF.p_squared.ravel(),
                                                broadened_WF.mask.ravel(),
                                                WF.mask.ravel(), len(WF.p))
        if delta_ind_refined >= 0 and delta_ind_refined <=1:
            break

    delta_t_best = -dt*this_ind
    delta_t_refined= -dt*(this_ind + delta_ind_refined)

    this_key=(K0_top, sigma, delta_t_best)
    M[this_key] = {'key':this_key,'sigma':sigma, 'K0':K0_top, 'R':R, 'A':A,
                   'delta_t':delta_t_refined}
    M[ (K0_top, sigma) ]['best'] = {'key':this_key, 'R':R, 'A':A,
                                    'delta_t':delta_t_refined}
    if False:
        plt.plot(WF.t, WF.p, '.'); plt.plot(broadened_WF.t+delta_t_refined, A*broadened_WF.p)
        # N.B. : this works much better:
        #plt.plot(WF.t-WF.t0, WF.p, '.'); plt.plot(broadened_WF.t-delta_t_best - (1+delta_ind_refined)*catalog.dt, A*broadened_WF.p)
    if 'best' not in M[(K0_top,)]:
        M[(K0_top,)]['best'] = {'key':this_key, 'R':R, 'delta_t':delta_t_refined}
    elif R < M[(K0_top,)]['best']['R']:
        M[(K0_top,)]['best'] = {'key':this_key, 'R':R, 'A':A, 'delta_t':delta_t_refined}
    return R



def broadened_misfit(delta_ts, sigma, WF, catalog, M, K0_top, WF_top,  t_tol=None, refine_parabolic=True,
                     update_catalog=True):
    """
    Calculate the misfit between a broadened template and a waveform (searching over a range of shifts)
    """
    this_key=(K0_top, sigma)
    if (this_key in M) and ('best' in M[this_key]):
        return M[this_key]['best']['R']
    else:
        M[this_key]={}
        if this_key in catalog:
            broadened_WF= catalog[this_key]
        else:
            parent_WF=WF_top
            # if we haven't already broadened the WF by sigma, try it now:

            if sigma==0:
                this_p=parent_WF.p
            else:
                this_p = broaden_p(parent_WF, sigma)
            if update_catalog:
                catalog.update(this_key, p=this_p, p_squared=this_p*this_p,
                               mask=np.isfinite(this_p).astype(np.int32),\
                                   t0=parent_WF.t0, tc=parent_WF.tc)
                broadened_WF = catalog[this_key]
            else:
                broadened_WF = waveform(parent_WF.t, this_p, p_squared=this_p*this_p, t0=parent_WF.t0, tc=parent_WF.tc)
                broadened_WF.mask = np.isfinite(this_p).astype(np.int32)

        return fit_shifted(delta_ts, sigma, catalog, WF,  M, K0_top, WF_top, t_tol=t_tol, broadened_WF=broadened_WF)

def fit_broadened(delta_ts, sigmas, WF, catalog,  M, K0_top, WF_top,
                  sigma_tol=None, sigma_max=5., t_tol=None, sigma_last=None):
    """
    Find the best broadening value that minimizes the misfit between a template and a waveform
    """
    if (K0_top,) not in M:
        M[(K0_top,)]={}
    fSigma = lambda sigma:broadened_misfit(delta_ts, sigma, WF, catalog, M, K0_top, WF_top, t_tol=t_tol)
    sigma_step=2*sigma_tol
    FWHM2sigma=2.355

    if sigma_tol is None:
        sigma_tol=.125

    if sigmas is None:
        try:
            sigma_template = WF_top.fwhm()[0]/FWHM2sigma
        except Exception as e:
            sigma_template = WF_top.fwhm()[0]/FWHM2sigma
            print(e)
        sigma_WF=WF.fwhm()[0]/FWHM2sigma

        broadening = sigma_WF**2-sigma_template**2
        if broadening < -sigma_tol**2:
            sigma0 = 0
            dSigma=sigma_tol/2
            sigmas=np.array([0, dSigma])
        else:
            sigma0=sigma_step*np.ceil(np.sqrt(np.maximum(0,  broadening))/sigma_step)
            dSigma=sigma_step*np.maximum(1, np.ceil(sigma0/sigma_step))
            sigmas=np.array([0., np.maximum(sigma_step, sigma0-dSigma), np.maximum(sigma_step, sigma0+dSigma)])
    else:
        dSigma=np.max(sigmas)/4.
    if np.any(~np.isfinite(sigmas)):
        print("NaN in sigma for %d " % WF.shots)


    if sigma_last is not None:
        i1=np.maximum(1, np.argmin(np.abs(sigmas-sigma_last)))
    else:
        i1=1
    sigma_list=[sigmas[0], sigmas[i1]]
    # search over sigma values:
    search_hist={}
    sigma_best, R_best = golden_section_search(fSigma, sigma_list, dSigma,
                                               bnds=[0, sigma_max], tol=sigma_tol,
                                               max_count=20, search_hist=search_hist)
    #if broadening < -sigma_tol**2  and sigma_best > 0:
    #    print("broadening is less than zero but sigma_best is greater than zero")
    this_key=(K0_top, sigma_best)
    M[(K0_top,)]['best']={'key':this_key,'R':R_best}
    return R_best

def fit_catalog(WFs, catalog_in, sigmas, delta_ts, t_tol=None, sigma_tol=None,
                return_data_est=False, return_catalog=False, catalog=None):
    """
    Search a library of waveforms for the best match between the broadened, shifted library waveform
    and the target waveforms

    Inputs:
        WFs: a waveform object, whose fields include:
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
        t_tol = WFs.dt*0.1
    if sigma_tol is None:
        sigma_tol = 0.25
    # make an empty output_dictionary
    WFp_empty={f:np.NaN for f in ['K0', 'K0_refined', 'R','A','B','delta_t','sigma','t0','Kmin','Kmax','shot']}
    if return_data_est:
        WFp_empty['wf_est']=np.zeros_like(WFs.t)+np.NaN

    # make an empty container where we will keep waveforms we've tried already
    if catalog is None:
        catalog=WFcatalog(WFs.nSamps, WFs.dt[0], t=catalog_in[0].t)
    input_K0 = [ kk for kk in np.sort(list(catalog_in)) ]

    W_catalog=np.zeros(len(input_K0))
    for ind, key in enumerate(input_K0):
        W_catalog[ind]=catalog_in[key].fwhm()[0]

    # loop over the library of templates
    for ii, kk in enumerate(input_K0):
        # check if we've searched this template before, otherwise copy it into
        # the library of checked templates
        if (kk,) not in catalog:
            # make a copy of the current template
            temp=catalog_in[kk]
            catalog.update((kk,), p=temp.p, p_squared=temp.p*temp.p, mask=np.isfinite(temp.mask).astype(np.int32), t0=temp.t0, tc=temp.tc)

    fit_params=[WFp_empty.copy() for ii in range(WFs.size)]
    sigma_last=None
    t_center=WFs.t.mean()
    # loop over input waveforms
    for WF_count in range(WFs.size):
        # this extracts the WF from the data, returns a waveform with t0=0
        WF=WFs[WF_count]
        if WF.nPeaks > 1:
            continue
        # shift the waveform to put its tc at the center of the time vector
        delta_samp=np.round((WF.tc-t_center)/WF.dt)
        WF.p=integer_shift(WF.p, -delta_samp)
        WF.p_squared=WF.p**2
        WF.t0=-delta_samp*WF.dt  # this is '+' in fit_2color_waveforms
        WF.mask=np.isfinite(WF.p).astype(np.int32)
        # set up a matching dictionary (contains keys of waveforms and their misfits)
        M={}
        # this is the bulk of the work, and it's where problems happen.  Wrap it in a try:
        # and write out errors to be examined later
        if True:
            if len(input_K0)>1:
                # Search over input keys to find the best misfit between this template and the waveform
                fB=lambda ind: fit_broadened(delta_ts, None,  WF, catalog, M, input_K0[ind], catalog[(input_K0[ind],)],
                                             sigma_tol=sigma_tol, t_tol=t_tol, sigma_last=sigma_last)
                W_match_ind=np.flatnonzero(W_catalog >= WF.fwhm()[0])
                if len(W_match_ind) >0:
                    ind=np.array(tuple(set([0, W_match_ind[0]-2,  W_match_ind[0]+2])))
                    ind=ind[(ind >= 0) & (ind<len(input_K0))]
                else:
                    ind=[2, 4]
                # search over K0 in integer steps
                iBest, Rbest = golden_section_search(fB, ind, delta_x=2, bnds=[0, len(input_K0)-1], integer_steps=True, tol=1)
                iBest=int(iBest)
            else:
                # only one key in input, return its misfit
                Rbest=fit_broadened(delta_ts, None, WF, catalog, M, input_K0, sigma_tol=sigma_tol, t_tol=t_tol, sigma_last=sigma_last)
                iBest=0
            this_key= (input_K0[iBest],)
            M['best']={'key':this_key, 'R':Rbest}
            searched_keys = np.array([this_key for this_key in input_K0 if (this_key,) in M])
            R=np.array([M[(ki,)]['best']['R'] for ki in searched_keys])

            # recursively traverse the M dict for the best match.  The lowest-level match
            # will not have a 'best' entry
            while 'best' in M[this_key]:
                this_key=M[this_key]['best']['key']
            # write out the best model information
            fit_params[WF_count].update(M[this_key])
            fit_params[WF_count]['delta_t'] -= WF.t0[0]
            fit_params[WF_count]['shot'] = WF.shots[0]
            sigma_last=M[this_key]['sigma']
            R_max=fit_params[WF_count]['R']*(1.+1./np.sqrt(WF.t.size))
            # default value for K0_ref
            fit_params[WF_count]['K0_refined'] = M[this_key]['K0']
            if np.sum(searched_keys>0)>=3:
                these=np.flatnonzero(searched_keys>0)
                if len(these) > 3:
                     ind_keys=np.argsort(R[these])
                     these=these[ind_keys[0:4]]
                E_roots=np.polynomial.polynomial.Polynomial.fit(np.log10(searched_keys[these]), R[these]-R_max, 2).roots()
                if np.any(np.imag(E_roots)!=0):
                    fit_params[WF_count]['Kmax']=10**np.minimum(3,np.polynomial.polynomial.Polynomial.fit(np.log10(searched_keys[these]), R[these]-R_max, 1).roots()[0])
                    try:
                        fit_params[WF_count]['Kmin']=np.min(searched_keys[R<R_max])
                    except Exception:
                        pass
                else:
                    fit_params[WF_count]['Kmin']=10**np.min(E_roots)
                    fit_params[WF_count]['Kmax']=10**np.max(E_roots)
                # parabolic refinement of R:
                ii = np.argmin(R) + np.array([-1, 0, 1], dtype=int)

                if np.max(ii) < len(searched_keys):
                    ki, ri = parabolic_search_refinement(searched_keys[ii], R[ii])
                    fit_params[WF_count]['K0_refined'] = ki

            if (0. in searched_keys) and R[searched_keys==0]<R_max:
                fit_params[WF_count]['Kmin']=0.

            if False:

                this=catalog[(fit_params[WF_count]['K0'],)];p1=broaden_p(this, M[this_key]['sigma']);plt.plot(WF.t+WF.t0, WF.p); plt.plot(this.t-fit_params[WF_count]['delta_t'], fit_params[WF_count]['A']*p1  )

            #print(this_key+[R[iR][0]])
            if return_data_est or DOPLOT:
                WF.t=WF.t-WF.t0
                R0, wf_est=wf_misfit(fit_params[WF_count]['delta_t'], fit_params[WF_count]['sigma'], WFs[WF_count], catalog, M, [this_key[0]], return_data_est=True)
                fit_params[WF_count]['wf_est']=wf_est#integer_shift(wf_est, -delta_samp)
            if DOPLOT:
                plt.figure();
                plt.plot(WF.t, integer_shift(WF.p, delta_samp),'k.')
                plt.plot(WF.t, wf_est,'r')
                plt.title('K=%f, dt=%f, sigma=%f, R=%f' % (this_key[0], fit_params[WF_count]['delta_t'], fit_params[WF_count]['sigma'], fit_params[WF_count]['R']))
                print(WF_count)
        #except KeyboardInterrupt:
        #    sys.exit()
        #except Exception as e:
        #    print("Exception thrown for shot %d" % WF.shots)
        #    print(e)
        #    pass
        if np.mod(WF_count, 1000)==0 and WF_count > 0:
            print('    N=%d, N_keys=%d' % (WF_count, len(list(catalog))))

    result=dict()
    for key in WFp_empty:
        if key in ['wf_est']:
            result[key]=np.concatenate( [ ii['wf_est'][:, None] for ii in fit_params ], axis=1 )
        else:
            result[key]=np.array([ii[key] for ii in fit_params]).ravel()

    if return_catalog:
        return result, catalog
    else:
        return result
