import numpy as np
import pandas as pd
import logging
import trizod.bmrb.bmrb as bmrb
import scipy
import warnings
from trizod.constants import BBATNS, REFINED_WEIGHTS, Z_CORRECTION

def convChi2CDF(rss,k):
    with np.errstate(divide='ignore', invalid='ignore'):
        # I expect to see RuntimeWarnings in this block
        # k can be 0 at some 
        res = ((((rss/k)**(1.0/6))-0.50*((rss/k)**(1.0/3))+1.0/3*((rss/k)**(1.0/2)))\
            - (5.0/6-1.0/9/k-7.0/648/(k**2)+25.0/2187/(k**3)))\
            / np.sqrt(1.0/18/k+1.0/162/(k**2)-37.0/11664/(k**3))
    return res

def comp2pred_arr(predshiftdct, bbshifts_arr, bbshifts_mask):
    #cmparr = np.zeros(shape=(len(seq), len(BBATNS)))
    # convert predshift dict to np array (TODO: do this in potenci...)
    predshift_arr = np.zeros(shape=bbshifts_arr.shape)
    predshift_mask = np.full(shape=bbshifts_mask.shape, fill_value=False)
    for res,aa in predshiftdct:
        i = res - 1
        for j,at in enumerate(BBATNS):
            if at in predshiftdct[(res,aa)]:
                if predshiftdct[(res,aa)][at] is not None:
                    predshift_arr[i, j] = predshiftdct[(res,aa)][at]
                    predshift_mask[i, j] = True
    cmparr =  np.subtract(bbshifts_arr, predshift_arr, where=bbshifts_mask & predshift_mask, out=bbshifts_arr)
    return cmparr, BBATNS, bbshifts_mask & predshift_mask

def compute_running_offsets(cmparr, mask, minAIC=999.):
    w_ = np.array([REFINED_WEIGHTS[at] for at in BBATNS]) # ensure same order
    shw_ = cmparr / w_
    df = pd.DataFrame(shw_).mask(~mask)
    # compute rolling stadard deviation over detected shifts (missing values are ignored and streched by rolling window)
    at_stdc = []
    at_roff = []
    at_std0 = []
    for i in range(7): # TODO: not necessary to compute anything but at_stdc for all values. Only the selected position would suffice
        roll = df[i].dropna().rolling(9, center=True)
        at_stdc.append(roll.std(ddof=0))
        at_roff.append(roll.mean())
        at_std0.append(roll.apply(lambda x : np.sqrt(x.pow(2).mean())))
    runstds_ = pd.concat(at_stdc, axis=1).reindex(pd.Index([i for i in range(len(cmparr))]))
    runoffs_ = pd.concat(at_roff, axis=1).reindex(pd.Index([i for i in range(len(cmparr))]))
    runstd0s_ = pd.concat(at_std0, axis=1).reindex(pd.Index([i for i in range(len(cmparr))]))
    # get index with the lowest mean rolling stddev for which all ats were detected (all that were detected anywhere for this sample)
    runstds_val = runstds_[runstds_.columns[mask.any(axis=0)]].dropna(axis=0).mean(axis=1)
    try:
        min_idx_ = runstds_val.idxmin()
    except ValueError:
        return None # still not found
    
    offdct_ = {}
    for col in runstds_.dropna(how='all', axis=1).columns: # for all at shifts that were detected anywhere in this sample
        at = BBATNS[col]
        roff = runoffs_.loc[min_idx_][col]
        std0 = runstd0s_.loc[min_idx_][col]
        stdc = runstds_.loc[min_idx_][col]
        dAIC = np.log(std0/stdc) * 9 - 1 # difference in Akaike’s information criterion, 9 is width of window
        logging.getLogger('trizod.scoring').info(f'minimum running average: {at} {roff} {dAIC}')
        if dAIC > minAIC:
            logging.getLogger('trizod.scoring').info(f'using offset correction: {at} {roff} {dAIC}')
            offdct_[at] = roff
        else:
            logging.getLogger('trizod.scoring').info(f'rejecting offset correction due to low dAIC: {at} {roff} {dAIC}')
            #offdct_[at] = 0.0

    return offdct_ #with the running offsets

def compute_offsets(shw_, accdct_, minAIC=999.):
    anum_ = np.sum(accdct_, axis=0)
    # I expect to see RuntimeWarnings in this block
    # accdct_ can contain fully-False columns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        newoffdct_ = np.nanmean(shw_, axis=0, where=accdct_)
        astd0_ = np.sqrt(np.nanmean(shw_ ** 2, axis=0, where=accdct_))
        astdc_ = np.nanstd(shw_, axis=0, where=accdct_)
        with np.errstate(divide='ignore'):
            # for anum_ == 1, astdc_ is 0. resulting in an adAIC_ of inf as consequence of division-by-zero
            # this would be problematic if not all offsets with anum_ < 4 were rejected anyways
            adAIC_ = np.log(astd0_ / astdc_) * anum_ - 1
    reject_mask = (adAIC_ < minAIC) | (anum_ < 4)
    astdc_[reject_mask] = astd0_[reject_mask]
    newoffdct_[reject_mask] = 0.
    newoffdct_ = {at:val for at,val in zip(BBATNS,newoffdct_)}
    return newoffdct_

def get_outlier_mask(cdfs3_, cdfs_, ashwi_, mask, cdfthr=6.0):
    oldct_ = ashwi_ > cdfthr
    totnum_ = mask.sum(axis=1)
    finaloutli_ = (cdfs_ > cdfthr) | ((cdfs3_ > cdfthr) & (cdfs_ > 0.0) & (totnum_ > 0))
    ol_ = mask & (np.bitwise_or(np.expand_dims(finaloutli_, axis=1), oldct_)) # mask for the outliers
    return ol_

def get_std_norm_diffs(cmparr, mask, offdct={}):
    w_ = np.array([REFINED_WEIGHTS[at] for at in BBATNS]) # ensure same order
    off_ = np.array([offdct.get(at, 0.) for at in BBATNS])
    shw_ = cmparr / w_
    ashwi_ = shw_.copy() # need to do the copy here, because shw_ is reused later and would otherwise be partially overwritten due to the out=
    ashwi_ = np.abs(np.subtract(shw_, off_, where=mask, out=ashwi_))
    return shw_, ashwi_

def compute_zscores(ashwi3, k3, mask, corr=False):
    indices = np.where(np.any(mask,axis=1))
    mini, maxi = indices[0][0], indices[0][-1]
    tot3f_ = (np.minimum(ashwi3, 4.0) ** 2).sum(axis=1)
    totn3f_ = k3
    cdfs3_ = convChi2CDF(tot3f_, totn3f_)
    if corr:
        for k in range(1,22):
            m = totn3f_ == k
            cdfs3_[m] += cdfs3_[m] * Z_CORRECTION[k]
    cdfs3_[:mini] = np.nan
    cdfs3_[maxi+1:] = np.nan
    return cdfs3_

def compute_pscores(ashwi3, k3, mask, quotient=2.0, limit=4.0):
    indices = np.where(np.any(mask,axis=1))
    mini, maxi = indices[0][0], indices[0][-1]

    if limit:
        p = np.prod(scipy.stats.norm.pdf(np.minimum(ashwi3, limit) / quotient) / scipy.stats.norm.pdf(0.), axis=1)
        with np.errstate(divide='ignore'): # zeros are to be expected; resulting NANs are ok
            p = p ** (1/k3)
        minimum = scipy.stats.norm.pdf(limit / quotient) / scipy.stats.norm.pdf(0.)
        p = (p - minimum) / (1. - minimum)
    else:
        p = np.prod(scipy.stats.norm.pdf(ashwi3 / quotient) / scipy.stats.norm.pdf(0.), axis=1)
        with np.errstate(divide='ignore'): # zeros are to be expected; resulting NANs are ok
            p = p ** (1/k3)
    p[k3 == 0] = np.nan
    p[:mini] = np.nan
    p[maxi+1:] = np.nan
    return p

def convert_to_triplet_data(ashwi_, mask):
    ashwi3 = ashwi_.copy()
    ashwi3[~mask] = 0.
    ashwi3 = np.column_stack([np.pad(ashwi3, ((1,1),(0,0)))[2:],   ashwi3,   np.pad(ashwi3, ((1,1),(0,0)))[:-2]])
    k3     = np.column_stack([np.pad(mask, ((1,1),(0,0)))[2:],mask,np.pad(mask, ((1,1),(0,0)))[:-2]]).sum(axis=1)
    return ashwi3, k3

def get_offset_corrected_wSCS(seq, shifts, predshiftdct):
    # get polymer sequence and chemical backbone shifts
    ret = bmrb.get_valid_bbshifts(shifts, seq)
    if ret is None:
        logging.getLogger('trizod.scoring').error(f'retrieving backbone shifts failed')
        return
    bbshifts_arr, bbshifts_mask = ret
    
    # compare predicted to actual shifts
    cmparr, _, cmp_mask = comp2pred_arr(predshiftdct, bbshifts_arr, bbshifts_mask)
    totbbsh = np.sum(cmp_mask)
    if totbbsh == 0:
        logging.getLogger('trizod.scoring').error(f'no comparable backbone shifts')
        return
    logging.getLogger('trizod.scoring').info(f"total number of backbone shifts: {totbbsh}")

    off0 = {at:0.0 for at in BBATNS}
    shw0, ashwi0 = get_std_norm_diffs(cmparr, cmp_mask, off0)
    cdfs0 = compute_zscores(ashwi0, cmp_mask.sum(axis=1), cmp_mask)
    cdfs30 = compute_zscores(*convert_to_triplet_data(ashwi0, cmp_mask), cmp_mask)
    ol0 = get_outlier_mask(cdfs30, cdfs0, ashwi0, cmp_mask, cdfthr=6.0)
    noff0 = compute_offsets(shw0, cmp_mask & ~ol0, minAIC=6.0)
    av0 = np.nanmean(cdfs30)
    offf = noff0
    olf = ol0

    offr = compute_running_offsets(cmparr, cmp_mask, minAIC=6.0)
    if offr is None:
        logging.getLogger('trizod.scoring').warning(f'no running offset could be estimated')
    elif np.any([v != 0. for v in offr.values()]):
        shwc, ashwic = get_std_norm_diffs(cmparr, cmp_mask, offr)
        cdfsc = compute_zscores(ashwic, cmp_mask.sum(axis=1), cmp_mask)
        cdfs3c = compute_zscores(*convert_to_triplet_data(ashwic, cmp_mask), cmp_mask)
        avc = np.nanmean(cdfs3c)
        if av0 >= avc: # use offset correction only if it leads to, in average, better accordance with the POTENCI model (more disordered)
            olc = get_outlier_mask(cdfs3c, cdfsc, ashwic, cmp_mask, cdfthr=6.0)
            noffc = compute_offsets(shwc, cmp_mask & ~olc, minAIC=6.)
            offf = noffc
            olf = olc

    shwf, ashwif = get_std_norm_diffs(cmparr, cmp_mask, offf)
    return shwf, ashwif, cmp_mask, olf, offf, shw0, ashwi0, ol0, off0