import numpy as np
import pandas as pd
import logging
import TriZOD.bmrb as bmrb
import scipy
import warnings

BBATNS = ['C','CA','CB','HA','H','N','HB']
REFINED_WEIGHTS = {'C':0.1846, 'CA':0.1982, 'CB':0.1544, 'HA':0.02631, 'H':0.06708, 'N':0.4722, 'HB':0.02154}
Z_CORRECTION = {
 1: 4.05308849681796,
 2: 2.277173086010837,
 3: 1.6090276518015432,
 4: 1.2447910781901539,
 5: 1.009432092595334,
 6: 0.8413717245284563,
 7: 0.7131281335003391,
 8: 0.6105038279581856,
 9: 0.5253913758783114,
 10: 0.4528092358941966,
 11: 0.3895201437851491,
 12: 0.3333250286536015,
 13: 0.2826753620267024,
 14: 0.23644774008331337,
 15: 0.1938064669517393,
 16: 0.15411640792754794,
 17: 0.11688583537247893,
 18: 0.08172784553947854,
 19: 0.048333648566945,
 20: 0.016453664764633225,
 21: 0.}#-0.014116118689604495}

def convChi2CDF(rss,k):
    with np.errstate(divide='ignore', invalid='ignore'):
        # I expect to see RuntimeWarnings in this block
        # k can be 0 at some 
        res = ((((rss/k)**(1.0/6))-0.50*((rss/k)**(1.0/3))+1.0/3*((rss/k)**(1.0/2)))\
            - (5.0/6-1.0/9/k-7.0/648/(k**2)+25.0/2187/(k**3)))\
            / np.sqrt(1.0/18/k+1.0/162/(k**2)-37.0/11664/(k**3))
    return res

#def convChi2CDF(rss,k):
#    mask = k>0
#    div_rss_k = np.zeros(rss.shape, dtype=rss.dtype)
#    div_rss_k = np.divide(rss, k, where=mask)
#    s1 = np.divide((1/9),k, where=mask)# np.zeros(k.shape, dtype=k.dtype)
#    s2 = np.divide((7/648),(k**2), where=mask)# np.zeros(k.shape, dtype=k.dtype)
#    s3 = np.divide((25/2187),(k**3), where=mask)# np.zeros(k.shape, dtype=k.dtype)
#    sqrt = np.sqrt(np.divide((1/18),k, where=mask) + np.divide((1/162),(k**2), where=mask) - np.divide((37/11664),(k**3), where=mask), where=mask)
#    ret = np.divide((((div_rss_k**(1/6)) - 0.50*(div_rss_k**(1/3)) + (1/3)*(div_rss_k**(1/2)))\
#                     - ((5/6) - s1 - s2 + s3)),
#                     sqrt, where=mask)
#    ret[~mask] = np.nan
#    return ret

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
    #runstds_val = runstds_.dropna(how='all', axis=1).dropna(axis=0).mean(axis=1)
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
        logging.getLogger('trizod.trizod').info(f'minimum running average: {at} {roff} {dAIC}')
        if dAIC > minAIC:
            logging.getLogger('trizod.trizod').info(f'using offset correction: {at} {roff} {dAIC}')
            offdct_[at] = roff
        else:
            logging.getLogger('trizod.trizod').info(f'rejecting offset correction due to low dAIC: {at} {roff} {dAIC}')
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
    #accdct_ = mask & ~ol_ # mask for the validated data (where there is shift data, predictions and not outliers)
    #numol_ = ol_.sum()
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
    #tot_ = (np.minimum(ashwi_, 4.0) ** 2).sum(axis=1)
    #totnum_ = mask.sum(axis=1)
    tot3f_ = (np.minimum(ashwi3, 4.0) ** 2).sum(axis=1)
    totn3f_ = k3
    #cdfs_ = convChi2CDF(tot_, totnum_)
    # TODO: find more elegant solution than using maxi, mini (not safe: using mask.any(axis=1) !?! --> gaps in between are to be accepted)
    # what would work is using pandas for this...
    #tot_[:mini] = 0.
    #tot_[maxi+1:] = 0.
    #totnum_[:mini] = 0.
    #totnum_[maxi+1:] = 0.
    #tot3f_ =  np.pad(tot_, 1)[2:]    + tot_    + np.pad(tot_, 1)[:-2]
    #totn3f_ = np.pad(totnum_, 1)[2:] + totnum_ + np.pad(totnum_, 1)[:-2]
    cdfs3_ = convChi2CDF(tot3f_, totn3f_)
    if corr:
        for k in range(1,22):
            m = totn3f_ == k
            cdfs3_[m] += cdfs3_[m] * Z_CORRECTION[k]
    #cdfs3_[k3 == 0] = np.nan # already nan due to division-by-zero
    cdfs3_[:mini] = np.nan
    cdfs3_[maxi+1:] = np.nan
    return cdfs3_#, cdfs_

def compute_pscores(ashwi3, k3, mask, quotient=2.0, limit=4.0):
    indices = np.where(np.any(mask,axis=1))
    mini, maxi = indices[0][0], indices[0][-1]
    #tot_ = ashwi_.copy()
    #tot_[~mask] = 0.
    #tot3f_ =  np.column_stack([np.pad(tot_, ((1,1),(0,0)))[2:],   tot_,   np.pad(tot_, ((1,1),(0,0)))[:-2]])
    #totn3f_ = np.column_stack([np.pad(mask, ((1,1),(0,0)))[2:],mask,np.pad(mask, ((1,1),(0,0)))[:-2]])

    if limit:
        p = np.prod(scipy.stats.norm.pdf(np.minimum(ashwi3, limit) / quotient) / scipy.stats.norm.pdf(0.), axis=1)
        with np.errstate(divide='ignore'): # zeros are to be expected; resulting NANs are expected 
            p = p ** (1/k3)
        minimum = scipy.stats.norm.pdf(limit / quotient) / scipy.stats.norm.pdf(0.)
        p = (p - minimum) / (1. - minimum)
    else:
        p = np.prod(scipy.stats.norm.pdf(ashwi3 / quotient) / scipy.stats.norm.pdf(0.), axis=1)
        with np.errstate(divide='ignore'): # zeros are to be expected; resulting NANs are expected 
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
    bbshifts, bbshifts_arr, bbshifts_mask = bmrb.get_valid_bbshifts(shifts, seq)
    if bbshifts is None:
        logging.getLogger('trizod.trizod').error(f'retrieving backbone shifts failed')
        return
    
    # compare predicted to actual shifts
    cmparr, _, cmp_mask = comp2pred_arr(predshiftdct, bbshifts_arr, bbshifts_mask)
    totbbsh = np.sum(cmp_mask)
    if totbbsh == 0:
        logging.getLogger('trizod.trizod').error(f'no comparable backbone shifts')
        return
    logging.getLogger('trizod.trizod').info(f"total number of backbone shifts: {totbbsh}")

    off0 = {at:0.0 for at in BBATNS}
    #armsd0,fra0,noff0,cdfs30 = results_w_offset(cmparr, cmp_mask, BBATNS, offdct=off0, minAIC=6.0)
    shw0, ashwi0 = get_std_norm_diffs(cmparr, cmp_mask, off0)
    #cdfs30,cdfs0 = compute_zscores(ashwi0, cmp_mask)
    cdfs0 = compute_zscores(ashwi0, cmp_mask.sum(axis=1), cmp_mask)
    cdfs30 = compute_zscores(*convert_to_triplet_data(ashwi0, cmp_mask), cmp_mask)
    ol0 = get_outlier_mask(cdfs30, cdfs0, ashwi0, cmp_mask, cdfthr=6.0)
    noff0 = compute_offsets(shw0, cmp_mask & ~ol0, minAIC=6.0)
    av0 = np.nanmean(cdfs30)
    offf = noff0
    olf = ol0

    offr = compute_running_offsets(cmparr, cmp_mask, minAIC=6.0)
    if offr is None:
        logging.getLogger('trizod.trizod').warning(f'no running offset could be estimated')
    elif np.any([v != 0. for v in offr.values()]):
        #armsdc,frac,noffc,cdfs3c = results_w_offset(cmparr, cmp_mask, BBATNS, offdct=offr, minAIC=6.0)
        shwc, ashwic = get_std_norm_diffs(cmparr, cmp_mask, offr)
        #cdfs3c,cdfsc = compute_zscores(ashwic, cmp_mask)
        cdfsc = compute_zscores(ashwic, cmp_mask.sum(axis=1), cmp_mask)
        cdfs3c = compute_zscores(*convert_to_triplet_data(ashwic, cmp_mask), cmp_mask)
        avc = np.nanmean(cdfs3c)
        if av0 >= avc: # use offset correction only if it leads to, in average, better accordance with the POTENCI model (more disordered)
            olc = get_outlier_mask(cdfs3c, cdfsc, ashwic, cmp_mask, cdfthr=6.0)
            noffc = compute_offsets(shwc, cmp_mask & ~olc, minAIC=6.)
            offf = noffc
            olf = olc

    #cdfs3 = results_w_offset(cmparr, cmp_mask, BBATNS, dataset=True, offdct=offdct, minAIC=6.0)
    shwf, ashwif = get_std_norm_diffs(cmparr, cmp_mask, offf)
    #_,cdfs3f,_,_ = compute_zscores(ashwif, cmp_mask)
    return shwf, ashwif, cmp_mask, olf, offf, shw0, ashwi0, ol0, off0