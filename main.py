import os
import logging
import argparse
import TriZOD.potenci as potenci
import TriZOD.trizod as trizod
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('ID', type=int,
                        help='identifier of a BMRB entry')
    parser.add_argument('--bmrb_dir', '-d', default='.',
                        help='directory to look for and store bmrb data files')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    args.bmrb_dir = os.path.abspath(args.bmrb_dir)
    return args

def comp2pred(predshiftdct,bbshifts,seq):
    bbatns =['C','CA','CB','HA','H','N','HB']
    cmpdct = {}
    shiftdct = {}
    for i in range(1,len(seq)-1):
        if i in bbshifts:
            trip=seq[i-1:i+1]
            if i==1:
                pent='n'+     trip+seq[i+2]
            elif i==len(seq)-2:
                pent=seq[i-2]+trip+'c'
            else:
                pent=seq[i-2]+trip+seq[i+2]
            for at in bbatns:
                if at in bbshifts[i]:
                    sho = bbshifts[i][at][0]
                    shp = predshiftdct[(i+1,seq[i])][at]
                    if shp is not None:
                        shiftdct[(i,at)] = [sho,pent]
                        diff=sho-shp
                        logging.debug(f"diff is: {i} {seq[i]} {at} {sho} {shp} {abs(diff)} {diff}")
                        if at not in cmpdct:
                            cmpdct[at] = {}
                        cmpdct[at][i] = diff
    return cmpdct, shiftdct

def convChi2CDF(rss,k):
    return ((((rss/k)**(1.0/6))-0.50*((rss/k)**(1.0/3))+1.0/3*((rss/k)**(1.0/2)))\
            - (5.0/6-1.0/9/k-7.0/648/(k**2)+25.0/2187/(k**3)))\
            / np.sqrt(1.0/18/k+1.0/162/(k**2)-37.0/11664/(k**3))

def get_offset_correction(dct, minAIC=999.):
    #bbatns=['C','CA','CB','HA','H','N','HB']
    refined_weights = {'C':0.1846, 'CA':0.1982, 'CB':0.1544, 'HA':0.02631, 'H':0.06708, 'N':0.4722, 'HB':0.02154}
    #outlivals =       {'C':5.0000, 'CA':7.0000, 'CB':7.0000, 'HA':1.80000, 'H':2.30000, 'N':12.000, 'HB':1.80000}
    #wdct = {
    #    'N':  [-0.0626, 0.0617,  0.2635],
    #    'C':  [ 0.2717, 0.2466,  0.0306],
    #    'CA': [ 0.2586, 0.2198,  0.0394],
    #    'CB': [-0.2635, 0.1830, -0.1877],
    #    'H':  [-0.3620, 1.3088,  0.3962],
    #    'HA': [-1.0732, 0.4440, -0.4673],
    #    'HB': [ 0.5743, 0.2262, -0.3388]
    #}
    #dats = {}
    maxi = max([max(dct[at].keys()) for at in dct])
    mini = min([min(dct[at].keys()) for at in dct])#is often 1
    nres = maxi - mini + 1
    #resids = range(mini+1,maxi+2)
    #tot = np.zeros(nres)
    #newtot = np.zeros(nres)
    #newtotsgn = np.zeros(nres)
    #newtotsgn1 = np.zeros(nres)
    #newtotsgn2 = np.zeros(nres)
    totnum = np.zeros(nres)
    #allrmsd = []
    #totbbsh = 0
    #oldct = {}
    allruns = np.zeros(nres)
    rdct = {}

    for at in dct:
        #vol = outlivals[at]
        #subtot = np.zeros(nres)
        #subtot1 = np.zeros(nres)
        #subtot2 = np.zeros(nres)
        #if dataset is not None:
        #    dataset[at][self.bmrID] = []
        A = np.array(list(dct[at].items()))
        #totbbsh += len(A)
        #I = bbatns.index(at)
        w = refined_weights[at]
        shw = A[:,1] / w
        #off = np.average(shw)
        #rms0 = np.sqrt(np.average(shw**2))
        #if offdct is not None:
        #    shw -= offdct[at] #offset correction
        #    print ('using predetermined offset correction', at, offdct[at], offdct[at]*w)
        #shwl = list(shw)
        for i in range(len(A)):
            resi = int(A[i][0]) - mini #minimum value for resi is 0
            #ashwi = abs(shw[i])
            #if ashwi > cdfthr:
            #    oldct[(at,resi)] = ashwi
            #tot[resi] += min(4.0, ashwi)**2
            #for k in [-1,0,1]:
            #    if 0 <= (resi + k) < len(subtot):
            #        subtot[resi+k]  += np.clip(shw[i]*w, -vol, vol) * wdct[at][0]
            #        subtot1[resi+k] += np.clip(shw[i]*w, -vol, vol) * wdct[at][1]
            #        subtot2[resi+k] += np.clip(shw[i]*w, -vol, vol) * wdct[at][2]
            totnum[resi]+=1
            #if offdct is None:
            if 3 < i < len(A)-4:
                vals = shw[i-4:i+5]
                runstd = np.std(vals)
                allruns[resi] += runstd
                if not resi in rdct:
                    rdct[resi]={}
                rdct[resi][at] = np.average(vals), np.sqrt(np.average(vals**2)), runstd
        #dats[at] = shw
        #stdw = np.std(shw)
        #dAIC = np.log(rms0 / stdw) * len(A) - 1
        #print('rmsd:', at, stdw, off, dAIC)
        #allrmsd.append(stdw)
        #newtot += (subtot / 3.0)**2
        #newtotsgn += subtot
        #newtotsgn1 += subtot1
        #newtotsgn2 += subtot2
    #T0 = list(tot / totnum)
    #cdfs = convChi2CDF(tot, totnum)
    #Th = list(tot / totnum * 0.5)
    #Ts = list(tot)
    #Tn = list(totnum)
    #tot3   = np.array([0,0] + Th) + np.array([0] + T0 + [0]) + np.array(Th + [0,0])
    #tot3f  = np.array([0,0] + Ts) + np.array([0] + Ts + [0]) + np.array(Ts + [0,0])
    #totn3f = np.array([0,0] + Tn) + np.array([0] + Tn + [0]) + np.array(Tn + [0,0])
    #cdfs3 = convChi2CDF(tot3f[1:-1], totn3f[1:-1])
    #newrms = (newtot*3) / totn3f[1:-1]
    #newcdfs = convChi2CDF(newtot*3, totn3f[1:-1])
    #avc = np.average(cdfs3[cdfs3<20.0])
    #numzs = len(cdfs3[cdfs3<20.0])
    #numzslt3 = len(cdfs3[cdfs3<cdfthr])
    #stdcp = np.std(cdfs3[cdfs3<20.0])
    #atot = np.sqrt(tot3/2)[1:-1]
    #aresids = np.array(resids)
    #if offdct is None:
    tr = (allruns / totnum)[4:-4]
    offdct = {}
    mintr = None
    minval = 999
    for j in range(len(tr)):
        if j+4 in rdct and len(rdct[j+4]) == len(dct): #all ats must be represented for this res
            if tr[j]<minval:
                minval = tr[j]
                mintr = j
    if mintr==None:
        return None #still not found
    #print(len(tr), len(resids[4:-4]), len(atot), mintr+4, min(tr), tr[mintr])##,tr)
    for at in rdct[mintr+4]:
        roff,std0,stdc = rdct[mintr+4][at]
        dAIC = np.log(std0/stdc) * 9 - 1
        print('minimum running average', at, roff, dAIC)
        if dAIC > minAIC:
            print('using offset correction:', at, roff, dAIC)
            offdct[at] = roff
        else:
            print('rejecting offset correction due to low dAIC:', at, roff, dAIC)
            offdct[at] = 0.0
    return offdct #with the running offsets

def results_w_offset(dct, shiftdct, dataset=None, offdct=None, minAIC=999., cdfthr=6.0):
    bbatns=['C','CA','CB','HA','H','N','HB']
    refined_weights = {'C':0.1846, 'CA':0.1982, 'CB':0.1544, 'HA':0.02631, 'H':0.06708, 'N':0.4722, 'HB':0.02154}
    outlivals =       {'C':5.0000, 'CA':7.0000, 'CB':7.0000, 'HA':1.80000, 'H':2.30000, 'N':12.000, 'HB':1.80000}
    wdct = {
        'N':  [-0.0626, 0.0617,  0.2635],
        'C':  [ 0.2717, 0.2466,  0.0306],
        'CA': [ 0.2586, 0.2198,  0.0394],
        'CB': [-0.2635, 0.1830, -0.1877],
        'H':  [-0.3620, 1.3088,  0.3962],
        'HA': [-1.0732, 0.4440, -0.4673],
        'HB': [ 0.5743, 0.2262, -0.3388]
    }
    dats = {}
    maxi = max([max(dct[at].keys()) for at in dct])
    mini = min([min(dct[at].keys()) for at in dct])#is often 1
    nres = maxi-mini+1
    resids = range(mini+1,maxi+2)
    tot = np.zeros(nres)
    newtot = np.zeros(nres)
    newtotsgn = np.zeros(nres)
    newtotsgn1 = np.zeros(nres)
    newtotsgn2 = np.zeros(nres)
    totnum = np.zeros(nres)
    allrmsd = []
    totbbsh = 0
    oldct = {}
    allruns = np.zeros(nres)
    rdct = {}

    for at in dct:
        vol = outlivals[at]
        subtot = np.zeros(nres)
        subtot1 = np.zeros(nres)
        subtot2 = np.zeros(nres)
        #if dataset is not None:
        #    dataset[at][self.bmrID] = []
        A = np.array(list(dct[at].items()))
        totbbsh += len(A)
        #I = bbatns.index(at)
        w = refined_weights[at]
        shw = A[:,1] / w
        off = np.average(shw)
        rms0 = np.sqrt(np.average(shw**2))
        #if offdct is not None:
        shw -= offdct[at] #offset correction
        print ('using predetermined offset correction', at, offdct[at], offdct[at]*w)
        #shwl = list(shw)
        for i in range(len(A)):
            resi = int(A[i][0]) - mini #minimum value for resi is 0
            ashwi = abs(shw[i])
            if ashwi > cdfthr:
                oldct[(at,resi)] = ashwi
            tot[resi] += min(4.0, ashwi)**2
            for k in [-1,0,1]:
                if 0 <= (resi + k) < len(subtot):
                    subtot[resi+k]  += np.clip(shw[i]*w, -vol, vol) * wdct[at][0]
                    subtot1[resi+k] += np.clip(shw[i]*w, -vol, vol) * wdct[at][1]
                    subtot2[resi+k] += np.clip(shw[i]*w, -vol, vol) * wdct[at][2]
            totnum[resi]+=1
            if offdct is None:
                if 3 < i < len(A)-4:
                    vals = shw[i-4:i+5]
                    runstd = np.std(vals)
                    allruns[resi] += runstd
                    if not resi in rdct:
                        rdct[resi]={}
                    rdct[resi][at] = np.average(vals), np.sqrt(np.average(vals**2)), runstd
        dats[at] = shw
        stdw = np.std(shw)
        dAIC = np.log(rms0 / stdw)*len(A)-1
        print('rmsd:', at, stdw, off, dAIC)
        allrmsd.append(stdw)
        newtot += (subtot / 3.0)**2
        newtotsgn += subtot
        newtotsgn1 += subtot1
        newtotsgn2 += subtot2
    T0 = list(tot / totnum)
    cdfs = convChi2CDF(tot, totnum)
    Th = list(tot / totnum * 0.5)
    Ts = list(tot)
    Tn = list(totnum)
    tot3   = np.array([0,0] + Th) + np.array([0] + T0 + [0]) + np.array(Th + [0,0])
    tot3f  = np.array([0,0] + Ts) + np.array([0] + Ts + [0]) + np.array(Ts + [0,0])
    totn3f = np.array([0,0] + Tn) + np.array([0] + Tn + [0]) + np.array(Tn + [0,0])
    cdfs3 = convChi2CDF(tot3f[1:-1], totn3f[1:-1])
    #newrms = (newtot*3) / totn3f[1:-1]
    newcdfs = convChi2CDF(newtot*3, totn3f[1:-1])
    #avc = np.average(cdfs3[cdfs3<20.0])
    #numzs = len(cdfs3[cdfs3<20.0])
    #numzslt3 = len(cdfs3[cdfs3<cdfthr])
    #stdcp = np.std(cdfs3[cdfs3<20.0])
    atot = np.sqrt(tot3/2)[1:-1]
    aresids = np.array(resids)
    #if offdct is None:
    #    tr = (allruns / totnum)[4:-4]
    #    offdct = {}
    #    mintr = None
    #    minval = 999
    #    for j in range(len(tr)):
    #        if j+4 in rdct and len(rdct[j+4]) == len(dct): #all ats must be represented for this res
    #            if tr[j]<minval:
    #                minval = tr[j]
    #                mintr = j
    #    if mintr==None:
    #        return None #still not found
    #    print(len(tr), len(resids[4:-4]), len(atot), mintr+4, min(tr), tr[mintr])##,tr)
    #    for at in rdct[mintr+4]:
    #        roff,std0,stdc = rdct[mintr+4][at]
    #        dAIC = np.log(std0/stdc) * 9 - 1
    #        print('minimum running average', at, roff, dAIC)
    #        if dAIC > minAIC:
    #            print('using offset correction:', at, roff, dAIC)
    #            offdct[at] = roff
    #        else:
    #            print('rejecting offset correction due to low dAIC:', at, roff, dAIC)
    #            offdct[at] = 0.0
    #    return offdct #with the running offsets
    #if dataset is not None:
    #    csgns =  newtotsgn / totn3f[1:-1] * 10
    #    csgnsq = newtotsgn / np.sqrt(totn3f[1:-1]) * 10
    sferr3 = 0.0
    for at in dats:
        I = bbatns.index(at)
        ashw = np.abs(dats[at])
        Terr = np.linspace(0.0,5.0,26)
        ferr = np.array([np.sum(ashw>T) for T in Terr])*1.0/len(ashw)+0.000001
        sferr3 += ferr[15] #3.0std-fractile
    #aferr3 = sferr3 / len(dats)
    F = np.zeros(2)
    for at in dats:
        ashw = np.abs(dats[at])
        fners = np.sum(ashw>1.0) * 1.0 / len(ashw), sum(ashw>2.0) * 1.0 / len(ashw)
        F += fners
    #totnorm = sum(atot>1.5) * 1.0 / len(atot)
    outli0 = aresids[atot>1.5]
    outli1 = aresids[cdfs>cdfthr]
    outli3 = aresids[cdfs3>cdfthr]
    newoutli3 = aresids[newcdfs>cdfthr]
    finaloutli = [i+mini+1 for i in range(nres) if cdfs[i]>cdfthr or (cdfs3[i]>cdfthr and cdfs[i]>0.0 and totnum[i]>0)]
    print('outliers:', len(outli0), len(outli1), len(outli3), len(finaloutli), np.sum(totnum==0))##,finaloutli
    #Fa = F / len(dats)
    #fout = len(finaloutli) * 1.0/ nres
    print(len(oldct), mini, maxi, nres, aresids[totnum==0])
    #print('summary_stat: %5d %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %4d'%\
    #  (sum(self.moldba.counts.values()),average(allrmsd),Fa[0],Fa[1],fout,aferr3,totnorm,totbbsh))

    # now accumulate the validated data
    accdct = {k:[] for k in dct.keys()}
    numol = 0
    iatns = list(shiftdct.keys())
    iatns.sort()
    for i,at in iatns: #i is seq enumeration (starting from 0, but terminal always excluded)
        I = bbatns.index(at)
        w = refined_weights[at]
        ol = False
        if (i+1) in finaloutli or (at,i-mini) in oldct:
            ol=True
        if not ol:
            accdct[at].append(dct[at][i])
        else:
            numol += 1
        #if dataset is not None:
        #    dataset[at][self.bmrID].append(self.shiftdct[(i,at)]+[ol])
        #    vals = dataset[at][self.bmrID][-1]
        #    shout.write('%3d %1s %2s %7.3f %5s %6.3f\n'%(i+1,vals[1][2],at,vals[0],vals[1],dct[at][i]))
    sumrmsd = 0.0
    totsh = 0
    newoffdct = {}
    for at in accdct:
        #I = bbatns.index(at)
        w = refined_weights[at]
        vals = accdct[at]
        vals = np.array(vals) / w
        anum = len(vals)
        if anum == 0:
            newoffdct[at] = 0.0
        else:
            aoff = np.average(vals)
            astd0 = np.sqrt(np.average(np.array(vals)**2))
            astdc = np.std(vals)
            adAIC = np.log(astd0 / astdc) * anum - 1
            if adAIC < minAIC or anum < 4:
                print('rejecting offset correction due to low adAIC:', at, aoff, adAIC, anum)
                #if lacsoffs is not None and at in lacsoffs:
                #    print('LACS', lacsoffs[at], -aoff * w)
                astdc = astd0
                aoff = 0.0
                #shout.write('off %2s   0.0\n'%at)
            else:
                print('using offset correction:', at, aoff, adAIC, anum)
                #if lacsoffs is not None and at in lacsoffs:
                #    print('LACS', lacsoffs[at], -aoff * w)
                #shout.write('off %2s %7.3f\n'%(at,aoff*w))
            sumrmsd += astdc * anum
            totsh += anum
            newoffdct[at] = aoff
    #compl = calc_complexity(cdfs3,self.bmrID,thr=cdfthr)
    #fullrmsd = np.average(allrmsd)
    #ps = self.phys_state
    #ps6 = ps.strip("'")[:6]
    #fraczlt3 = numzslt3 * 1.0 / numzs
    if totsh == 0:
        avewrmsd,fracacc = 9.99, 0.0
    else:
        avewrmsd,fracacc = sumrmsd / totsh, totsh / (0.0+totsh+numol)
    #allsh = sum(totnum)
    #ratsh = allsh * 1.0 / numzs
    #print('finalstats %5s %8s %6s %7.4f %6.4f %6.4f %4d %4d %4d %7.3f %3d %3d %4d %6.4f %6.4f %7.3f %8.5f'\
    #  %(self.bmrID,label,ps6,avewrmsd,fullrmsd,fracacc,nres,totsh,numol,avc,numzs,numzslt3,allsh,fraczlt3,ratsh,stdcp,compl))##,
    if dataset is not None:
        #if len(newoffdct) > 6 or len(newoffdct) == 6 and 'HB' not in newoffdct:
        #    print('testoff:')
        #    for atn in ['CA','C','CB','N','H','HA']:
        #        print(newoffdct[atn])
        #    print()
        #fracol3 = len(outli3) * 1.0 / len(totnum > 0)
        #newfracol3 = len(newoutli3) * 1.0 / len(totnum>0)
        #if newfracol3 <= 0:
        #    lratf = 0.0
        #else:
        #    lratf = log(fracol3 / newfracol3)
        #print('fraccdfs3gt3 %7.4f %7.4f %6.3f'%(fracol3,newfracol3,lratf))
        return resids, \
               cdfs3, \
               newtotsgn  / np.sqrt(totn3f[1:-1]) * 8.0, \
               newtotsgn1 / np.sqrt(totn3f[1:-1]) * 8.0, \
               newtotsgn2 / np.sqrt(totn3f[1:-1]) * 8.0
    return avewrmsd, fracacc, newoffdct, cdfs3 #offsets from accepted stats

def visresults(dct, dataset=None, offdct=None, minAIC=999., cdfthr=6.0):
    bbatns=['C','CA','CB','HA','H','N','HB']
    refined_weights = {'C':0.1846, 'CA':0.1982, 'CB':0.1544, 'HA':0.02631, 'H':0.06708, 'N':0.4722, 'HB':0.02154}
    outlivals =       {'C':5.0000, 'CA':7.0000, 'CB':7.0000, 'HA':1.80000, 'H':2.30000, 'N':12.000, 'HB':1.80000}
    wdct = {
        'N':  [-0.0626, 0.0617,  0.2635],
        'C':  [ 0.2717, 0.2466,  0.0306],
        'CA': [ 0.2586, 0.2198,  0.0394],
        'CB': [-0.2635, 0.1830, -0.1877],
        'H':  [-0.3620, 1.3088,  0.3962],
        'HA': [-1.0732, 0.4440, -0.4673],
        'HB': [ 0.5743, 0.2262, -0.3388]
    }
    dats = {}
    maxi = max([max(dct[at].keys()) for at in dct])
    mini = min([min(dct[at].keys()) for at in dct])#is often 1
    nres = maxi-mini+1
    resids = range(mini+1,maxi+2)
    tot = np.zeros(nres)
    newtot = np.zeros(nres)
    newtotsgn = np.zeros(nres)
    newtotsgn1 = np.zeros(nres)
    newtotsgn2 = np.zeros(nres)
    totnum = np.zeros(nres)
    allrmsd = []
    totbbsh = 0
    oldct = {}
    allruns = np.zeros(nres)
    rdct = {}

    for at in dct:
        vol = outlivals[at]
        subtot = np.zeros(nres)
        subtot1 = np.zeros(nres)
        subtot2 = np.zeros(nres)
        if dataset is not None:
            dataset[at][self.bmrID] = []
        A = np.array(list(dct[at].items()))
        totbbsh += len(A)
        I = bbatns.index(at)
        w = refined_weights[at]
        shw = A[:,1] / w
        off = np.average(shw)
        rms0 = np.sqrt(np.average(shw**2))
        if offdct is not None:
            shw -= offdct[at] #offset correction
            print ('using predetermined offset correction', at, offdct[at], offdct[at]*w)
        shwl = list(shw)
        for i in range(len(A)):
            resi = int(A[i][0]) - mini #minimum value for resi is 0
            ashwi = abs(shw[i])
            if ashwi > cdfthr:
                oldct[(at,resi)] = ashwi
            tot[resi] += min(4.0, ashwi)**2
            for k in [-1,0,1]:
                if 0 <= (resi + k) < len(subtot):
                    subtot[resi+k]  += np.clip(shw[i]*w, -vol, vol) * wdct[at][0]
                    subtot1[resi+k] += np.clip(shw[i]*w, -vol, vol) * wdct[at][1]
                    subtot2[resi+k] += np.clip(shw[i]*w, -vol, vol) * wdct[at][2]
            totnum[resi]+=1
            if offdct is None:
                if 3 < i < len(A)-4:
                    vals = shw[i-4:i+5]
                    runstd = np.std(vals)
                    allruns[resi] += runstd
                    if not resi in rdct:
                        rdct[resi]={}
                    rdct[resi][at] = np.average(vals), np.sqrt(np.average(vals**2)), runstd
        dats[at] = shw
        stdw = np.std(shw)
        dAIC = np.log(rms0 / stdw)*len(A)-1
        print('rmsd:', at, stdw, off, dAIC)
        allrmsd.append(stdw)
        newtot += (subtot / 3.0)**2
        newtotsgn += subtot
        newtotsgn1 += subtot1
        newtotsgn2 += subtot2
    T0 = list(tot / totnum)
    cdfs = convChi2CDF(tot, totnum)
    Th = list(tot / totnum * 0.5)
    Ts = list(tot)
    Tn = list(totnum)
    tot3   = np.array([0,0] + Th) + np.array([0] + T0 + [0]) + np.array(Th + [0,0])
    tot3f  = np.array([0,0] + Ts) + np.array([0] + Ts + [0]) + np.array(Ts + [0,0])
    totn3f = np.array([0,0] + Tn) + np.array([0] + Tn + [0]) + np.array(Tn + [0,0])
    cdfs3 = convChi2CDF(tot3f[1:-1], totn3f[1:-1])
    newrms = (newtot*3) / totn3f[1:-1]
    newcdfs = convChi2CDF(newtot*3, totn3f[1:-1])
    avc = np.average(cdfs3[cdfs3<20.0])
    numzs = len(cdfs3[cdfs3<20.0])
    numzslt3 = len(cdfs3[cdfs3<cdfthr])
    stdcp = np.std(cdfs3[cdfs3<20.0])
    atot = np.sqrt(tot3/2)[1:-1]
    aresids = np.array(resids)
    if offdct is None:
        tr = (allruns / totnum)[4:-4]
        offdct = {}
        mintr = None
        minval = 999
        for j in range(len(tr)):
            if j+4 in rdct and len(rdct[j+4]) == len(dct): #all ats must be represented for this res
                if tr[j]<minval:
                    minval = tr[j]
                    mintr = j
        if mintr==None:
            return None #still not found
        print(len(tr), len(resids[4:-4]), len(atot), mintr+4, min(tr), tr[mintr])##,tr)
        for at in rdct[mintr+4]:
            roff,std0,stdc = rdct[mintr+4][at]
            dAIC = np.log(std0/stdc) * 9 - 1
            print('minimum running average', at, roff, dAIC)
            if dAIC > minAIC:
                print('using offset correction:', at, roff, dAIC)
                offdct[at] = roff
            else:
                print('rejecting offset correction due to low dAIC:', at, roff, dAIC)
                offdct[at] = 0.0
        return offdct #with the running offsets

def savedata(cdfs3, bmrID, seq, mini, stID, condID, assemID, entityID):
    out=open(f'zscores{bmrID}_{stID}_{condID}_{assemID}_{entityID}.txt','w')
    for i,x in enumerate(cdfs3):
      if x < 99: #not nan
        I = i + mini
        aai=seq[I]
        out.write('%s %3d %6.3f\n'%(aai,I+1,x))
    out.close()

def main():
    entry = trizod.BmrbEntry(args.ID, args.bmrb_dir)
    print("Parsed the following BMRB entry:")
    print(entry)
    print()
    peptide_shifts = entry.get_peptide_shifts()
    for (stID, condID, assemID, entityID), shifts in peptide_shifts.items():
        # get polymer sequence and chemical backbone shifts
        seq = entry.entities[entityID].seq
        bbshifts = trizod.get_valid_bbshifts(shifts, seq)
        if bbshifts is None:
            logging.info(f'skipping shifts for {(stID, condID, assemID, entityID)} due to previous error')
            continue

        # use POTENCI to predict shifts
        ion = entry.conditions[condID].ionic_strength
        pH = entry.conditions[condID].pH
        temperature = entry.conditions[condID].temperature
        if ion is None:
            logging.warning(f'No information on ionic strength for sample condition {condID}, assuming 0.1 M')
            ion = 0.1
        if pH is None:
            logging.warning(f'No information on pH for sample condition {condID}, assuming 7.0')
            pH = 7.0
        if temperature is None:
            logging.warning(f'No information on temperature for sample condition {condID}, assuming 298 K')
            temperature = 298.
        usephcor = pH < 6.99 or pH > 7.01
        try:
            predshiftdct = potenci.getpredshifts(seq,temperature,pH,ion,usephcor,pkacsvfile=None)
        except:
            logging.error(f"POTENCI failed for {(stID, condID, assemID, entityID)}")
            continue
        
        # compare predicted to actual shifts
        cmpdct, shiftdct = comp2pred(predshiftdct,bbshifts,seq)
        totbbsh = sum([len(cmpdct[at].keys()) for at in cmpdct])
        logging.info(f"total number of backbone shifts: {totbbsh}")
        offr2 = visresults(cmpdct, minAIC=6.0)
        print('####')
        offr = get_offset_correction(cmpdct, minAIC=6.0)
        print(offr2)
        print('####')
        print(offr)
        if offr is None:
            logging.warning(f'no running offset could be estimated for {(stID, condID, assemID, entityID)}')
            bbatns = ['C','CA','CB','HA','H','N','HB']
            off0 = dict(zip(bbatns,[0.0 for _ in bbatns]))
            armsdc = 999.9
            frac = 0.0
        else:
            atns = offr.keys()
            off0 = dict(zip(atns,[0.0 for _ in atns]))
            armsdc,frac,noffc,cdfs3c = results_w_offset(cmpdct, shiftdct, offdct=offr, minAIC=6.0)
        armsd0,fra0,noff0,cdfs30 = results_w_offset(cmpdct, shiftdct, offdct=off0, minAIC=6.0)
        usefirst = (armsd0 / (0.01 + fra0)) < (armsdc / (0.01 + frac))
        av0 = np.average(cdfs30[cdfs30 < 20.0]) #to avoid nan
        if offr is not None:
            avc = np.average(cdfs3c[cdfs3c < 20.0])
            orusefirst = av0 < avc
            if usefirst != orusefirst:
                print('WARNING: hard decission',usefirst,orusefirst)
            print('decide',orusefirst,armsd0,fra0,av0,armsdc,frac,avc)
        else:
            orusefirst=True
        if orusefirst:
            resids,cdfs3,pc1ws,pc2ws,pc3ws = results_w_offset(cmpdct, shiftdct, dataset=True, offdct=noff0, minAIC=6.0)
        else:
            resids,cdfs3,pc1ws,pc2ws,pc3ws = results_w_offset(cmpdct, shiftdct, dataset=True, offdct=noffc, minAIC=6.0)
        
        mini = min([min(cmpdct[at].keys()) for at in cmpdct])
        savedata(cdfs3, args.ID, seq, mini, stID, condID, assemID, entityID)

if __name__ == '__main__':
    args = parse_args()
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level) #filename='example.log', encoding='utf-8'
    main()