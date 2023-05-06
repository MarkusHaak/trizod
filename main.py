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
    refined_weights = {'C':0.1846, 'CA':0.1982, 'CB':0.1544, 'HA':0.02631, 'H':0.06708, 'N':0.4722, 'HB':0.02154}
    maxi = max([max(dct[at].keys()) for at in dct])
    mini = min([min(dct[at].keys()) for at in dct])#is often 1
    nres = maxi - mini + 1
    totnum = np.zeros(nres)
    allruns = np.zeros(nres)
    rdct = {}

    for at in dct:
        A = np.array(list(dct[at].items()))
        w = refined_weights[at]
        shw = A[:,1] / w
        for i in range(len(A)):
            resi = int(A[i][0]) - mini #minimum value for resi is 0
            totnum[resi]+=1
            if 3 < i < len(A)-4:
                vals = shw[i-4:i+5]
                runstd = np.std(vals)
                allruns[resi] += runstd
                if not resi in rdct:
                    rdct[resi]={}
                rdct[resi][at] = np.average(vals), np.sqrt(np.average(vals**2)), runstd
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
    refined_weights = {'C':0.1846, 'CA':0.1982, 'CB':0.1544, 'HA':0.02631, 'H':0.06708, 'N':0.4722, 'HB':0.02154}
    maxi = max([max(dct[at].keys()) for at in dct])
    mini = min([min(dct[at].keys()) for at in dct])#is often 1
    nres = maxi-mini+1
    tot = np.zeros(nres)
    totnum = np.zeros(nres)
    oldct = set() # was {}

    for at in dct:
        A = np.array(list(dct[at].items()))
        w = refined_weights[at]
        shw = A[:,1] / w
        shw -= offdct[at] #offset correction
        print ('using predetermined offset correction', at, offdct[at], offdct[at] * w)
        for i in range(len(A)):
            resi = int(A[i][0]) - mini #minimum value for resi is 0
            ashwi = abs(shw[i])
            if ashwi > cdfthr:
                oldct.add((at,resi)) # was oldct[(at,resi)] = ashwi
            tot[resi] += min(4.0, ashwi)**2
            totnum[resi]+=1
    cdfs = convChi2CDF(tot, totnum)
    tot3f  = np.pad(tot, 1)[2:]    + tot    + np.pad(tot, 1)[:-2]
    totn3f = np.pad(totnum, 1)[2:] + totnum + np.pad(totnum, 1)[:-2]
    cdfs3 = convChi2CDF(tot3f, totn3f)
    if dataset is not None:
        return cdfs3

    finaloutli = [i+mini+1 for i in range(nres) if cdfs[i]>cdfthr or (cdfs3[i]>cdfthr and cdfs[i]>0.0 and totnum[i]>0)]
    print('outliers:', len(finaloutli), np.sum(totnum==0), finaloutli)
    print(len(oldct), mini, maxi, nres)

    # now accumulate the validated data
    accdct = {k:[] for k in dct.keys()}
    numol = 0
    iatns = list(shiftdct.keys())
    iatns.sort()
    for i,at in iatns: #i is seq enumeration (starting from 0, but terminal always excluded)
        w = refined_weights[at]
        ol = False
        if (i+1) in finaloutli or (at,i-mini) in oldct:
            ol=True
        if not ol:
            accdct[at].append(dct[at][i])
        else:
            numol += 1
    sumrmsd = 0.0
    totsh = 0
    newoffdct = {}
    for at in accdct:
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
                astdc = astd0
                aoff = 0.0
            else:
                print('using offset correction:', at, aoff, adAIC, anum)
            sumrmsd += astdc * anum
            totsh += anum
            newoffdct[at] = aoff
    if totsh == 0:
        avewrmsd,fracacc = 9.99, 0.0
    else:
        avewrmsd,fracacc = sumrmsd / totsh, totsh / (0.0+totsh+numol)
    return avewrmsd, fracacc, newoffdct, cdfs3 #offsets from accepted stats

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
        offr = get_offset_correction(cmpdct, minAIC=6.0)
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
            cdfs3 = results_w_offset(cmpdct, shiftdct, dataset=True, offdct=noff0, minAIC=6.0)
        else:
            cdfs3 = results_w_offset(cmpdct, shiftdct, dataset=True, offdct=noffc, minAIC=6.0)
        
        mini = min([min(cmpdct[at].keys()) for at in cmpdct])
        savedata(cdfs3, args.ID, seq, mini, stID, condID, assemID, entityID)

if __name__ == '__main__':
    args = parse_args()
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level) #filename='example.log', encoding='utf-8'
    main()