#!/bin/bash python3

# version of the POTENCI script, adapted by haak@rostlab.org
# original by fmulder@chem.au.dk
# original taken from https://github.com/protein-nmr/POTENCI on 03.05.2023, commit 17dd2e6f3733c702323894697238c87e6723f934
# original version (filename): pytenci1_3.py

import sys
import string
from scipy.special import erfc
from scipy.optimize import curve_fit
from scipy import sparse
import numpy as np
import pandas as pd
import os
import csv
from trizod.potenci.constants import R, e, a, b, cutoff, ncycles, pK0
import logging
import pkgutil
from io import StringIO

outer_matrices = []
alltuples_ = []
for smallN in range(0,6):
    alltuples = np.array([[int(c) for c in np.binary_repr(i, smallN)] for i in range(2 ** (smallN))])
    outerm = np.array([np.outer(c,c) for c in alltuples])
    outer_matrices.append(outerm)
    alltuples_.append(alltuples)

def smallmatrixlimits(ires, cutoff, len):
    ileft = max(1, ires - cutoff)
    iright = min(ileft + 2 * cutoff, len)
    if iright == len:
        ileft = max(1, iright - 2 * cutoff)
    return (ileft, iright)

def smallmatrixpos(ires, cutoff, len):
    resi = cutoff + 1
    if ires < cutoff + 1:
        resi = ires
    if ires > len - cutoff:
        resi = min(len, 2 * cutoff + 1) - (len - ires)
    return resi

def fun(pH, pK, nH):
    #return (10 ** ( nH*(pK - pH) ) ) / (1. + (10 **( nH*(pK - pH) ) ) )
    return 1. - 1. / ((10 ** ( nH*(pK - pH) ) ) + 1.) # identical

def log_fun(pH, pK, nH):
    return -np.log10(1 + 10**(nH*(pH - pK)))

def W(r,Ion=0.1):
    k = np.sqrt(Ion) / 3.08 #Ion=0.1 is default
    x = k.astype(np.float64) * r.astype(np.float64) / np.sqrt(6)
    i1 = 332.286 * np.sqrt(6 / np.pi)
    i2_3 = erfc(x)
    i2_2 = np.sqrt(np.pi) * x

    i3 = (e * r)
    i4 = np.exp(((x ** 2) - np.log(i3))) # always equal to np.exp(x ** 2) / (e * r), but intermediates are smaller
    i4 = np.nan_to_num(i4) # to convert inf values to the largest possible value
    return i1 * ((1 / i3) - np.nan_to_num(i4 * i2_2 * i2_3))

def w2logp(x,T=293.15):
    return x * 4181.2 / (R * T * np.log(10))

def calc_pkas_from_seq(seq=None, T=293.15, Ion=0.1):
    #pH range
    pHs = np.arange(1.99, 10.01, 0.15)

    pos = np.array([i for i in range(len(seq)) if seq[i] in pK0.keys()])
    N = pos.shape[0]
    I = np.diag(np.ones(N))
    sites = ''.join([seq[i] for i in pos])
    neg = np.array([i for i in range(len(sites)) if sites[i] in 'DEYc'])
    l = np.array([abs(pos - pos[i]) for i in range(N)])
    d = a + np.sqrt(l) * b

    tmp = W(d,Ion)
    tmp[I == 1] = 0

    ww = w2logp(tmp,T) / 2

    chargesempty = np.zeros(pos.shape[0])
    if len(neg): chargesempty[neg] = -1

    pK0s = np.array([pK0[c] for c in sites])
    nH0s = np.array([0.9 for c in sites])

    titration = np.zeros((N,len(pHs)))

    smallN = min(2 * cutoff + 1, len(pos))
    alltuples = alltuples_[smallN]
    outerm = outer_matrices[smallN]
    gmatrix = [np.zeros((smallN, smallN)) for _ in range(len(pHs))]

    #perform iterative fitting.........................
    for icycle in range(ncycles):
        ##print (icycle)

        if icycle == 0:
            fractionhold = np.array([[fun(pHs[p], pK0s[i], nH0s[i]) for i in range(N)] for p in range(len(pHs))])
        else:
            fractionhold = titration.transpose()

        for ires in range(1, N+1):
            (ileft,iright) = smallmatrixlimits(ires, cutoff, N)
            resi = smallmatrixpos(ires, cutoff, N)
            fraction = fractionhold.copy()
            fraction[:,ileft - 1:iright] = 0
            charges = fraction + chargesempty
            ww0 = 2 * (ww * np.expand_dims(charges, axis=1)).sum(axis=-1)
            ww0 = (np.expand_dims(ww0, 1) * I) # array of diagonal matrices
            gmatrixfull = (ww + ww0 + np.expand_dims(pHs,(1,2)) * I - np.diag(pK0s))
            gmatrix = gmatrixfull[:, ileft - 1 : iright, ileft - 1 : iright]

            E = (10 ** -(np.expand_dims(gmatrix, axis=1) * outerm).sum(axis=(2,3)))#.sum(axis=-1)
            E_all = E.sum(axis=-1)
            E_sel = E[:,(alltuples[:,resi-1] == 1)].sum(axis=-1)
            titration[ires-1] = E_sel/E_all
        sol = np.array([curve_fit(fun, pHs, titration[p], [pK0s[p], nH0s[p]], maxfev=5000)[0] for p in range(len(pK0s))])
        (pKs, nHs) = sol.transpose()

    dct={}
    for p,i in enumerate(pos):
        dct[i-1]=(pKs[p],nHs[p],seq[i])

    return dct


##--------------- POTENCI core code and data tables from here -----------------

_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

#AAstandard='ACDEFGHIKLMNPQRSTVY'
AAstandard='ACDEFGHIKLMNPQRSTVWY'


def _load_csv(filename):
    """Load a CSV file from the data directory."""
    filepath = os.path.join(_DATA_DIR, filename)
    with open(filepath) as f:
        return list(csv.DictReader(f))


def initcorcents():
    rows = _load_csv('centshifts.csv')
    aas = ['C', 'CA', 'CB', 'N', 'H', 'HA', 'HB']
    dct = {}
    for row in rows:
        aa = row['aa']
        dct[aa] = {}
        for atn in aas:
            val = row[atn]
            dct[aa][atn] = None if val == 'None' else float(val)
    return dct


def initcorneis():
    dct = {}
    # Load neighbor corrections
    for row in _load_csv('neicorrs.csv'):
        atn = row['atn']
        aa = row['aa']
        if aa not in dct:
            dct[aa] = {}
        dct[aa][atn] = [float(row[f'c{j}']) for j in range(4)]
    # Load terminal corrections
    for row in _load_csv('termcorrs.csv'):
        atn = row['atn']
        term = row['term']
        val = float(row['value'])
        if term not in dct:
            dct[term] = {}
        if term == 'n':
            dct['n'][atn] = [None, None, None, val]
        elif term == 'c':
            dct['c'][atn] = [val, None, None, None]
    return dct


def gettempkoeff():
    rows = _load_csv('tempcoeffs.csv')
    headers = [k for k in rows[0].keys() if k != 'aa']
    dct = {}
    for atn in headers:
        dct[atn] = {}
    for row in rows:
        aa = row['aa']
        for atn in headers:
            dct[atn][aa] = float(row[atn])
    return dct


tablephshifts='''
D (pKa 3.86)
D H  8.55 8.38 -0.17 0.02 -0.03
D HA 4.78 4.61 -0.17 0.01 -0.01
D HB 2.93 2.70 -0.23
D CA 52.9 54.3 1.4 0.0 0.1
D CB 38.0 41.1 3.0
D CG 177.1 180.3 3.2
D C  175.8 176.9 1.1 -0.2 0.4
D N  118.7 120.2 1.5  0.3 0.1
D Np na na 0.1
E (pKa 4.34)
E H  8.45 8.57  0.12 0.00 0.02
E HA 4.39 4.29 -0.10 0.01 0.00
E HB 2.08 2.02 -0.06
E HG 2.49 2.27 -0.22
E CA 56.0 56.9 1.0 0.0 0.0
E CB 28.5 30.0 1.5
E CG 32.7 36.1 3.5
E CD 179.7 183.8 4.1
E C  176.5 177.0 0.6  0.1 0.1
E N  119.9 120.9 1.0  0.2 0.1
E Np na na 0.1
H (pKa 6.45)
H H  8.55 8.35 -0.2  -0.01  0.0
H HA 4.75 4.59 -0.2  -0.01 -0.06
H HB 3.25 3.08 -0.17
H HD2 7.30 6.97 -0.33
H HE1 8.60 7.68 -0.92
H CA 55.1 56.7 1.6 -0.1 0.1
H CB 28.9 31.3 2.4
H CG 131.0 135.3 4.2
H CD2 120.3 120.0 -0.3
H CE1 136.6 139.2 2.6
H C 174.8 176.2 1.5  0.0 0.6
H N 117.9 119.7 1.8  0.3 0.5
H Np na na 0.5
H ND1 175.8 231.3 56
H NE2 173.1 181.1 8
C (pKa 8.49)
C H 8.49 8.49 0.0
C HA 4.56 4.28 -0.28 -0.01 -0.01
C HB 2.97 2.88 -0.09
C CA 58.5 60.6 2.1 0.0 0.1
C CB 28.0 29.7 1.7
C C 175.0 176.9 1.9 -0.4 0.5
C N 118.7 122.2 3.6  0.4 0.6
C Np na na 0.6
Y (pKa 9.76)
Y H  8.16 8.16 0.0
Y HA 4.55 4.49 -0.06
Y HB 3.02 2.94 -0.08
Y HD 7.14 6.97 -0.17
Y HE 6.85 6.57 -0.28
Y CA 58.0 58.2 0.3
Y CB 38.6 38.7 0.1
Y CG 130.5 123.8 -6.7
Y CD 133.3 133.2 -0.1
Y CE 118.4 121.7 3.3
Y CZ 157.0 167.4 10.4
Y C 176.3 176.7 0.4
Y N 120.1 120.7 0.6
K (pKa 10.34)
K H  8.4  8.4  0.0
K HA 4.34 4.30 -0.04
K HB 1.82 1.78 -0.04
K HG 1.44 1.36 -0.08
K HD 1.68 1.44 -0.24
K HE 3.00 2.60 -0.40
K CA 56.4 56.9 0.4
K CB 32.8 33.2 0.3
K CG 24.7 25.0 0.4
K CD 28.9 33.9 5.0
K CE 42.1 43.1 1.0
K C 177.0 177.5 0.5
K N 121.0 121.7 0.7
K Np na na 0.1
R (pKa 13.9)
R H  7.81 7.81 0.0
R HA 3.26 3.19 -0.07
R HB 1.60 1.55 0.05
R HG 1.60 1.55 0.05
R HD 3.19 3.00 -0.19
R CA 58.4 58.6 0.2
R CB 34.4 35.2 0.9
R CG 27.2 28.1 1.0
R CD 43.8 44.3 0.5
R CZ 159.6 163.5 4.0
R C 185.8 186.1 0.2
R N 122.4 122.8 0.4
R NE 85.6 91.5 5.9
R NG 71.2 93.2 22'''


def initcorrcomb():
    dct = {}
    for row in _load_csv('combdevs.csv'):
        atn = row['atn']
        if atn not in dct:
            dct[atn] = {}
        segment = row['segment']
        key = (int(row['neipos']), row['centgroup'], row['neigroup'])
        dct[atn][segment] = key, float(row['value'])
    return dct


TEMPCORRS=gettempkoeff()
CENTSHIFTS=initcorcents()
NEICORRS =initcorneis()
COMBCORRS=initcorrcomb()


def predPentShift(pent,atn):
    aac=pent[2]
    sh=CENTSHIFTS[aac][atn]
    allneipos=[2,1,-1,-2]
    for i in range(4):
        aai=pent[2+allneipos[i]]
        if aai in NEICORRS:
            corr=NEICORRS[aai][atn][i]
            sh+=corr
    groups=['G','P','FYW','LIVMCA','KR','DE']##,'NQSTHncX']
    labels='GPra+-p' #(Gly,Pro,Arom,Aliph,pos,neg,polar)
    grstr=''
    for i in range(5):
        aai=pent[i]
        found=False
        for j,gr in enumerate(groups):
            if aai in gr:
                grstr+=labels[j]
                found=True
                break
        if not found:grstr+='p'#polar
    centgr=grstr[2]
    for segm in COMBCORRS[atn]:
        key,combval=COMBCORRS[atn][segm]
        neipos,centgroup,neigroup=key#(k,l,m)
        if centgroup==centgr and grstr[2+neipos]==neigroup:
            if (centgr,neigroup)!=('p','p') or pent[2] in 'ST':
                #pp comb only used when center is Ser or Thr!
                sh+=combval
    return sh

def gettempcorr(aai,atn,tempdct,temp):
    return tempdct[atn][aai]/1000*(temp-298)

def _parse_phshift_val(s):
    """Parse a pH shift value, treating 'na' as None."""
    if s == 'na':
        return None
    return float(s)

def get_phshifts():
    datc=tablephshifts.split('\n')
    buf=[lin.split() for lin in datc]
    dct={}
    for lin in buf:
        if len(lin)>3:
            resn=lin[0]
            atn=lin[1]
            sh0=_parse_phshift_val(lin[2])
            sh1=_parse_phshift_val(lin[3])
            shd=_parse_phshift_val(lin[4])
            if not resn in dct:dct[resn]={}
            dct[resn][atn]=shd
            if len(lin)>6:#neighbor data
                for n in range(2):
                    shdn=float(lin[5+n])
                    nresn=resn+'ps'[n]
                    if not nresn in dct:dct[nresn]={}
                    dct[nresn][atn]=shdn
    return dct

def initfilcsv(filename):
    file=open(filename,'r')
    buffer=file.readlines()
    file.close()
    for i in range(len(buffer)):
        buffer[i]=buffer[i][:-1].split(',')
    return buffer

def write_csv_pkaoutput(pkadct,seq,temperature,ion):
        seq=seq[:min(150,len(seq))]
        name='outpepKalc_%s_T%6.2f_I%4.2f.csv'%(seq,temperature,ion)
        out=open(name,'w')
        out.write('Site,pKa value,pKa shift,Hill coefficient\n')
        for i in pkadct:
            pKa,nH,resi=pkadct[i]
            reskey=resi+str(i+1)
            diff=pKa-pK0[resi]
            out.write('%s,%5.3f,%5.3f,%5.3f\n'%(reskey,pKa,diff,nH))
        out.close()

def read_csv_pkaoutput(seq,temperature,ion,name=None):
    seq=seq[:min(150,len(seq))]
    logging.getLogger('trizod.potenci').debug(f'reading csv {name}')
    if name==None:name='outpepKalc_%s_T%6.2f_I%4.2f.csv'%(seq,temperature,ion)
    try:out=open(name,'r')
    except IOError:return None
    buf=initfilcsv(name)
    for lnum,data in enumerate(buf):
        if len(data)>0 and data[0]=='Site':break
    pkadct={}
    for data in buf[lnum+1:]:
        reskey,pKa,diff,nH=data
        i=int(reskey[1:])-1
        resi=reskey[0]
        pKaval=float(pKa)
        nHval=float(nH)
        pkadct[i]=pKaval,nHval,resi
    return pkadct

def getphcorrs(seq,temperature,pH,ion,pkacsvfilename=None):
    bbatns=['C','CA','CB','HA','H','N','HB']
    dct=get_phshifts()
    Ion=max(0.0001,ion)
    if pkacsvfilename == False:
        pkadct=None
    else:
        pkadct=read_csv_pkaoutput(seq,temperature,ion,pkacsvfilename)
    if pkadct==None:
        pkadct=calc_pkas_from_seq('n'+seq+'c',temperature,Ion)
        if pkacsvfilename != False:
            write_csv_pkaoutput(pkadct,seq,temperature,ion)
    outdct={}
    for i in pkadct:
        logging.getLogger('trizod.potenci').debug('pkares: %6.3f %6.3f %1s'%pkadct[i] + str(i))
        pKa,nH,resi=pkadct[i]
        frac =fun(pH,pKa,nH)
        frac7=fun(7.0,pK0[resi],nH)
        if resi in 'nc':jump=0.0#so far
        else:
            for atn in bbatns:
                if not atn in outdct:outdct[atn]={}
                logging.getLogger('trizod.potenci').debug(f'data: {atn}, {pKa}, {nH}, {resi}, {i}, {atn}, {pH}')
                dctresi=dct[resi]
                try:
                    delta=dctresi[atn]
                    # delta = PHSHIFTS.loc[(resi,atn), 'shd']
                    jump =frac *delta
                    jump7=frac7*delta
                    key=(resi,atn)
                except KeyError:
                    ##if not (resi in 'RKCY' and atn=='H') and not (resi == 'R' and atn=='N'):
                    logging.getLogger('trizod.potenci').warning(f'no key: {resi}, {i}, {atn}')
                    delta=999;jump=999;jump7=999
                if delta<99:
                    jumpdelta=jump-jump7
                    if not i in outdct[atn]:outdct[atn][i]=[resi,jumpdelta]
                    else:
                        outdct[atn][i][0]=resi
                        outdct[atn][i][1]+=jumpdelta
                    logging.getLogger('trizod.potenci').debug('%3s %5.2f %6.4f %s %3d %5s %8.5f %8.5f %4.2f'%(atn,pKa,nH,resi,i,atn,jump,jump7,pH))
                    if resi+'p' in dct and atn in dct[resi+'p']:
                    # if (resi+'p', atn) in PHSHIFTS.index:
                        for n in range(2):
                            ni=i+2*n-1
                            ##if ni is somewhere in seq...
                            nresi=resi+'ps'[n]
                            ndelta=dct[nresi][atn]
                            # ndelta = PHSHIFTS.loc[(nresi,atn), 'shd']
                            jump =frac *ndelta
                            jump7=frac7*ndelta
                            jumpdelta=jump-jump7
                            if not ni in outdct[atn]:outdct[atn][ni]=[None,jumpdelta]
                            else:outdct[atn][ni][1]+=jumpdelta
    return outdct

def getpredshifts(seq,temperature,pH,ion,usephcor=True,pkacsvfile=None,identifier=''):
    tempdct=gettempkoeff()
    bbatns = ['C','CA','CB','HA','H','N','HB']
    if usephcor:
        phcorrs=getphcorrs(seq,temperature,pH,ion,pkacsvfile)
    else:
        phcorrs={}
    shiftdct={}
    for i in range(1,len(seq)-1):
        if seq[i] in AAstandard:#else: do nothing
            res=str(i+1)
            trip=seq[i-1]+seq[i]+seq[i+1]
            phcorr=None
            shiftdct[(i+1,seq[i])]={}
            for at in bbatns:
                if not (trip[1],at) in [('G','CB'),('G','HB'),('P','H')]:
                    if i == 1:
                        pent = 'n'      + trip + seq[i+2]
                    elif i==len(seq)-2:
                        pent = seq[i-2] + trip + 'c'
                    else:
                        pent = seq[i-2] + trip + seq[i+2]
                    shp=predPentShift(pent,at)
                    if shp!=None:
                        if at!='HB':shp+=gettempcorr(trip[1],at,tempdct,temperature)
                        if at in phcorrs and i in phcorrs[at]:
                            phdata=phcorrs[at][i]
                            resi=phdata[0]
                            ##assert resi==seq[i]
                            if seq[i] in 'CDEHRKY' and resi != seq[i]:
                                logging.getLogger('trizod.potenci').warning(f'residue mismatch: {resi},{seq[i]},{i},{phdata},{at}')
                            phcorr=phdata[1]
                            if abs(phcorr)<9.9:
                                shp-=phcorr
                        shiftdct[(i+1,seq[i])][at]=shp
                        logging.getLogger('trizod.potenci').debug('predictedshift: %5s %3d %1s %2s %8.4f'%(identifier,i,seq[i],at,shp) + ' ' + str(phcorr))
    return shiftdct
