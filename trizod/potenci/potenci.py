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
##from matplotlib import pyplot as pl
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

#AAstandard='ACDEFGHIKLMNPQRSTVY'
AAstandard='ACDEFGHIKLMNPQRSTVWY'

tablecent='''aa C CA CB N H HA HB
A 177.44069  52.53002  19.21113 125.40155   8.20964   4.25629   1.31544
C 174.33917  58.48976  28.06269 120.71212   8.29429   4.44261   2.85425
D 176.02114  54.23920  41.18408 121.75726   8.28460   4.54836   2.60054
E 176.19215  56.50755  30.30204 122.31578   8.35949   4.22124   1.92383
F 175.42280  57.64849  39.55984 121.30500   8.10906   4.57507   3.00036
G 173.83294  45.23929  None     110.09074   8.32746   3.91016   None
H 175.00142  56.20256  30.60335 120.69141   8.27133   4.55872   3.03080
I 175.88231  61.04925  38.68742 122.37586   8.06407   4.10574   1.78617
K 176.22644  56.29413  33.02478 122.71282   8.24902   4.25873   1.71982
L 177.06101  55.17464  42.29215 123.48611   8.14330   4.28545   1.54067
M 175.90708  55.50643  32.83806 121.54592   8.24848   4.41483   1.97585
N 174.94152  53.22822  38.87465 119.92746   8.37189   4.64308   2.72756
P 176.67709  63.05232  32.03750 137.40612   None      4.36183   2.03318
Q 175.63494  55.79861  29.44174 121.49225   8.30042   4.28006   1.97653
R 175.92194  56.06785  30.81298 122.40365   8.26453   4.28372   1.73437
S 174.31005  58.36048  63.82367 117.11419   8.25730   4.40101   3.80956
T 174.27772  61.86928  69.80612 115.48126   8.11378   4.28923   4.15465
V 175.80621  62.20156  32.77934 121.71912   8.06572   4.05841   1.99302
W 175.92744  57.23836  29.56502 122.10991   7.97816   4.61061   3.18540
Y 175.49651  57.82427  38.76184 121.43652   8.05749   4.51123   2.91782'''

def initcorcents():
    datc=tablecent.split('\n')
    aas=datc[0].split()[1:]
    dct={}
    for i in range(20):
        vals=datc[1+i].split()
        aai=vals[0]
        dct[aai]={}
        for j in range(7):
            atnj=aas[j]
            dct[aai][atnj]=eval(vals[1+j])
    return dct
        

tablenei='''C A  0.06131 -0.04544  0.14646  0.01305
 C C  0.04502  0.12592 -0.03407 -0.02654
 C D  0.08180 -0.08589  0.22948  0.10934
 C E  0.05388  0.22264  0.06962  0.01929
 C F -0.06286 -0.22396 -0.34442  0.00950
 C G  0.12772  0.72041  0.16048  0.01324
 C H -0.00628 -0.03355  0.13309 -0.03906
 C I -0.11709  0.06591 -0.06361 -0.03628
 C K  0.03368  0.15830  0.04518 -0.01576
 C L -0.03877  0.11608  0.02535  0.01976
 C M  0.04611  0.25233 -0.00747 -0.01624
 C N  0.07068 -0.06118  0.10077  0.05547
 C P -0.36018 -1.90872  0.16158 -0.05286
 C Q  0.10861  0.19878  0.01596 -0.01757
 C R  0.01933  0.13237  0.03606 -0.02468
 C S  0.09888  0.28691  0.07601  0.01379
 C T  0.05658  0.41659 -0.01103 -0.00114
 C V -0.11591  0.09565 -0.03355 -0.03368
 C W -0.01954 -0.19134 -0.37965  0.01582
 C Y -0.08380 -0.24519 -0.32700 -0.00577
CA A  0.03588  0.03480 -0.00468 -0.00920
CA C  0.02749  0.15742  0.14376  0.03681
CA D -0.00751  0.12494  0.17354  0.14157
CA E  0.00985  0.13936  0.03289 -0.00702
CA F  0.01122  0.03732 -0.19586 -0.00377
CA G -0.00885  0.23403 -0.03184 -0.01144
CA H -0.02102  0.04621  0.03122 -0.02826
CA I -0.00656  0.05965 -0.10588 -0.04372
CA K  0.01817  0.11216 -0.00341 -0.02950
CA L  0.04507  0.07829 -0.03526  0.00858
CA M  0.07553  0.18840  0.04987 -0.01749
CA N -0.00649  0.11842  0.18729  0.06401
CA P -0.27536 -2.02189  0.01327 -0.08732
CA Q  0.06365  0.15281  0.04575 -0.01356
CA R  0.04338  0.11783  0.00345 -0.02873
CA S  0.02867  0.07846  0.09443  0.02061
CA T -0.01625  0.10626  0.03880 -0.00126
CA V -0.04935  0.04248 -0.10195 -0.03778
CA W  0.00434  0.16188 -0.08742  0.03983
CA Y  0.02782  0.02846 -0.24750  0.00759
CB A -0.00953  0.05704 -0.04838  0.00755
CB C -0.00164  0.00760 -0.03293 -0.05613
CB D  0.02064  0.09849 -0.08746 -0.06691
CB E  0.01283  0.05404 -0.01342  0.02238
CB F  0.01028  0.03363  0.18112  0.01493
CB G -0.02758  0.04383  0.06071 -0.02639
CB H -0.01760 -0.02367  0.00343  0.00415
CB I  0.02783  0.01052  0.00641  0.05090
CB K  0.00350  0.02852 -0.00408  0.01218
CB L  0.01223 -0.02940 -0.07268  0.00884
CB M -0.02925 -0.03912 -0.06587  0.03490
CB N -0.02242  0.03403 -0.09759 -0.08018
CB P  0.08431 -0.35696 -0.04680  0.05192
CB Q -0.01649 -0.01016 -0.03663  0.01723
CB R -0.01887  0.00618 -0.00385  0.02884
CB S -0.00921  0.07096 -0.06338 -0.03707
CB T  0.02601  0.04904 -0.01728  0.00781
CB V  0.03068  0.06325  0.01928  0.05011
CB W -0.07651 -0.11334  0.13806 -0.03339
CB Y  0.00082  0.01466  0.18107 -0.01181
 N A  0.09963 -0.00873 -2.31666 -0.14051
 N C  0.11905 -0.01296  1.15573  0.01820
 N D  0.11783 -0.11817 -1.16322 -0.37601
 N E  0.10825 -0.00605 -0.41856  0.01187
 N F -0.12280 -0.27542  0.34635  0.09102
 N G  0.10365 -0.05667 -1.50346 -0.00146
 N H -0.04145 -0.26494  0.26356  0.18198
 N I -0.09249  0.12136  2.75071  0.40643
 N K -0.02472  0.07224 -0.07057  0.12261
 N L  0.01542 -0.12800 -0.85172 -0.15460
 N M -0.11266 -0.27311 -0.33192  0.09384
 N N -0.00295 -0.20562 -1.00652 -0.30971
 N P  0.03252  1.35296 -1.17173  0.06026
 N Q  0.00900 -0.09950 -0.07389  0.08415
 N R -0.07819  0.00802 -0.04821  0.08524
 N S  0.12057  0.02242  0.48924 -0.25423
 N T  0.04631  0.09935  1.02269  0.20228
 N V -0.03610  0.21959  2.42228  0.39686
 N W -0.15643 -0.19285  0.05515 -0.53172
 N Y -0.10497 -0.25228  0.46023  0.01399
 H A  0.01337 -0.00605 -0.04371 -0.02485
 H C  0.01324  0.05107  0.12857  0.00610
 H D  0.02859  0.02436 -0.06510  0.02085
 H E  0.02737  0.01790  0.03740  0.01969
 H F -0.02633 -0.08287 -0.11364 -0.03603
 H G  0.02753  0.05640 -0.10477  0.06876
 H H -0.00124 -0.02861  0.04126  0.10004
 H I -0.02258 -0.00929  0.07962  0.01880
 H K -0.00512 -0.00744  0.04443  0.03434
 H L -0.01088 -0.01230 -0.03640 -0.03719
 H M -0.01961 -0.00749 -0.00097  0.02041
 H N  0.01134  0.02121 -0.01837 -0.00629
 H P -0.01246  0.02956  0.13007 -0.00810
 H Q  0.00783  0.00751  0.05643  0.02413
 H R -0.00734  0.00546  0.07003  0.04051
 H S  0.02133  0.03964  0.04978 -0.03749
 H T  0.00976  0.06072  0.03531  0.01657
 H V -0.01267  0.00994  0.09630  0.03420
 H W -0.02348 -0.09617 -0.24207 -0.18741
 H Y -0.01881 -0.07345 -0.14345 -0.06721
HA A  0.00350 -0.02371 -0.00654  0.00652
HA C  0.00660  0.01073  0.01921  0.00919
HA D  0.01717 -0.00854 -0.00802 -0.00597
HA E  0.01090 -0.01091  0.00472  0.00790
HA F -0.02271 -0.06316 -0.03057 -0.02350
HA G  0.02155 -0.00151  0.02477  0.01526
HA H -0.01132 -0.05617 -0.01514  0.01264
HA I  0.00459  0.00571  0.02984  0.00416
HA K  0.00492 -0.01788  0.00555  0.01259
HA L -0.00599 -0.01558  0.00358  0.00167
HA M  0.00100 -0.02037  0.00678  0.00930
HA N  0.00651 -0.01499 -0.00361  0.00203
HA P  0.01542  0.28350 -0.01496  0.00796
HA Q  0.00711 -0.02142  0.00734  0.00971
HA R -0.00472 -0.01414  0.00966  0.01180
HA S  0.01572  0.02791  0.03762  0.00133
HA T  0.01714  0.06590  0.03085  0.00143
HA V  0.00777  0.01505  0.02525  0.00659
HA W -0.06818 -0.08412 -0.09386 -0.06072
HA Y -0.02701 -0.05585 -0.03243 -0.02987
HB A  0.01473  0.01843  0.01428  0.00451
HB C  0.01180  0.03340  0.03081  0.00169
HB D  0.01786  0.01626  0.02221  0.01030
HB E  0.01796  0.01820  0.00835 -0.00045
HB F -0.04867 -0.09154 -0.04858 -0.00164
HB G  0.01718  0.03852  0.01043  0.00051
HB H -0.00817 -0.04557 -0.00820  0.00855
HB I  0.00446  0.00111  0.00049 -0.00283
HB K  0.01570  0.01156  0.00771  0.00646
HB L  0.00700  0.01236  0.00880  0.00150
HB M  0.01607  0.02294  0.01385 -0.00038
HB N  0.01893  0.01561  0.02760  0.01215
HB P -0.01199 -0.02752  0.00891 -0.00033
HB Q  0.01636  0.01861  0.01177 -0.00099
HB R  0.01324  0.01526  0.01082  0.00378
HB S  0.01859  0.03487  0.02890 -0.00477
HB T  0.01624  0.04073  0.01936 -0.00348
HB V  0.00380  0.00271 -0.00144 -0.00315
HB W -0.09045 -0.06895 -0.10934 -0.01948
HB Y -0.05069 -0.06698 -0.05666 -0.01193'''

tabletermcorrs='''C n -0.15238
C c -0.90166
CB n 0.12064
CB c 0.06854
CA n -0.04616
CA c -0.06680
N n 0.347176
N c 0.619141
H n 0.156786
H c 0.023189
HB n 0.0052692
HB c 0.0310875
HA n 0.048624
HA c 0.042019'''

def initcorneis():
    datc=tablenei.split('\n')
    dct={}
    for i in range(20*7):
        vals=datc[i].split()
        atn=vals[0]
        aai=vals[1]
        if not aai in dct:dct[aai]={}
        dct[aai][atn]=[eval(vals[2+j]) for j in range(4)]
    datc=tabletermcorrs.split('\n')
    for i in range(len(datc)):
        vals=datc[i].split()
        atn=vals[0]
        term=vals[1]
        if not term in dct:dct[term]={}
        if term=='n':  dct['n'][atn]=[None,None,None,eval(vals[-1])]
        elif term=='c':dct['c'][atn]=[eval(vals[-1]),None,None,None]
    return dct
        

tabletempk='''aa  CA   CB   C     N    H    HA
A  -2.2  4.7 -7.1  -5.3 -9.0  0.7
C  -0.9  1.3 -2.6  -8.2 -7.0  0.0
D   2.8  6.5 -4.8  -3.9 -6.2 -0.1
E   0.9  4.6 -4.9  -3.7 -6.5  0.3
F  -4.7  2.4 -6.9 -11.2 -7.5  0.4
G   3.3  0.0 -3.2  -6.2 -9.1  0.0
H   7.8 15.5  3.1   3.3 -7.8  0.4
I  -2.0  4.6 -8.7 -12.7 -7.8  0.4
K  -0.8  2.4 -7.1  -7.6 -7.5  0.4
L   1.7  4.9 -8.2  -2.9 -7.5  0.1
M   4.1  9.4 -8.2  -6.2 -7.1 -0.5
N   2.8  5.1 -6.1  -3.3 -7.0 -2.9
P   1.1 -0.2 -4.0   0.0  0.0  0.0
Q   2.3  3.6 -5.7  -6.5 -7.2  0.3
R  -1.4  3.5 -6.9  -5.3 -7.1  0.4
S  -1.7  4.4 -4.7  -3.8 -7.6  0.1
T   0.0  2.2 -5.2  -6.7 -7.3  0.0
V  -2.8  2.5 -8.1 -14.2 -7.6  0.5
W  -2.7  3.1 -7.9 -10.1 -7.8  0.4
Y  -5.0  2.9 -7.7 -12.0 -7.7  0.5'''

def gettempkoeff():
    datc=tabletempk.split('\n')
    buf=[lin.split() for lin in datc]
    headers=buf[0][1:]
    dct={}
    for atn in headers:
        dct[atn]={}
    for lin in buf[1:]:
        aa=lin[0]
        for j,atn in enumerate(headers):
            dct[atn][aa]=eval(lin[1+j])
    return dct

tablecombdevs='''C -1 G r xrGxx  0.2742  1.4856
 C -1 G - x-Gxx  0.0522  0.2827
 C -1 P P xPPxx -0.0822  0.4450
 C -1 P r xrPxx  0.2640  1.4303
 C -1 r P xPrxx -0.1027  0.5566
 C -1 + P xP+xx  0.0714  0.3866
 C -1 - - x--xx -0.0501  0.2712
 C -1 p r xrpxx  0.0582  0.3151
 C  1 G r xxGrx  0.0730  0.3955
 C  1 P a xxPax -0.0981  0.5317
 C  1 P + xxP+x -0.0577  0.3128
 C  1 P p xxPpx -0.0619  0.3356
 C  1 r r xxrrx -0.1858  1.0064
 C  1 r a xxrax -0.1888  1.0226
 C  1 r + xxr+x -0.1805  0.9779
 C  1 r - xxr-x -0.1756  0.9512
 C  1 r p xxrpx -0.1208  0.6544
 C  1 + P xx+Px -0.0533  0.2886
 C  1 - P xx-Px  0.1867  1.0115
 C  1 p P xxpPx  0.2321  1.2574
 C -2 G r rxGxx -0.1457  0.7892
 C -2 r p pxrxx  0.0555  0.3008
 C  2 P P xxPxP  0.1007  0.5455
 C  2 P - xxPx-  0.0634  0.3433
 C  2 r P xxrxP -0.1447  0.7841
 C  2 a r xxaxr -0.1488  0.8061
 C  2 a - xxax- -0.0093  0.0506
 C  2 + G xx+xG -0.0394  0.2132
 C  2 + P xx+xP  0.1016  0.5502
 C  2 + a xx+xa  0.0299  0.1622
 C  2 + + xx+x+  0.0427  0.2312
 C  2 - a xx-xa  0.0611  0.3308
 C  2 p P xxpxP -0.0753  0.4078
CA -1 G P xPGxx -0.0641  0.3233
CA -1 G r xrGxx  0.2107  1.0630
CA -1 P P xPPxx -0.2042  1.0303
CA -1 P p xpPxx  0.0444  0.2240
CA -1 r G xGrxx  0.2030  1.0241
CA -1 r + x+rxx -0.0811  0.4093
CA -1 - P xP-xx  0.0744  0.3755
CA -1 - - x--xx -0.0263  0.1326
CA -1 p p xppxx -0.0094  0.0475
CA  1 G P xxGPx  1.3044  6.5813
CA  1 G - xxG-x -0.0632  0.3188
CA  1 P G xxPGx  0.2642  1.3329
CA  1 P P xxPPx  0.3025  1.5262
CA  1 P r xxPrx  0.1455  0.7343
CA  1 P - xxP-x  0.1188  0.5994
CA  1 P p xxPpx  0.1201  0.6062
CA  1 r P xxrPx -0.1958  0.9878
CA  1 r - xxr-x -0.0931  0.4696
CA  1 a P xxaPx -0.1428  0.7204
CA  1 a - xxa-x -0.0262  0.1324
CA  1 a p xxapx  0.0392  0.1977
CA  1 + P xx+Px -0.1059  0.5344
CA  1 + a xx+ax -0.0377  0.1901
CA  1 + + xx++x -0.0595  0.3001
CA  1 - P xx-Px -0.1156  0.5831
CA  1 - + xx-+x  0.0316  0.1593
CA  1 - p xx-px  0.0612  0.3090
CA  1 p r xxprx -0.0511  0.2576
CA -2 P - -xPxx -0.1028  0.5185
CA -2 r r rxrxx  0.1933  0.9752
CA -2 - G Gx-xx  0.0559  0.2818
CA -2 - p px-xx  0.0391  0.1973
CA -2 p a axpxx -0.0293  0.1479
CA -2 p + +xpxx -0.0173  0.0873
CA  2 G - xxGx-  0.0357  0.1802
CA  2 + G xx+xG -0.0315  0.1591
CA  2 - P xx-xP  0.0426  0.2150
CA  2 - r xx-xr  0.0784  0.3954
CA  2 - a xx-xa  0.1084  0.5467
CA  2 - - xx-x-  0.0836  0.4216
CA  2 p P xxpxP  0.0685  0.3456
CA  2 p - xxpx- -0.0481  0.2428
CB -1 P r xrPxx -0.2678  1.7345
CB -1 P p xpPxx  0.0355  0.2300
CB -1 r P xPrxx -0.1137  0.7367
CB -1 a p xpaxx  0.0249  0.1613
CB -1 + - x-+xx -0.0762  0.4935
CB -1 - P xP-xx -0.0889  0.5757
CB -1 - r xr-xx -0.0533  0.3451
CB -1 - - x--xx  0.0496  0.3215
CB -1 - p xp-xx -0.0148  0.0960
CB -1 p P xPpxx  0.0119  0.0768
CB -1 p r xrpxx -0.0673  0.4358
CB  1 P G xxPGx -0.0522  0.3379
CB  1 P P xxPPx -0.8458  5.4779
CB  1 P r xxPrx -0.1573  1.0187
CB  1 r r xxrrx  0.1634  1.0581
CB  1 a G xxaGx -0.0393  0.2544
CB  1 a r xxarx  0.0274  0.1777
CB  1 a - xxa-x  0.0394  0.2553
CB  1 a p xxapx  0.0149  0.0968
CB  1 + G xx+Gx -0.0784  0.5076
CB  1 + P xx+Px -0.1170  0.7580
CB  1 - P xx-Px -0.0913  0.5912
CB  1 - - xx--x  0.0284  0.1838
CB  1 p P xxpPx  0.0880  0.5697
CB  1 p p xxppx -0.0113  0.0733
CB -2 P - -xPxx  0.0389  0.2521
CB -2 P p pxPxx  0.0365  0.2362
CB -2 r + +xrxx  0.0809  0.5242
CB -2 a - -xaxx -0.0452  0.2927
CB -2 + - -x+xx -0.0651  0.4218
CB -2 - G Gx-xx -0.0883  0.5717
CB -2 p G Gxpxx  0.0378  0.2445
CB -2 p p pxpxx  0.0207  0.1341
CB  2 r G xxrxG -0.0362  0.2344
CB  2 r - xxrx- -0.0219  0.1419
CB  2 a - xxax- -0.0298  0.1929
CB  2 + p xx+xp  0.0189  0.1223
CB  2 - - xx-x- -0.0525  0.3400
 N -1 G P xPGxx  0.2411  0.5105
 N -1 G + x+Gxx -0.1773  0.3754
 N -1 G - x-Gxx  0.1905  0.4035
 N -1 P P xPPxx -0.9177  1.9434
 N -1 P p xpPxx  0.2609  0.5525
 N -1 r G xGrxx  0.2417  0.5119
 N -1 r a xarxx -0.0139  0.0295
 N -1 r + x+rxx -0.4122  0.8729
 N -1 r p xprxx  0.1440  0.3049
 N -1 a G xGaxx -0.5177  1.0963
 N -1 a r xraxx  0.0890  0.1885
 N -1 a a xaaxx  0.1393  0.2950
 N -1 a p xpaxx -0.0825  0.1747
 N -1 + G xG+xx -0.4908  1.0394
 N -1 + a xa+xx  0.1709  0.3619
 N -1 + + x++xx  0.1868  0.3955
 N -1 + - x-+xx -0.0951  0.2014
 N -1 - P xP-xx -0.3027  0.6410
 N -1 - r xr-xx -0.1670  0.3537
 N -1 - + x+-xx -0.3501  0.7414
 N -1 - - x--xx  0.1266  0.2681
 N -1 p G xGpxx -0.1707  0.3614
 N -1 p - x-pxx  0.0011  0.0023
 N  1 G G xxGGx  0.2555  0.5412
 N  1 G P xxGPx -0.9725  2.0595
 N  1 G r xxGrx  0.0165  0.0349
 N  1 G p xxGpx  0.0703  0.1489
 N  1 r a xxrax -0.0237  0.0503
 N  1 a r xxarx -0.1816  0.3845
 N  1 a - xxa-x -0.1050  0.2224
 N  1 a p xxapx -0.1196  0.2533
 N  1 - r xx-rx -0.1762  0.3731
 N  1 - a xx-ax  0.0006  0.0013
 N  1 p P xxpPx  0.2797  0.5923
 N  1 p a xxpax  0.0938  0.1986
 N  1 p + xxp+x  0.1359  0.2878
 N -2 G r rxGxx -0.5140  1.0885
 N -2 G - -xGxx -0.0639  0.1354
 N -2 P P PxPxx -0.4215  0.8927
 N -2 r P Pxrxx -0.3696  0.7828
 N -2 r p pxrxx -0.1937  0.4101
 N -2 a - -xaxx -0.0351  0.0743
 N -2 a p pxaxx -0.1031  0.2183
 N -2 - G Gx-xx -0.2152  0.4558
 N -2 - P Px-xx -0.1375  0.2912
 N -2 - p px-xx -0.1081  0.2290
 N -2 p P Pxpxx -0.1489  0.3154
 N -2 p - -xpxx  0.0952  0.2015
 N  2 G - xxGx-  0.1160  0.2457
 N  2 r p xxrxp -0.1288  0.2728
 N  2 a P xxaxP  0.1632  0.3456
 N  2 + + xx+x+ -0.0106  0.0226
 N  2 + - xx+x-  0.0389  0.0824
 N  2 - a xx-xa -0.0815  0.1726
 N  2 p G xxpxG -0.0779  0.1649
 N  2 p p xxpxp -0.0683  0.1447
 H -1 G P xPGxx -0.0317  0.4730
 H -1 G r xrGxx  0.0549  0.8186
 H -1 G + x+Gxx -0.0192  0.2867
 H -1 G - x-Gxx  0.0138  0.2055
 H -1 r P xPrxx -0.0964  1.4367
 H -1 r - x-rxx -0.0245  0.3648
 H -1 a G xGaxx -0.0290  0.4320
 H -1 a a xaaxx  0.0063  0.0944
 H -1 + G xG+xx -0.0615  0.9168
 H -1 + r xr+xx -0.0480  0.7153
 H -1 + - x-+xx -0.0203  0.3030
 H -1 - + x+-xx -0.0232  0.3455
 H -1 p G xGpxx -0.0028  0.0411
 H -1 p P xPpxx -0.0121  0.1805
 H  1 G P xxGPx -0.1418  2.1144
 H  1 G r xxGrx  0.0236  0.3520
 H  1 G a xxGax  0.0173  0.2580
 H  1 a - xxa-x  0.0091  0.1349
 H  1 + P xx+Px -0.0422  0.6290
 H  1 + p xx+px  0.0191  0.2842
 H  1 - P xx-Px -0.0474  0.7065
 H  1 - a xx-ax  0.0102  0.1515
 H -2 G G GxGxx  0.0169  0.2517
 H -2 G r rxGxx -0.3503  5.2220
 H -2 a P Pxaxx  0.0216  0.3227
 H -2 a - -xaxx -0.0276  0.4118
 H -2 + - -x+xx -0.0260  0.3874
 H -2 - G Gx-xx  0.0273  0.4073
 H -2 - a ax-xx -0.0161  0.2400
 H -2 - - -x-xx -0.0285  0.4255
 H -2 p P Pxpxx -0.0101  0.1503
 H -2 p a axpxx -0.0157  0.2343
 H -2 p + +xpxx -0.0122  0.1815
 H -2 p p pxpxx  0.0107  0.1601
 H  2 G G xxGxG -0.0190  0.2826
 H  2 r G xxrxG  0.0472  0.7036
 H  2 r P xxrxP  0.0337  0.5027
 H  2 a + xxax+ -0.0159  0.2376
 H  2 + G xx+xG  0.0113  0.1685
 H  2 + r xx+xr -0.0307  0.4575
 H  2 - P xx-xP -0.0088  0.1318
HA -1 P P xPPxx  0.0307  1.1685
HA -1 P r xrPxx  0.0621  2.3592
HA -1 r G xGrxx -0.0371  1.4092
HA -1 r + x+rxx  0.0125  0.4733
HA -1 r p xprxx -0.0199  0.7569
HA -1 a G xGaxx  0.0073  0.2779
HA -1 a a xaaxx  0.0044  0.1683
HA -1 - G xG-xx  0.0116  0.4409
HA -1 - r xr-xx  0.0228  0.8679
HA -1 - p xp-xx  0.0074  0.2828
HA  1 G G xxGGx  0.0175  0.6636
HA  1 G - xxG-x  0.0107  0.4081
HA  1 P a xxPax  0.0089  0.3369
HA  1 - r xx-rx  0.0113  0.4291
HA -2 G G GxGxx -0.0154  0.5847
HA -2 P - -xPxx  0.0136  0.5179
HA -2 r G Gxrxx -0.0159  0.6045
HA -2 + + +x+xx -0.0137  0.5190
HA -2 p - -xpxx -0.0068  0.2592
HA -2 p p pxpxx  0.0046  0.1763
HB -1 P r xrPxx  0.0460  2.1365
HB -1 a - x-axx  0.0076  0.3551
HB -1 + - x-+xx  0.0110  0.5122
HB -1 - r xr-xx  0.0233  1.0819
HB  1 a P xxaPx  0.0287  1.3310
HB  1 + P xx+Px  0.0324  1.5056
HB  1 + r xx+rx -0.0231  1.0709
HB  1 p r xxprx  0.0077  0.3586
HB  1 p + xxp+x -0.0074  0.3426
HB -2 a P Pxaxx -0.0026  0.1192
HB -2 a r rxaxx -0.0098  0.4559
HB -2 - - -x-xx  0.0016  0.0751
HB  2 P r xxPxr -0.0595  2.7608
HB  2 P + xxPx+ -0.0145  0.6744
HB  2 P - xxPx-  0.0107  0.4976
HB  2 a + xxax+ -0.0015  0.0691
HB  2 p r xxpxr  0.0262  1.2178'''

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
    datc=tablecombdevs.split('\n')
    buf=[lin.split() for lin in datc]
    dct={}
    for lin in buf:
        atn=lin[0]
        if not atn in dct:dct[atn]={}
        neipos=int(lin[1])
        centgroup=lin[2]
        neigroup= lin[3]
        key=(neipos,centgroup,neigroup)#(k,l,m)
        segment=lin[4]
        dct[atn][segment]=key,eval(lin[-2])
    return dct
        
TEMPCORRS=gettempkoeff()
CENTSHIFTS=initcorcents()
NEICORRS =initcorneis()
COMBCORRS=initcorrcomb()

# data = pkgutil.get_data(__name__, "data_tables/phshifts2.csv")
# PHSHIFTS = pd.read_csv(StringIO(data.decode()), header=0, comment='#', index_col=['resn', 'atn'])

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

def get_phshifts():
    datc=tablephshifts.split('\n')
    buf=[lin.split() for lin in datc]
    dct={}
    na=None
    for lin in buf:
        if len(lin)>3:
            resn=lin[0]
            atn=lin[1]
            sh0=eval(lin[2])
            sh1=eval(lin[3])
            shd=eval(lin[4])
            if not resn in dct:dct[resn]={}
            dct[resn][atn]=shd
            if len(lin)>6:#neighbor data
                for n in range(2):
                    shdn=eval(lin[5+n])
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
        pKaval=eval(pKa)
        nHval=eval(nH)
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
                    logging.getLogger('trizod.potenci').waring(f'no key: {resi}, {i}, {atn}')
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

def getphcorrs_arr(seq,temperature,pH,ion):
    bbatns=['C','CA','CB','HA','H','N','HB']
    dct=get_phshifts()
    
    Ion=max(0.0001,ion)

    pkadct=calc_pkas_from_seq('n'+seq+'c',temperature,Ion)
    #outdct={}
    residues = [[None]*7 for i in range(len(seq))]
    outarr = np.zeros(shape=(len(seq), len(bbatns)), dtype=np.float)
    for i in pkadct:
        logging.getLogger('trizod.potenci').debug('pkares: %6.3f %6.3f %1s'%pkadct[i] + str(i))
        pKa,nH,resi=pkadct[i]
        frac  = fun(pH,pKa,nH)
        frac7 = fun(7.0,pK0[resi],nH)
        if resi in 'nc':
            jump = 0.0#so far
        else:
            for col,atn in enumerate(bbatns):
                #if not atn in outdct:outdct[atn]={}
                logging.getLogger('trizod.potenci').debug(f'data: {atn}, {pKa}, {nH}, {resi}, {i}, {atn}, {pH}')
                dctresi = dct[resi]
                try:
                    delta = dctresi[atn]
                    jump  = frac *delta
                    jump7 = frac7*delta
                    #key=(resi,atn)
                except KeyError:
                    ##if not (resi in 'RKCY' and atn=='H') and not (resi == 'R' and atn=='N'):
                    logging.getLogger('trizod.potenci').waring(f'no key: {resi}, {i}, {atn}')
                    delta=999;jump=999;jump7=999
                if delta<99:
                    jumpdelta = jump - jump7
                    #if not i in outdct[atn]:outdct[atn][i]=[resi,jumpdelta]
                    #else:
                    #    outdct[atn][i][0]=resi
                    #    outdct[atn][i][1]+=jumpdelta
                    residues[i][col] = resi
                    outarr[i][col] += jumpdelta
                    logging.getLogger('trizod.potenci').debug('%3s %5.2f %6.4f %s %3d %5s %8.5f %8.5f %4.2f'%(atn,pKa,nH,resi,i,atn,jump,jump7,pH))
                    if resi+'p' in dct and atn in dct[resi+'p']:
                        for n in range(2):
                            ni=i+2*n-1
                            ##if ni is somewhere in seq...
                            nresi=resi+'ps'[n]
                            ndelta=dct[nresi][atn]
                            jump = frac  * ndelta
                            jump7= frac7 * ndelta
                            jumpdelta = jump - jump7
                            #if not ni in outdct[atn]:outdct[atn][ni]=[None,jumpdelta]
                            #else:outdct[atn][ni][1]+=jumpdelta
                            residues[i][col] = None
                            outarr[ni][col] += jumpdelta
    return outarr

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

def getpredshifts_arr(seq,temperature,pH,ion,usephcor=True,pkacsvfile=None,identifier=''):
    tempdct=gettempkoeff()
    bbatns = ['C','CA','CB','HA','H','N','HB']
    if usephcor:
        phcorrs=getphcorrs_arr(seq,temperature,pH,ion,pkacsvfile)
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
                            if seq[i] in 'CDEHRKY' and resi!=seq[i]:
                                logging.getLogger('trizod.potenci').warning(f'residue mismatch: {resi},{seq[i]},{i},{phdata},{at}')
                            phcorr=phdata[1]
                            if abs(phcorr)<9.9:
                                shp-=phcorr
                        shiftdct[(i+1,seq[i])][at]=shp
                        logging.getLogger('trizod.potenci').debug('predictedshift: %5s %3d %1s %2s %8.4f'%(identifier,i,seq[i],at,shp) + ' ' + str(phcorr))
    return shiftdct

def writeOutput(name,dct):
    out=open(name,'w')
    bbatns =['N','C','CA','CB','H','HA','HB']
    out.write('#NUM AA   N ')
    out.write(' %7s %7s %7s %7s %7s %7s\n'%tuple(bbatns[1:]))
    reskeys=list(dct.keys())
    reskeys.sort()
    for resnum,resn in reskeys:
        shdct=dct[(resnum,resn)]
        if len(shdct)>0:
            out.write('%-4d %1s '%(resnum,resn))
            for at in bbatns:
                shp=0.0
                if at in shdct:shp=shdct[at]
                out.write(' %7.3f'%shp)
        out.write('\n')
    out.close()

def main():
    ##README##......
    #requires: python2.x with numpy and scipy
    #usage: potenci1_2.py seqstring pH temperature ionicstrength [pkacsvfile] > logfile
    #optional filename in csv format contained predicted pKa values and Hill parameters,
    #the format of the pkacsvfile must be the same as the output for pepKalc,
    #only lines after "Site" is read. If this is not found no pH corrections are applied.
    #Output: Table textfile in SHIFTY format (space separated)
    #average of methylene protons are provided for Gly HA2/HA3 and HB2/HB3.
    #NOTE:pH corrections is applied if pH is not 7.0
    #NOTE:pKa predictions are stored locally and reloaded if the seq, temp and ion is the same.
    #NOTE:at least 5 residues are required. Chemical shift predictions are not given for terminal residues.
    args=sys.argv[1:]#first is scriptname
    if len(args)<4:
        logging.getLogger('trizod.potenci').error('FAILED: please provide 4 arguments (exiting)') 
        logging.getLogger('trizod.potenci').info('usage: potenci1_2.py seqstring pH temperature ionicstrength [pkacsvfile] > logfile')
        raise SystemExit
    seq=args[0] #one unbroken line with single-letter amino acid labels
    pH=float(args[1])#e.g. 7.0
    temperature=float(args[2])#e.g. 298.0 / K
    ion=float(args[3])#e.g. 0.1 / M
    pkacsvfile=None
    if len(args)>4:pkacsvfile=args[4]
    ##name='outPOTENCI_%s_T%6.2f_I%4.2f_pH%4.2f.txt'%(seq,temperature,ion,pH)
    name='outPOTENCI_%s_T%6.2f_I%4.2f_pH%4.2f.txt'%(seq[:min(150,len(seq))],temperature,ion,pH)
    usephcor = pH<6.99 or pH>7.01
    if len(seq)<5:
        logging.getLogger('trizod.potenci').error('FAILED: at least 5 residues are required (exiting)') 
        raise SystemExit
    #------------- now ready to generate predicted shifts ---------------------
    logging.getLogger('trizod.potenci').info('predicting random coil chemical shift with POTENCI using:',seq,pH,temperature,ion,pkacsvfile)
    shiftdct=getpredshifts(seq,temperature,pH,ion,usephcor,pkacsvfile)
    #------------- write output nicely is SHIFTY format -----------------------$
    writeOutput(name,shiftdct)
    logging.getLogger('trizod.potenci').info('chemical shift succesfully predicted, see output:',name)

if __name__ == '__main__':
    main()
