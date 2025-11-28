#!/usr/bin/env python3
"""POTENCI implementation for predicting random coil NMR chemical shifts.

This is a refactored version of the POTENCI algorithm, originally developed by
Frans A. A. Mulder's group. The implementation has been modernized with:
- Type hints and dataclasses
- CSV-based data storage
- Improved code organization
- Security improvements (no eval())

Original author: fmulder@chem.au.dk
Adapted by: markus.haak@tum.de & tobias.senoner@tum.de
Original source: https://github.com/protein-nmr/POTENCI
Original commit: 17dd2e6f3733c702323894697238c87e6723f934 (2019-06-07)
Original filename: pytenci1_3.py

References:
    Nielsen, J. T., & Mulder, F. A. (2018). POTENCI: prediction of temperature,
    neighbor and pH-corrected chemical shifts for intrinsically disordered proteins.
    Journal of Biomolecular NMR, 70(3), 141-165.
"""

import logging
import sys

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc

from trizod.potenci.constants import (
    AA_STANDARD,
    PHYSICAL_CONSTANTS,
    PK0,
    alltuples_,
    load_central_shifts,
    load_combinatorial_deviations,
    load_neighbor_corrections,
    load_ph_shifts,
    load_temperature_coefficients,
    load_terminal_corrections,
    outer_matrices,
)

# Load data tables at module level (cached automatically)
CENTSHIFTS = load_central_shifts()
NEICORRS = {**load_neighbor_corrections(), **load_terminal_corrections()}
COMBCORRS = load_combinatorial_deviations()
TEMPCORRS = load_temperature_coefficients()
PHSHIFTS = load_ph_shifts()

# Physical constants
R = PHYSICAL_CONSTANTS.gas_constant
e = PHYSICAL_CONSTANTS.dielectric_constant
a = PHYSICAL_CONSTANTS.distance_param_a
b = PHYSICAL_CONSTANTS.distance_param_b
cutoff = PHYSICAL_CONSTANTS.cutoff
ncycles = PHYSICAL_CONSTANTS.n_cycles


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
    # return (10 ** ( nH*(pK - pH) ) ) / (1. + (10 **( nH*(pK - pH) ) ) )
    return 1.0 - 1.0 / ((10 ** (nH * (pK - pH))) + 1.0)  # identical


def log_fun(pH, pK, nH):
    return -np.log10(1 + 10 ** (nH * (pH - pK)))


def w(r, Ion=0.1):
    k = np.sqrt(Ion) / 3.08  # Ion=0.1 is default
    x = k.astype(np.float64) * r.astype(np.float64) / np.sqrt(6)
    i1 = 332.286 * np.sqrt(6 / np.pi)
    i2_3 = erfc(x)
    i2_2 = np.sqrt(np.pi) * x

    i3 = e * r
    i4 = np.exp(
        (x**2) - np.log(i3)
    )  # always equal to np.exp(x ** 2) / (e * r), but intermediates are smaller
    i4 = np.nan_to_num(i4)  # to convert inf values to the largest possible value
    return i1 * ((1 / i3) - np.nan_to_num(i4 * i2_2 * i2_3))


def w2logp(x, T=293.15):
    return x * 4181.2 / (R * T * np.log(10))


def calc_pkas_from_seq(seq=None, T=293.15, Ion=0.1):
    # pH range
    pHs = np.arange(1.99, 10.01, 0.15)

    pos = np.array([i for i in range(len(seq)) if seq[i] in PK0])
    N = pos.shape[0]
    identity_matrix = np.diag(np.ones(N))
    sites = "".join([seq[i] for i in pos])
    neg = np.array([i for i in range(len(sites)) if sites[i] in "DEYc"])
    lengths = np.array([abs(pos - pos[i]) for i in range(N)])
    d = a + np.sqrt(lengths) * b

    tmp = w(d, Ion)
    tmp[identity_matrix == 1] = 0

    ww = w2logp(tmp, T) / 2

    chargesempty = np.zeros(pos.shape[0])
    if len(neg):
        chargesempty[neg] = -1

    pK0s = np.array([PK0[c] for c in sites])
    nH0s = np.array([0.9 for c in sites])

    titration = np.zeros((N, len(pHs)))

    smallN = min(2 * cutoff + 1, len(pos))
    alltuples = alltuples_[smallN]
    outerm = outer_matrices[smallN]
    gmatrix = [np.zeros((smallN, smallN)) for _ in range(len(pHs))]

    # Perform iterative fitting for pKa calculation
    for icycle in range(ncycles):
        if icycle == 0:
            fractionhold = np.array(
                [[fun(pHs[p], pK0s[i], nH0s[i]) for i in range(N)] for p in range(len(pHs))]
            )
        else:
            fractionhold = titration.transpose()

        for ires in range(1, N + 1):
            (ileft, iright) = smallmatrixlimits(ires, cutoff, N)
            resi = smallmatrixpos(ires, cutoff, N)
            fraction = fractionhold.copy()
            fraction[:, ileft - 1 : iright] = 0
            charges = fraction + chargesempty
            ww0 = 2 * (ww * np.expand_dims(charges, axis=1)).sum(axis=-1)
            ww0 = np.expand_dims(ww0, 1) * identity_matrix  # array of diagonal matrices
            gmatrixfull = ww + ww0 + np.expand_dims(pHs, (1, 2)) * identity_matrix - np.diag(pK0s)
            gmatrix = gmatrixfull[:, ileft - 1 : iright, ileft - 1 : iright]

            E = 10 ** -(np.expand_dims(gmatrix, axis=1) * outerm).sum(axis=(2, 3))  # .sum(axis=-1)
            E_all = E.sum(axis=-1)
            E_sel = E[:, (alltuples[:, resi - 1] == 1)].sum(axis=-1)
            titration[ires - 1] = E_sel / E_all
        sol = np.array(
            [
                curve_fit(fun, pHs, titration[p], [pK0s[p], nH0s[p]], maxfev=5000)[0]
                for p in range(len(pK0s))
            ]
        )
        (pKs, nHs) = sol.transpose()

    dct = {}
    for p, i in enumerate(pos):
        dct[i - 1] = (pKs[p], nHs[p], seq[i])

    return dct


def pred_pent_shift(pent, atn):
    aac = pent[2]
    sh = CENTSHIFTS[aac][atn]
    allneipos = [2, 1, -1, -2]
    for i in range(4):
        aai = pent[2 + allneipos[i]]
        if aai in NEICORRS:
            corr = NEICORRS[aai][atn][i]
            sh += corr
    groups = ["G", "P", "FYW", "LIVMCA", "KR", "DE"]  # Group classifications for residues
    labels = "GPra+-p"  # (Gly,Pro,Arom,Aliph,pos,neg,polar)
    grstr = ""
    for i in range(5):
        aai = pent[i]
        found = False
        for j, gr in enumerate(groups):
            if aai in gr:
                grstr += labels[j]
                found = True
                break
        if not found:
            grstr += "p"  # polar
    centgr = grstr[2]
    for segm in COMBCORRS[atn]:
        key, combval = COMBCORRS[atn][segm]
        neipos, centgroup, neigroup = key  # (k,l,m)
        if (
            centgroup == centgr
            and grstr[2 + neipos] == neigroup
            and ((centgr, neigroup) != ("p", "p") or pent[2] in "ST")
        ):
            # pp comb only used when center is Ser or Thr!
            sh += combval
    return sh


def gettempcorr(aai, atn, tempdct, temp):
    return tempdct[atn][aai] / 1000 * (temp - 298)


def initfilcsv(filename):
    with open(filename) as file:
        buffer = file.readlines()
    for i in range(len(buffer)):
        buffer[i] = buffer[i][:-1].split(",")
    return buffer


def write_csv_pkaoutput(pkadct, seq, temperature, ion):
    seq = seq[: min(150, len(seq))]
    name = f"outpepKalc_{seq}_T{temperature:6.2f}_I{ion:4.2f}.csv"
    with open(name, "w") as out:
        out.write("Site,pKa value,pKa shift,Hill coefficient\n")
        for i in pkadct:
            pKa, nH, resi = pkadct[i]
            reskey = resi + str(i + 1)
            diff = pKa - PK0[resi]
            out.write(f"{reskey},{pKa:5.3f},{diff:5.3f},{nH:5.3f}\n")


def read_csv_pkaoutput(seq, temperature, ion, name=None):
    seq = seq[: min(150, len(seq))]
    logging.getLogger("trizod.potenci").debug(f"reading csv {name}")
    if name is None:
        name = f"outpepKalc_{seq}_T{temperature:6.2f}_I{ion:4.2f}.csv"
    try:
        with open(name):
            pass
    except OSError:
        return None
    buf = initfilcsv(name)
    for _lnum, data in enumerate(buf):
        if len(data) > 0 and data[0] == "Site":
            break
    pkadct = {}
    for data in buf[_lnum + 1 :]:
        reskey, pKa, diff, nH = data
        i = int(reskey[1:]) - 1
        resi = reskey[0]
        pKaval = eval(pKa)
        nHval = eval(nH)
        pkadct[i] = pKaval, nHval, resi
    return pkadct


def getphcorrs(seq, temperature, pH, ion, pkacsvfilename=None):
    bbatns = ["C", "CA", "CB", "HA", "H", "N", "HB"]
    dct = PHSHIFTS
    Ion = max(0.0001, ion)
    if not pkacsvfilename:
        pkadct = None
    else:
        pkadct = read_csv_pkaoutput(seq, temperature, ion, pkacsvfilename)
    if pkadct is None:
        pkadct = calc_pkas_from_seq("n" + seq + "c", temperature, Ion)
        if pkacsvfilename:
            write_csv_pkaoutput(pkadct, seq, temperature, ion)
    outdct = {}
    for i in pkadct:
        pKa, nH, resi = pkadct[i]
        logging.getLogger("trizod.potenci").debug(f"pkares: {pKa:6.3f} {nH:6.3f} {resi:1s}{i}")
        frac = fun(pH, pKa, nH)
        frac7 = fun(7.0, PK0[resi], nH)
        if resi in "nc":
            jump = 0.0  # so far
        else:
            for atn in bbatns:
                if atn not in outdct:
                    outdct[atn] = {}
                logging.getLogger("trizod.potenci").debug(
                    f"data: {atn}, {pKa}, {nH}, {resi}, {i}, {atn}, {pH}"
                )
                dctresi = dct[resi]
                try:
                    delta = dctresi[atn]
                    jump = frac * delta
                    jump7 = frac7 * delta
                except KeyError:
                    logging.getLogger("trizod.potenci").warning(f"no key: {resi}, {i}, {atn}")
                    delta = 999
                    jump = 999
                    jump7 = 999
                if delta < 99:
                    jumpdelta = jump - jump7
                    if i not in outdct[atn]:
                        outdct[atn][i] = [resi, jumpdelta]
                    else:
                        outdct[atn][i][0] = resi
                        outdct[atn][i][1] += jumpdelta
                    logging.getLogger("trizod.potenci").debug(
                        f"{atn:3s} {pKa:5.2f} {nH:6.4f} {resi} {i:3d} {atn:5s} {jump:8.5f} {jump7:8.5f} {pH:4.2f}"
                    )
                    if resi + "p" in dct and atn in dct[resi + "p"]:
                        for n in range(2):
                            ni = i + 2 * n - 1
                            # Apply neighbor pH corrections
                            nresi = resi + "ps"[n]
                            ndelta = dct[nresi][atn]
                            # ndelta = PHSHIFTS.loc[(nresi,atn), 'shd']
                            jump = frac * ndelta
                            jump7 = frac7 * ndelta
                            jumpdelta = jump - jump7
                            if ni not in outdct[atn]:
                                outdct[atn][ni] = [None, jumpdelta]
                            else:
                                outdct[atn][ni][1] += jumpdelta
    return outdct


def getphcorrs_arr(seq, temperature, pH, ion):
    bbatns = ["C", "CA", "CB", "HA", "H", "N", "HB"]
    dct = PHSHIFTS

    Ion = max(0.0001, ion)

    pkadct = calc_pkas_from_seq("n" + seq + "c", temperature, Ion)
    # outdct={}
    residues = [[None] * 7 for i in range(len(seq))]
    outarr = np.zeros(shape=(len(seq), len(bbatns)), dtype=np.float)
    for i in pkadct:
        pKa, nH, resi = pkadct[i]
        logging.getLogger("trizod.potenci").debug(f"pkares: {pKa:6.3f} {nH:6.3f} {resi:1s}{i}")
        frac = fun(pH, pKa, nH)
        frac7 = fun(7.0, PK0[resi], nH)
        if resi in "nc":
            jump = 0.0  # so far
        else:
            for col, atn in enumerate(bbatns):
                # if not atn in outdct:outdct[atn]={}
                logging.getLogger("trizod.potenci").debug(
                    f"data: {atn}, {pKa}, {nH}, {resi}, {i}, {atn}, {pH}"
                )
                dctresi = dct[resi]
                try:
                    delta = dctresi[atn]
                    jump = frac * delta
                    jump7 = frac7 * delta
                except KeyError:
                    logging.getLogger("trizod.potenci").warning(f"no key: {resi}, {i}, {atn}")
                    delta = 999
                    jump = 999
                    jump7 = 999
                if delta < 99:
                    jumpdelta = jump - jump7
                    # if not i in outdct[atn]:outdct[atn][i]=[resi,jumpdelta]
                    # else:
                    #    outdct[atn][i][0]=resi
                    #    outdct[atn][i][1]+=jumpdelta
                    residues[i][col] = resi
                    outarr[i][col] += jumpdelta
                    logging.getLogger("trizod.potenci").debug(
                        f"{atn:3s} {pKa:5.2f} {nH:6.4f} {resi} {i:3d} {atn:5s} {jump:8.5f} {jump7:8.5f} {pH:4.2f}"
                    )
                    if resi + "p" in dct and atn in dct[resi + "p"]:
                        for n in range(2):
                            ni = i + 2 * n - 1
                            # Apply neighbor pH corrections
                            nresi = resi + "ps"[n]
                            ndelta = dct[nresi][atn]
                            jump = frac * ndelta
                            jump7 = frac7 * ndelta
                            jumpdelta = jump - jump7
                            residues[i][col] = None
                            outarr[ni][col] += jumpdelta
    return outarr


def getpredshifts(seq, temperature, pH, ion, usephcor=True, pkacsvfile=None, identifier=""):
    tempdct = TEMPCORRS
    bbatns = ["C", "CA", "CB", "HA", "H", "N", "HB"]
    phcorrs = getphcorrs(seq, temperature, pH, ion, pkacsvfile) if usephcor else {}
    shiftdct = {}
    for i in range(1, len(seq) - 1):
        if seq[i] in AA_STANDARD:  # else: do nothing
            str(i + 1)
            trip = seq[i - 1] + seq[i] + seq[i + 1]
            phcorr = None
            shiftdct[(i + 1, seq[i])] = {}
            for at in bbatns:
                if (trip[1], at) not in [("G", "CB"), ("G", "HB"), ("P", "H")]:
                    if i == 1:
                        pent = "n" + trip + seq[i + 2]
                    elif i == len(seq) - 2:
                        pent = seq[i - 2] + trip + "c"
                    else:
                        pent = seq[i - 2] + trip + seq[i + 2]
                    shp = pred_pent_shift(pent, at)
                    if shp is not None:
                        if at != "HB":
                            shp += gettempcorr(trip[1], at, tempdct, temperature)
                        if at in phcorrs and i in phcorrs[at]:
                            phdata = phcorrs[at][i]
                            resi = phdata[0]
                            # Verify residue identity matches
                            if seq[i] in "CDEHRKY" and resi != seq[i]:
                                logging.getLogger("trizod.potenci").warning(
                                    f"residue mismatch: {resi},{seq[i]},{i},{phdata},{at}"
                                )
                            phcorr = phdata[1]
                            if abs(phcorr) < 9.9:
                                shp -= phcorr
                        shiftdct[(i + 1, seq[i])][at] = shp
                        logging.getLogger("trizod.potenci").debug(
                            f"predictedshift: {identifier:5s} {i:3d} {seq[i]:1s} {at:2s} {shp:8.4f} {phcorr}"
                        )
    return shiftdct


def getpredshifts_arr(seq, temperature, pH, ion, usephcor=True, pkacsvfile=None, identifier=""):
    tempdct = TEMPCORRS
    bbatns = ["C", "CA", "CB", "HA", "H", "N", "HB"]
    phcorrs = getphcorrs_arr(seq, temperature, pH, ion, pkacsvfile) if usephcor else {}
    shiftdct = {}
    for i in range(1, len(seq) - 1):
        if seq[i] in AA_STANDARD:  # else: do nothing
            str(i + 1)
            trip = seq[i - 1] + seq[i] + seq[i + 1]
            phcorr = None
            shiftdct[(i + 1, seq[i])] = {}
            for at in bbatns:
                if (trip[1], at) not in [("G", "CB"), ("G", "HB"), ("P", "H")]:
                    if i == 1:
                        pent = "n" + trip + seq[i + 2]
                    elif i == len(seq) - 2:
                        pent = seq[i - 2] + trip + "c"
                    else:
                        pent = seq[i - 2] + trip + seq[i + 2]
                    shp = pred_pent_shift(pent, at)
                    if shp is not None:
                        if at != "HB":
                            shp += gettempcorr(trip[1], at, tempdct, temperature)
                        if at in phcorrs and i in phcorrs[at]:
                            phdata = phcorrs[at][i]
                            resi = phdata[0]
                            # Verify residue identity matches
                            if seq[i] in "CDEHRKY" and resi != seq[i]:
                                logging.getLogger("trizod.potenci").warning(
                                    f"residue mismatch: {resi},{seq[i]},{i},{phdata},{at}"
                                )
                            phcorr = phdata[1]
                            if abs(phcorr) < 9.9:
                                shp -= phcorr
                        shiftdct[(i + 1, seq[i])][at] = shp
                        logging.getLogger("trizod.potenci").debug(
                            f"predictedshift: {identifier:5s} {i:3d} {seq[i]:1s} {at:2s} {shp:8.4f} {phcorr}"
                        )
    return shiftdct


def write_output(name, dct):
    with open(name, "w") as out:
        bbatns = ["N", "C", "CA", "CB", "H", "HA", "HB"]
        out.write("#NUM AA   N ")
        out.write(
            f" {bbatns[1]:>7s} {bbatns[2]:>7s} {bbatns[3]:>7s} {bbatns[4]:>7s} {bbatns[5]:>7s} {bbatns[6]:>7s}\n"
        )
        reskeys = list(dct.keys())
        reskeys.sort()
        for resnum, resn in reskeys:
            shdct = dct[(resnum, resn)]
            if len(shdct) > 0:
                out.write(f"{resnum:<4d} {resn:1s} ")
                for at in bbatns:
                    shp = 0.0
                    if at in shdct:
                        shp = shdct[at]
                    out.write(f" {shp:7.3f}")
            out.write("\n")


def main():
    """Command-line interface for POTENCI chemical shift prediction.

    Usage:
        python -m trizod.potenci.potenci <sequence> <pH> <temperature> <ionic_strength> [pkacsvfile]

    Arguments:
        sequence: Protein sequence (one-letter amino acid codes)
        pH: pH value (e.g., 7.0)
        temperature: Temperature in Kelvin (e.g., 298.0)
        ionic_strength: Ionic strength in M (e.g., 0.1)
        pkacsvfile: Optional CSV file with pre-computed pKa values

    Requirements:
        - Python 3.10+
        - numpy, scipy

    Output:
        - Text file in SHIFTY format (space-separated columns)
        - Averaged methylene proton shifts for Gly HA2/HA3 and HB2/HB3
        - CSV file with pKa predictions (if not provided)

    Notes:
        - pH corrections applied if pH != 7.0
        - pKa predictions cached and reused for same (sequence, temp, ion) combination
        - Minimum 5 residues required
        - Chemical shifts not predicted for terminal residues
    """
    args = sys.argv[1:]
    if len(args) < 4:
        logging.getLogger("trizod.potenci").error("FAILED: please provide 4 arguments (exiting)")
        logging.getLogger("trizod.potenci").info(
            "Usage: python -m trizod.potenci.potenci <sequence> <pH> <temperature> <ionic_strength> [pkacsvfile]"
        )
        raise SystemExit
    seq = args[0]  # one unbroken line with single-letter amino acid labels
    pH = float(args[1])  # e.g. 7.0
    temperature = float(args[2])  # e.g. 298.0 / K
    ion = float(args[3])  # e.g. 0.1 / M
    pkacsvfile = None
    if len(args) > 4:
        pkacsvfile = args[4]
    # Generate output filename with truncated sequence (max 150 chars)
    name = f"outPOTENCI_{seq[: min(150, len(seq))]}_T{temperature:6.2f}_I{ion:4.2f}_pH{pH:4.2f}.txt"
    usephcor = pH < 6.99 or pH > 7.01
    if len(seq) < 5:
        logging.getLogger("trizod.potenci").error(
            "FAILED: at least 5 residues are required (exiting)"
        )
        raise SystemExit
    # Generate predicted chemical shifts
    logging.getLogger("trizod.potenci").info(
        "predicting random coil chemical shift with POTENCI using:",
        seq,
        pH,
        temperature,
        ion,
        pkacsvfile,
    )
    shiftdct = getpredshifts(seq, temperature, pH, ion, usephcor, pkacsvfile)
    # Write output in SHIFTY format
    write_output(name, shiftdct)
    logging.getLogger("trizod.potenci").info(
        "chemical shift succesfully predicted, see output:", name
    )


if __name__ == "__main__":
    main()
