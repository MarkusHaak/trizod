"""POTENCI — random coil chemical shift prediction for proteins.

Adapted from https://github.com/protein-nmr/POTENCI (commit 17dd2e6).
Original author: fmulder@chem.au.dk
Adapted by: haak@rostlab.org

Public API:
    get_pred_shifts(seq, temperature, pH, ion, ...) -> dict
"""

import csv
import logging
import os

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc

from trizod.potenci.constants import (
    DIELECTRIC_WATER,
    DISTANCE_SCALE,
    GAS_CONSTANT,
    MIN_CHARGE_DISTANCE,
    PKA_FIT_CYCLES,
    PKA_WINDOW_HALF,
    REFERENCE_PKA,
)

logger = logging.getLogger("trizod.potenci")

# Pre-compute binary tuples and outer-product matrices for pKa calculation.
# Window size ranges from 0..5; used as sliding window in calc_pkas_from_seq.
_OUTER_MATRICES = []
_ALL_TUPLES = []
for _window_size in range(0, 6):
    _tuples = np.array(
        [
            [int(c) for c in np.binary_repr(i, _window_size)]
            for i in range(2**_window_size)
        ]
    )
    _OUTER_MATRICES.append(np.array([np.outer(c, c) for c in _tuples]))
    _ALL_TUPLES.append(_tuples)


def _titration_fraction(pH, pK, nH):
    """Henderson-Hasselbalch titration fraction (used by curve_fit)."""
    return 1.0 - 1.0 / ((10 ** (nH * (pK - pH))) + 1.0)


def _debye_huckel_W(r, Ion=0.1):
    """Electrostatic interaction energy (Debye-Hückel model)."""
    kappa = np.sqrt(Ion) / 3.08
    x = kappa.astype(np.float64) * r.astype(np.float64) / np.sqrt(6)
    prefactor = 332.286 * np.sqrt(6 / np.pi)
    erfc_x = erfc(x)
    sqrt_pi_x = np.sqrt(np.pi) * x

    er = DIELECTRIC_WATER * r
    # Log-space to avoid overflow: exp(x²)/(ε*r) = exp(x² - log(ε*r))
    exp_term = np.exp((x**2) - np.log(er))
    exp_term = np.nan_to_num(exp_term)
    return prefactor * ((1 / er) - np.nan_to_num(exp_term * sqrt_pi_x * erfc_x))


def _w_to_logp(x, T=293.15):
    """Convert interaction energy to log-probability shift."""
    return x * 4181.2 / (GAS_CONSTANT * T * np.log(10))


def _small_matrix_limits(res_idx, half_window, n_sites):
    """Get left/right bounds for a sliding window around a residue."""
    left = max(1, res_idx - half_window)
    right = min(left + 2 * half_window, n_sites)
    if right == n_sites:
        left = max(1, right - 2 * half_window)
    return (left, right)


def _small_matrix_pos(res_idx, half_window, n_sites):
    """Get position of a residue within its sliding window."""
    pos = half_window + 1
    if res_idx < half_window + 1:
        pos = res_idx
    if res_idx > n_sites - half_window:
        pos = min(n_sites, 2 * half_window + 1) - (n_sites - res_idx)
    return pos


def calc_pkas_from_seq(seq=None, T=293.15, Ion=0.1):
    """Iteratively predict pKa values for titratable residues in a sequence."""
    ph_values = np.arange(1.99, 10.01, 0.15)

    titratable_pos = np.array([i for i in range(len(seq)) if seq[i] in REFERENCE_PKA])
    N = titratable_pos.shape[0]
    I = np.diag(np.ones(N))
    sites = "".join([seq[i] for i in titratable_pos])
    neg_indices = np.array([i for i in range(len(sites)) if sites[i] in "DEYc"])
    l = np.array([abs(titratable_pos - titratable_pos[i]) for i in range(N)])
    d = MIN_CHARGE_DISTANCE + np.sqrt(l) * DISTANCE_SCALE

    w_energies = _debye_huckel_W(d, Ion)
    w_energies[I == 1] = 0

    log_interactions = _w_to_logp(w_energies, T) / 2

    base_charges = np.zeros(titratable_pos.shape[0])
    if len(neg_indices):
        base_charges[neg_indices] = -1

    pka_initial = np.array([REFERENCE_PKA[c] for c in sites])
    hill_initial = np.array([0.9 for c in sites])

    titration = np.zeros((N, len(ph_values)))

    window_size = min(2 * PKA_WINDOW_HALF + 1, len(titratable_pos))
    tuples = _ALL_TUPLES[window_size]
    outer_mats = _OUTER_MATRICES[window_size]
    g_matrix = [np.zeros((window_size, window_size)) for _ in range(len(ph_values))]

    for cycle in range(PKA_FIT_CYCLES):
        if cycle == 0:
            prev_fractions = np.array(
                [
                    [
                        _titration_fraction(
                            ph_values[p], pka_initial[i], hill_initial[i]
                        )
                        for i in range(N)
                    ]
                    for p in range(len(ph_values))
                ]
            )
        else:
            prev_fractions = titration.transpose()

        for res_idx in range(1, N + 1):
            (left, right) = _small_matrix_limits(res_idx, PKA_WINDOW_HALF, N)
            window_pos = _small_matrix_pos(res_idx, PKA_WINDOW_HALF, N)
            fraction = prev_fractions.copy()
            fraction[:, left - 1 : right] = 0
            charges = fraction + base_charges
            interaction_diag = 2 * (
                log_interactions * np.expand_dims(charges, axis=1)
            ).sum(axis=-1)
            interaction_diag = np.expand_dims(interaction_diag, 1) * I
            g_matrix_full = (
                log_interactions
                + interaction_diag
                + np.expand_dims(ph_values, (1, 2)) * I
                - np.diag(pka_initial)
            )
            g_matrix = g_matrix_full[:, left - 1 : right, left - 1 : right]

            E = 10 ** -(np.expand_dims(g_matrix, axis=1) * outer_mats).sum(axis=(2, 3))
            E_all = E.sum(axis=-1)
            E_sel = E[:, (tuples[:, window_pos - 1] == 1)].sum(axis=-1)
            titration[res_idx - 1] = E_sel / E_all
        fit_results = np.array(
            [
                curve_fit(
                    _titration_fraction,
                    ph_values,
                    titration[p],
                    [pka_initial[p], hill_initial[p]],
                    maxfev=5000,
                )[0]
                for p in range(len(pka_initial))
            ]
        )
        (pKs, nHs) = fit_results.transpose()

    result = {}
    for p, i in enumerate(titratable_pos):
        result[i - 1] = (pKs[p], nHs[p], seq[i])

    return result


# --------------- Data loading and module-level caches -----------------

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

AA_STANDARD = "ACDEFGHIKLMNPQRSTVWY"


def _load_csv(filename):
    """Load a CSV file from the data directory."""
    filepath = os.path.join(_DATA_DIR, filename)
    with open(filepath) as f:
        return list(csv.DictReader(f))


def _load_center_shifts():
    rows = _load_csv("centshifts.csv")
    atom_names = ["C", "CA", "CB", "N", "H", "HA", "HB"]
    result = {}
    for row in rows:
        aa = row["aa"]
        result[aa] = {}
        for atom in atom_names:
            val = row[atom]
            result[aa][atom] = None if val == "None" else float(val)
    return result


def _load_neighbor_corrs():
    result = {}
    for row in _load_csv("neicorrs.csv"):
        atom = row["atn"]
        aa = row["aa"]
        if aa not in result:
            result[aa] = {}
        result[aa][atom] = [float(row[f"c{j}"]) for j in range(4)]
    for row in _load_csv("termcorrs.csv"):
        atom = row["atn"]
        term = row["term"]
        val = float(row["value"])
        if term not in result:
            result[term] = {}
        if term == "n":
            result["n"][atom] = [None, None, None, val]
        elif term == "c":
            result["c"][atom] = [val, None, None, None]
    return result


def _load_temp_coeffs():
    rows = _load_csv("tempcoeffs.csv")
    headers = [k for k in rows[0] if k != "aa"]
    result = {}
    for atom in headers:
        result[atom] = {}
    for row in rows:
        aa = row["aa"]
        for atom in headers:
            result[atom][aa] = float(row[atom])
    return result


def _load_comb_devs():
    result = {}
    for row in _load_csv("combdevs.csv"):
        atom = row["atn"]
        if atom not in result:
            result[atom] = {}
        segment = row["segment"]
        key = (int(row["neipos"]), row["centgroup"], row["neigroup"])
        result[atom][segment] = key, float(row["value"])
    return result


def _load_ph_shifts():
    rows = _load_csv("phshifts.csv")
    result = {}
    for row in rows:
        res_name = row["resn"]
        atom = row["atn"]
        shift_delta = float(row["shd"])
        if res_name not in result:
            result[res_name] = {}
        result[res_name][atom] = shift_delta
        prev_nei = row["prev_nei"]
        succ_nei = row["succ_nei"]
        if prev_nei and succ_nei:
            for n, val in enumerate([prev_nei, succ_nei]):
                neighbor_key = res_name + "ps"[n]
                if neighbor_key not in result:
                    result[neighbor_key] = {}
                result[neighbor_key][atom] = float(val)
    return result


TEMP_CORRS = _load_temp_coeffs()
CENTER_SHIFTS = _load_center_shifts()
NEIGHBOR_CORRS = _load_neighbor_corrs()
COMB_CORRS = _load_comb_devs()
PH_SHIFTS = _load_ph_shifts()

# Pre-built amino acid → group lookup for pred_pent_shift()
_AA_GROUP = {}
for _gr, _label in zip(["G", "P", "FYW", "LIVMCA", "KR", "DE"], "GPra+-"):
    for _aa in _gr:
        _AA_GROUP[_aa] = _label

# Neighbor position offsets relative to center (index 2) of pentamer
_NEI_OFFSETS = [2, 1, -1, -2]


def pred_pent_shift(pentamer, atom):
    """Predict chemical shift for a 5-residue window and atom type."""
    center_aa = pentamer[2]
    shift = CENTER_SHIFTS[center_aa][atom]
    for i in range(4):
        neighbor_aa = pentamer[2 + _NEI_OFFSETS[i]]
        if neighbor_aa in NEIGHBOR_CORRS:
            shift += NEIGHBOR_CORRS[neighbor_aa][atom][i]
    group_str = "".join(_AA_GROUP.get(pentamer[i], "p") for i in range(5))
    center_group = group_str[2]
    for segment in COMB_CORRS[atom]:
        key, comb_value = COMB_CORRS[atom][segment]
        nei_pos, expected_center, nei_group = key
        if (
            expected_center == center_group
            and group_str[2 + nei_pos] == nei_group
            and ((center_group, nei_group) != ("p", "p") or pentamer[2] in "ST")
        ):
            # pp combination only used when center is Ser or Thr
            shift += comb_value
    return shift


def _get_temp_corr(aa, atom, temperature):
    """Get temperature correction for a residue/atom at given temperature."""
    return TEMP_CORRS[atom][aa] / 1000 * (temperature - 298)


def _read_csv_lines(filename):
    file = open(filename)
    buffer = file.readlines()
    file.close()
    for i in range(len(buffer)):
        buffer[i] = buffer[i][:-1].split(",")
    return buffer


def _write_csv_pka_output(pka_dict, seq, temperature, ion):
    seq = seq[: min(150, len(seq))]
    name = f"outpepKalc_{seq}_T{temperature:6.2f}_I{ion:4.2f}.csv"
    out = open(name, "w")
    out.write("Site,pKa value,pKa shift,Hill coefficient\n")
    for i in pka_dict:
        pKa, nH, res = pka_dict[i]
        res_key = res + str(i + 1)
        diff = pKa - REFERENCE_PKA[res]
        out.write(f"{res_key},{pKa:5.3f},{diff:5.3f},{nH:5.3f}\n")
    out.close()


def _read_csv_pka_output(seq, temperature, ion, name=None):
    seq = seq[: min(150, len(seq))]
    logger.debug(f"reading csv {name}")
    if name is None:
        name = f"outpepKalc_{seq}_T{temperature:6.2f}_I{ion:4.2f}.csv"
    try:
        open(name)
    except OSError:
        return None
    buf = _read_csv_lines(name)
    for line_num, data in enumerate(buf):  # noqa: B007
        if len(data) > 0 and data[0] == "Site":
            break
    pka_dict = {}
    for data in buf[line_num + 1 :]:
        res_key, pKa, diff, nH = data
        i = int(res_key[1:]) - 1
        res = res_key[0]
        pka_dict[i] = float(pKa), float(nH), res
    return pka_dict


def get_ph_corrs(seq, temperature, pH, ion, pka_csv_path=None):
    """Compute pH-dependent chemical shift corrections."""
    bb_atoms = ["C", "CA", "CB", "HA", "H", "N", "HB"]
    ph_shifts = PH_SHIFTS
    Ion = max(0.0001, ion)
    if not pka_csv_path:
        pka_dict = None
    else:
        pka_dict = _read_csv_pka_output(seq, temperature, ion, pka_csv_path)
    if pka_dict is None:
        pka_dict = calc_pkas_from_seq("n" + seq + "c", temperature, Ion)
        if pka_csv_path:
            _write_csv_pka_output(pka_dict, seq, temperature, ion)
    corrections = {}
    for i in pka_dict:
        logger.debug(
            f"pkares: {pka_dict[i][0]:6.3f} {pka_dict[i][1]:6.3f} {pka_dict[i][2]:1s}{i}"
        )
        pKa, nH, res = pka_dict[i]
        frac = _titration_fraction(pH, pKa, nH)
        frac_ref = _titration_fraction(7.0, REFERENCE_PKA[res], nH)
        if res in "nc":
            pass  # terminal residues: no correction (yet)
        else:
            for atom in bb_atoms:
                if atom not in corrections:
                    corrections[atom] = {}
                logger.debug(f"data: {atom}, {pKa}, {nH}, {res}, {i}, {atom}, {pH}")
                res_shifts = ph_shifts[res]
                try:
                    delta = res_shifts[atom]
                    jump = frac * delta
                    jump_ref = frac_ref * delta
                except KeyError:
                    logger.warning(f"no key: {res}, {i}, {atom}")
                    delta = 999
                    jump = 999
                    jump_ref = 999
                if delta < 99:
                    delta_jump = jump - jump_ref
                    if i not in corrections[atom]:
                        corrections[atom][i] = [res, delta_jump]
                    else:
                        corrections[atom][i][0] = res
                        corrections[atom][i][1] += delta_jump
                    logger.debug(
                        f"{atom:3s} {pKa:5.2f} {nH:6.4f} {res} {i:3d} {atom:5s} {jump:8.5f} {jump_ref:8.5f} {pH:4.2f}"
                    )
                    if res + "p" in ph_shifts and atom in ph_shifts[res + "p"]:
                        for n in range(2):
                            neighbor_idx = i + 2 * n - 1
                            neighbor_key = res + "ps"[n]
                            neighbor_delta = ph_shifts[neighbor_key][atom]
                            jump = frac * neighbor_delta
                            jump_ref = frac_ref * neighbor_delta
                            delta_jump = jump - jump_ref
                            if neighbor_idx not in corrections[atom]:
                                corrections[atom][neighbor_idx] = [None, delta_jump]
                            else:
                                corrections[atom][neighbor_idx][1] += delta_jump
    return corrections


def get_pred_shifts(
    seq, temperature, pH, ion, use_ph_corr=True, pka_csv_path=None, identifier=""
):
    """Predict random coil chemical shifts for a protein sequence.

    Returns dict[(residue_num, aa)] -> dict[atom_type -> shift_value].
    """
    bb_atoms = ["C", "CA", "CB", "HA", "H", "N", "HB"]
    ph_corrs = (
        get_ph_corrs(seq, temperature, pH, ion, pka_csv_path) if use_ph_corr else {}
    )
    shift_dict = {}
    for i in range(1, len(seq) - 1):
        if seq[i] in AA_STANDARD:
            triplet = seq[i - 1] + seq[i] + seq[i + 1]
            ph_corr = None
            shift_dict[(i + 1, seq[i])] = {}
            for atom in bb_atoms:
                if (triplet[1], atom) not in [("G", "CB"), ("G", "HB"), ("P", "H")]:
                    if i == 1:
                        pentamer = "n" + triplet + seq[i + 2]
                    elif i == len(seq) - 2:
                        pentamer = seq[i - 2] + triplet + "c"
                    else:
                        pentamer = seq[i - 2] + triplet + seq[i + 2]
                    shift = pred_pent_shift(pentamer, atom)
                    if shift is not None:
                        if atom != "HB":
                            shift += _get_temp_corr(triplet[1], atom, temperature)
                        if atom in ph_corrs and i in ph_corrs[atom]:
                            ph_data = ph_corrs[atom][i]
                            res = ph_data[0]
                            if seq[i] in "CDEHRKY" and res != seq[i]:
                                logger.warning(
                                    f"residue mismatch: {res},{seq[i]},{i},{ph_data},{atom}"
                                )
                            ph_corr = ph_data[1]
                            if abs(ph_corr) < 9.9:
                                shift -= ph_corr
                        shift_dict[(i + 1, seq[i])][atom] = shift
                        logger.debug(
                            f"predictedshift: {identifier:5s} {i:3d} {seq[i]:1s} {atom:2s} {shift:8.4f}"
                            + " "
                            + str(ph_corr)
                        )
    return shift_dict
