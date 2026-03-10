"""POTENCI — random coil chemical shift prediction for proteins.

Adapted from https://github.com/protein-nmr/POTENCI (commit 17dd2e6).
Original author: fmulder@chem.au.dk
Adapted by: markus.haak@tum.de & tobias.senoner@tum.de

Public API:
    get_pred_shifts(seq, temperature, pH, ion, ...) -> dict
"""

import csv
import logging
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit
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

# ── Constants ────────────────────────────────────────────────────────────

BB_ATOMS = ["C", "CA", "CB", "HA", "H", "N", "HB"]

AA_STANDARD = "ACDEFGHIKLMNPQRSTVWY"

# Atom/residue combinations that have no prediction
_SKIP_ATOM_PAIRS = {("G", "CB"), ("G", "HB"), ("P", "H")}

# ── Data loading ─────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_csv(filename):
    """Load a CSV file from the data directory."""
    filepath = _DATA_DIR / filename
    with filepath.open() as f:
        return list(csv.DictReader(f))


def _load_center_shifts():
    result = {}
    for row in _load_csv("centshifts.csv"):
        aa = row["aa"]
        result[aa] = {}
        for atom in BB_ATOMS:
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
            for nei_side, nei_val in enumerate([prev_nei, succ_nei]):
                neighbor_key = res_name + "ps"[nei_side]
                if neighbor_key not in result:
                    result[neighbor_key] = {}
                result[neighbor_key][atom] = float(nei_val)
    return result


# Module-level caches (loaded once at import time)
TEMP_CORRS = _load_temp_coeffs()
CENTER_SHIFTS = _load_center_shifts()
NEIGHBOR_CORRS = _load_neighbor_corrs()
COMB_CORRS = _load_comb_devs()
PH_SHIFTS = _load_ph_shifts()

# Amino acid → group lookup for combination corrections in pred_pent_shift()
_AA_GROUP = {}
for _group_aas, _group_label in zip(["G", "P", "FYW", "LIVMCA", "KR", "DE"], "GPra+-"):
    for _aa in _group_aas:
        _AA_GROUP[_aa] = _group_label

# Neighbor position offsets relative to center (index 2) of pentamer
_NEI_OFFSETS = [2, 1, -1, -2]

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


# ── Shift prediction helpers ─────────────────────────────────────────────


def _get_temp_corr(aa, atom, temperature):
    """Get temperature correction for a residue/atom at given temperature."""
    return TEMP_CORRS[atom][aa] / 1000 * (temperature - 298)


def pred_pent_shift(pentamer, atom):
    """Predict chemical shift for a 5-residue window and atom type."""
    center_aa = pentamer[2]
    shift = CENTER_SHIFTS[center_aa][atom]
    for offset_idx in range(4):
        neighbor_aa = pentamer[2 + _NEI_OFFSETS[offset_idx]]
        if neighbor_aa in NEIGHBOR_CORRS:
            shift += NEIGHBOR_CORRS[neighbor_aa][atom][offset_idx]
    group_str = "".join(_AA_GROUP.get(pentamer[pos], "p") for pos in range(5))
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


def _build_pentamer(seq, pos):
    """Build the 5-residue context window for position pos (1-based interior)."""
    triplet = seq[pos - 1] + seq[pos] + seq[pos + 1]
    if pos == 1 and pos == len(seq) - 2:
        return triplet, "n" + triplet + "c"
    elif pos == 1:
        return triplet, "n" + triplet + seq[pos + 2]
    elif pos == len(seq) - 2:
        return triplet, seq[pos - 2] + triplet + "c"
    else:
        return triplet, seq[pos - 2] + triplet + seq[pos + 2]


# ── pKa prediction (Debye-Hückel + iterative fitting) ───────────────────


def _titration_fraction(pH, pK, nH):
    """Henderson-Hasselbalch titration fraction (used by curve_fit)."""
    with np.errstate(over="ignore"):
        return 1.0 - 1.0 / ((10 ** (nH * (pK - pH))) + 1.0)


def _debye_huckel_W(distances, Ion=0.1):
    """Electrostatic interaction energy (Debye-Hückel model)."""
    kappa = np.sqrt(Ion) / 3.08
    scaled_dist = kappa.astype(np.float64) * distances.astype(np.float64) / np.sqrt(6)
    prefactor = 332.286 * np.sqrt(6 / np.pi)
    erfc_val = erfc(scaled_dist)
    sqrt_pi_x = np.sqrt(np.pi) * scaled_dist

    dielectric_r = DIELECTRIC_WATER * distances
    # Log-space to avoid overflow: exp(x²)/(ε*r) = exp(x² - log(ε*r))
    exp_term = np.exp((scaled_dist**2) - np.log(dielectric_r))
    exp_term = np.nan_to_num(exp_term)
    return prefactor * (
        (1 / dielectric_r) - np.nan_to_num(exp_term * sqrt_pi_x * erfc_val)
    )


def _w_to_logp(energy, T=293.15):
    """Convert interaction energy to log-probability shift."""
    return energy * 4181.2 / (GAS_CONSTANT * T * np.log(10))


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
    n_sites = titratable_pos.shape[0]
    identity = np.diag(np.ones(n_sites))
    sites = "".join([seq[i] for i in titratable_pos])
    neg_indices = np.array([i for i in range(len(sites)) if sites[i] in "DEYc"])
    seq_separations = np.array(
        [abs(titratable_pos - titratable_pos[i]) for i in range(n_sites)]
    )
    distances = MIN_CHARGE_DISTANCE + np.sqrt(seq_separations) * DISTANCE_SCALE

    w_energies = _debye_huckel_W(distances, Ion)
    w_energies[identity == 1] = 0

    log_interactions = _w_to_logp(w_energies, T) / 2

    base_charges = np.zeros(n_sites)
    if len(neg_indices):
        base_charges[neg_indices] = -1

    pka_initial = np.array([REFERENCE_PKA[residue] for residue in sites])
    hill_initial = np.array([0.9 for _ in sites])

    titration = np.zeros((n_sites, len(ph_values)))

    window_size = min(2 * PKA_WINDOW_HALF + 1, len(titratable_pos))
    tuples = _ALL_TUPLES[window_size]
    outer_mats = _OUTER_MATRICES[window_size]

    for _cycle in range(PKA_FIT_CYCLES):
        if _cycle == 0:
            prev_fractions = np.array(
                [
                    [
                        _titration_fraction(
                            ph_values[ph_idx],
                            pka_initial[site_idx],
                            hill_initial[site_idx],
                        )
                        for site_idx in range(n_sites)
                    ]
                    for ph_idx in range(len(ph_values))
                ]
            )
        else:
            prev_fractions = titration.transpose()

        for res_idx in range(1, n_sites + 1):
            (left, right) = _small_matrix_limits(res_idx, PKA_WINDOW_HALF, n_sites)
            window_pos = _small_matrix_pos(res_idx, PKA_WINDOW_HALF, n_sites)
            fraction = prev_fractions.copy()
            fraction[:, left - 1 : right] = 0
            charges = fraction + base_charges
            interaction_diag = 2 * (
                log_interactions * np.expand_dims(charges, axis=1)
            ).sum(axis=-1)
            interaction_diag = np.expand_dims(interaction_diag, 1) * identity
            g_matrix_full = (
                log_interactions
                + interaction_diag
                + np.expand_dims(ph_values, (1, 2)) * identity
                - np.diag(pka_initial)
            )
            g_matrix = g_matrix_full[:, left - 1 : right, left - 1 : right]

            boltzmann = 10 ** -(np.expand_dims(g_matrix, axis=1) * outer_mats).sum(
                axis=(2, 3)
            )
            boltzmann_total = boltzmann.sum(axis=-1)
            boltzmann_selected = boltzmann[:, (tuples[:, window_pos - 1] == 1)].sum(
                axis=-1
            )
            titration[res_idx - 1] = boltzmann_selected / boltzmann_total
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            fit_results = np.array(
                [
                    curve_fit(
                        _titration_fraction,
                        ph_values,
                        titration[site_idx],
                        [pka_initial[site_idx], hill_initial[site_idx]],
                        maxfev=5000,
                    )[0]
                    for site_idx in range(len(pka_initial))
                ]
            )
        (pKs, nHs) = fit_results.transpose()

    result = {}
    for site_idx, seq_pos in enumerate(titratable_pos):
        result[seq_pos - 1] = (pKs[site_idx], nHs[site_idx], seq[seq_pos])

    return result


# ── pH corrections ───────────────────────────────────────────────────────


def _get_ph_corrs(seq, temperature, pH, ion):
    """Compute pH-dependent chemical shift corrections.

    Returns dict[atom_type -> dict[residue_index -> (residue_name, delta_shift)]].
    """
    Ion = max(0.0001, ion)
    pka_dict = calc_pkas_from_seq("n" + seq + "c", temperature, Ion)

    corrections = {}
    for res_idx in pka_dict:
        pKa, nH, res = pka_dict[res_idx]
        if res in "nc":
            continue  # terminal residues: no correction (yet)

        frac = _titration_fraction(pH, pKa, nH)
        frac_ref = _titration_fraction(7.0, REFERENCE_PKA[res], nH)

        for atom in BB_ATOMS:
            if atom not in corrections:
                corrections[atom] = {}
            if atom not in PH_SHIFTS[res]:
                continue
            delta = PH_SHIFTS[res][atom]
            delta_jump = frac * delta - frac_ref * delta
            if res_idx not in corrections[atom]:
                corrections[atom][res_idx] = [res, delta_jump]
            else:
                corrections[atom][res_idx][0] = res
                corrections[atom][res_idx][1] += delta_jump

            # Neighbor effects (preceding and succeeding residues)
            if res + "p" in PH_SHIFTS and atom in PH_SHIFTS[res + "p"]:
                for nei_side in range(2):
                    neighbor_idx = res_idx + 2 * nei_side - 1
                    neighbor_key = res + "ps"[nei_side]
                    neighbor_delta = PH_SHIFTS[neighbor_key][atom]
                    neighbor_jump = frac * neighbor_delta - frac_ref * neighbor_delta
                    if neighbor_idx not in corrections[atom]:
                        corrections[atom][neighbor_idx] = [None, neighbor_jump]
                    else:
                        corrections[atom][neighbor_idx][1] += neighbor_jump
    return corrections


# ── Public API ───────────────────────────────────────────────────────────


def get_pred_shifts(seq, temperature, pH, ion, use_ph_corr=True, **kwargs):
    """Predict random coil chemical shifts for a protein sequence.

    Parameters:
        seq: protein sequence (single-letter amino acids)
        temperature: in Kelvin (e.g. 298.0)
        pH: sample pH (e.g. 7.0)
        ion: ionic strength in M (e.g. 0.1)
        use_ph_corr: apply pH-dependent corrections (set False for pH=7.0)

    Returns:
        dict[(residue_num, aa_letter)] -> dict[atom_type -> shift_value]
    """
    ph_corrs = _get_ph_corrs(seq, temperature, pH, ion) if use_ph_corr else {}

    shift_dict = {}
    for pos in range(1, len(seq) - 1):
        if seq[pos] not in AA_STANDARD:
            continue
        triplet, pentamer = _build_pentamer(seq, pos)
        shift_dict[(pos + 1, seq[pos])] = {}
        for atom in BB_ATOMS:
            if (triplet[1], atom) in _SKIP_ATOM_PAIRS:
                continue
            shift = pred_pent_shift(pentamer, atom)
            if shift is None:
                continue
            if atom != "HB":
                shift += _get_temp_corr(triplet[1], atom, temperature)
            if atom in ph_corrs and pos in ph_corrs[atom]:
                ph_data = ph_corrs[atom][pos]
                corr_res = ph_data[0]
                if seq[pos] in "CDEHRKY" and corr_res != seq[pos]:
                    logger.warning(
                        f"residue mismatch: {corr_res},{seq[pos]},{pos},{ph_data},{atom}"
                    )
                ph_corr = ph_data[1]
                if abs(ph_corr) < 9.9:
                    shift -= ph_corr
            shift_dict[(pos + 1, seq[pos])][atom] = shift
    return shift_dict
