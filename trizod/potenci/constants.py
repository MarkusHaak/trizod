"""Constants and data tables for the POTENCI algorithm.

This module contains physical constants, pKa values, and chemical shift correction tables
used in the POTENCI (Prediction Of The chemical shift ENvironment Induced)
algorithm for predicting random coil chemical shifts.

Data is loaded from CSV files in the data/ subdirectory. The original data comes from
the POTENCI algorithm published by Frans A. A. Mulder's group.

References:
    Nielsen, J. T., & Mulder, F. A. (2018). POTENCI: prediction of temperature,
    neighbor and pH-corrected chemical shifts for intrinsically disordered proteins.
    Journal of Biomolecular NMR, 70(3), 141-165.
"""

from __future__ import annotations

import csv
import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

ShiftDict = dict[str, dict[str, float | None]]
CorrectionDict = dict[str, dict[str, list[float | None]]]
CombinationDict = dict[str, dict[str, tuple[tuple[int, str, str], float]]]
TempCoeffDict = dict[str, dict[str, float]]
PKDict = dict[str, float]


# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================


@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants for POTENCI calculations."""

    gas_constant: float = 8.314472  # J/(mol*K)
    dielectric_constant: float = 79.0  # Relative permittivity
    distance_param_a: float = 5.0  # Angstroms
    distance_param_b: float = 7.5  # Angstroms
    cutoff: int = 2  # Number of residues for local calculations
    n_cycles: int = 5  # Number of iterations for pKa calculations


# Global instance
PHYSICAL_CONSTANTS = PhysicalConstants()


# ============================================================================
# CHEMICAL CONSTANTS
# ============================================================================

# Titratable groups pKa values (at 298K, ionic strength 0.1M)
PK0: PKDict = {
    "n": 8.23,  # N-terminus
    "D": 3.86,  # Aspartate
    "E": 4.34,  # Glutamate
    "H": 6.45,  # Histidine
    "C": 8.49,  # Cysteine
    "K": 10.34,  # Lysine
    "R": 13.9,  # Arginine
    "Y": 9.76,  # Tyrosine
    "c": 3.55,  # C-terminus
}

# Standard amino acids (one-letter codes)
AA_STANDARD = "ACDEFGHIKLMNPQRSTVWY"


# ============================================================================
# PRE-COMPUTED MATRICES
# ============================================================================


def _compute_outer_matrices() -> tuple[list[npt.NDArray[np.float64]], list[npt.NDArray[np.int_]]]:
    """Compute outer product matrices for efficient pKa calculations.

    Returns:
        Tuple of (outer_matrices, alltuples) lists indexed by matrix size.
    """
    outer_matrices_list: list[npt.NDArray[np.float64]] = []
    alltuples_list: list[npt.NDArray[np.int_]] = []

    for small_n in range(0, 6):
        alltuples = np.array(
            [[int(c) for c in np.binary_repr(i, small_n)] for i in range(2**small_n)],
            dtype=np.int_,
        )
        outerm = np.array([np.outer(c, c) for c in alltuples], dtype=np.float64)
        outer_matrices_list.append(outerm)
        alltuples_list.append(alltuples)

    return outer_matrices_list, alltuples_list


outer_matrices, alltuples_ = _compute_outer_matrices()


# ============================================================================
# DATA FILE PATHS
# ============================================================================

_DATA_DIR = Path(__file__).parent / "data"


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================


def _safe_float(value: str) -> float | None:
    """Safely convert string to float, returning None for empty strings.

    Args:
        value: String value to convert.

    Returns:
        Float value or None if empty/invalid.
    """
    if not value or value.lower() in ("none", "na", "nan", ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


@functools.lru_cache(maxsize=1)
def load_central_shifts() -> ShiftDict:
    """Load central residue chemical shift corrections from CSV.

    Returns:
        Dictionary mapping amino acid to atom type to shift value.
        Format: {aa: {atom: shift}}
    """
    data: ShiftDict = {}
    csv_path = _DATA_DIR / "tablecent.csv"

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            aa = row["aa"]
            data[aa] = {}
            for atom in ["C", "CA", "CB", "N", "H", "HA", "HB"]:
                data[aa][atom] = _safe_float(row[atom])

    return data


@functools.lru_cache(maxsize=1)
def load_neighbor_corrections() -> CorrectionDict:
    """Load neighbor residue chemical shift corrections from CSV.

    Returns:
        Dictionary mapping amino acid to atom type to list of corrections.
        Format: {aa: {atom: [corr1, corr2, corr3, corr4]}}
    """
    data: CorrectionDict = {}
    csv_path = _DATA_DIR / "tablenei.csv"

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            atom = row["atom"]
            aa = row["aa"]

            if aa not in data:
                data[aa] = {}

            data[aa][atom] = [
                _safe_float(row["corr1"]),
                _safe_float(row["corr2"]),
                _safe_float(row["corr3"]),
                _safe_float(row["corr4"]),
            ]

    return data


@functools.lru_cache(maxsize=1)
def load_terminal_corrections() -> CorrectionDict:
    """Load terminal correction values from CSV.

    Returns:
        Dictionary with 'n' and 'c' terminal corrections.
        Format: {'n'/'c': {atom: [None, None, None, value]}}
    """
    data: CorrectionDict = {}
    csv_path = _DATA_DIR / "tabletermcorrs.csv"

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            atom = row["atom"]
            term = row["term"]
            value = _safe_float(row["corr"])

            if term not in data:
                data[term] = {}

            if term == "n":
                data["n"][atom] = [None, None, None, value]
            elif term == "c":
                data["c"][atom] = [value, None, None, None]

    return data


@functools.lru_cache(maxsize=1)
def load_temperature_coefficients() -> TempCoeffDict:
    """Load temperature correction coefficients from CSV.

    Returns:
        Dictionary mapping atom type to amino acid to coefficient.
        Format: {atom: {aa: coeff}}
    """
    data: TempCoeffDict = {}
    csv_path = _DATA_DIR / "tabletempk.csv"

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        atoms = ["CA", "CB", "C", "N", "H", "HA"]

        for atom in atoms:
            data[atom] = {}

        for row in reader:
            aa = row["aa"]
            for atom in atoms:
                value = _safe_float(row[atom])
                if value is not None:
                    data[atom][aa] = value

    return data


@functools.lru_cache(maxsize=1)
def load_combinatorial_deviations() -> CombinationDict:
    """Load combinatorial deviation corrections from CSV.

    Returns:
        Dictionary mapping atom type to segment to (key tuple, correction).
        Format: {atom: {segment: ((position, cent_group, nei_group), value)}}
    """
    data: CombinationDict = {}
    csv_path = _DATA_DIR / "tablecombdevs.csv"

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            atom = row["atom"]
            if atom not in data:
                data[atom] = {}

            position = int(row["neipos"])
            cent_group = row["centgroup"]
            nei_group = row["neigroup"]
            segment = row["segment"]
            correction = float(row["val1"])  # Using val1 as the correction value

            key = (position, cent_group, nei_group)
            data[atom][segment] = (key, correction)

    return data


@functools.lru_cache(maxsize=1)
def load_ph_shifts() -> ShiftDict:
    """Load pH-dependent chemical shift changes from CSV.

    Returns:
        Dictionary mapping residue (with 'p'/'s' suffixes for neighbors) to atom to shift delta.
        Format: {residue: {atom: delta}}
    """
    data: ShiftDict = {}
    csv_path = _DATA_DIR / "tablephshifts.csv"

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            residue = row["residue"]
            atom = row["atom"]
            # val3 is the shift delta (difference between protonated and deprotonated)
            shift_delta = _safe_float(row["val3"])

            if residue not in data:
                data[residue] = {}

            data[residue][atom] = shift_delta

            # Handle neighbor data if present (val4 and val5)
            neighbor_prev = _safe_float(row.get("val4", ""))
            neighbor_next = _safe_float(row.get("val5", ""))

            if neighbor_prev is not None:
                residue_prev = residue + "p"
                if residue_prev not in data:
                    data[residue_prev] = {}
                data[residue_prev][atom] = neighbor_prev

            if neighbor_next is not None:
                residue_next = residue + "s"
                if residue_next not in data:
                    data[residue_next] = {}
                data[residue_next][atom] = neighbor_next

    return data


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Physical constants
    "PhysicalConstants",
    "PHYSICAL_CONSTANTS",
    # Chemical constants
    "PK0",
    "AA_STANDARD",
    # Pre-computed matrices
    "outer_matrices",
    "alltuples_",
    # Data loading functions
    "load_central_shifts",
    "load_neighbor_corrections",
    "load_terminal_corrections",
    "load_temperature_coefficients",
    "load_combinatorial_deviations",
    "load_ph_shifts",
]
