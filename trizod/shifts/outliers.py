"""Flag far-outlier backbone shifts against a frozen per-residue x atom reference.

Sandro Kuppel found deposition errors -- shifts hundreds of ppm off, spectra
referenced to the wrong nucleus -- skewing the train/validation/test losses of
a shift predictor, and removed them with a median-absolute-deviation filter per
residue type x atom: k = 8 for C, CA and N, k = 12 for CB, no filter for
protons, thresholds chosen from the histograms. The release ships that rule as
two columns of ``trizod_shifts.parquet`` instead of applying it:

``mad_z``
    ``(val_ppm - median) / MAD`` of the shift's ``(comp_id, atom_id)`` group in
    the frozen reference, rounded to 0.01 and stored as float32. The MAD is
    **raw**, not scaled by 1.4826, so divide by 1.4826 for a z-score in units of
    a normal sigma.
``is_far_outlier``
    ``|mad_z| > MAD_K[atom_id]``, evaluated on the rounded value, so the
    published column reproduces the flag exactly.

Both are NULL -- "not assessed", never False -- for protons, side chains, the N
of residue 1 (a free amine near 35-40 ppm, which is real physics and not part
of the amide distribution), and any group missing from the reference.

Raw ``val_ppm`` is assessed, not ``val_corrected_ppm``: the gross errors this
catches are tens to hundreds of ppm, far beyond any referencing offset, and the
raw value is the one every row has.

Nothing is dropped and the scoring path never reads these columns (it already
caps a residue's per-atom deviation at 4 and filters ambiguity codes), so the
flag costs no rescore. The reference lives in this package as
``mad_reference.csv``, generated once by ``scripts/build_mad_reference.py`` and
committed, so the flag does not drift with the next BMRB snapshot. Keep these
constants out of ``trizod/constants.py``: that file is part of the scoring cache
key, and changing it would force a full rescore for a flag scoring never uses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: Thresholds in raw-MAD units, per backbone heavy atom (Sandro Kuppel's).
MAD_K: dict[str, float] = {"C": 8.0, "CA": 8.0, "N": 8.0, "CB": 12.0}

#: MAD -> sigma for a normal distribution. Not applied; documented so a consumer
#: who wants sigma units can convert ``mad_z``.
MAD_TO_SIGMA = 1.4826

#: ``mad_z`` is rounded to this many decimals. 0.01 MAD is ~0.016 ppm for CA,
#: finer than shifts are deposited, and it shrinks the column from 31 to 7 MB.
MAD_Z_DECIMALS = 2

#: The committed reference table.
MAD_REFERENCE_PATH = Path(__file__).with_name("mad_reference.csv")

REFERENCE_COLUMNS = ["comp_id", "atom_id", "n", "median_ppm", "mad_ppm"]

_GROUP = ["comp_id", "atom_id"]


def _assessable(df: pd.DataFrame) -> np.ndarray:
    """Rows the rule applies to: a flagged atom, and not the N-terminal amine."""
    atom = df["atom_id"].astype(str)
    n_terminal_amine = (df["seq_id"] == 1) & (atom == "N")
    return (atom.isin(list(MAD_K)) & ~n_terminal_amine).to_numpy()


def compute_mad_reference(df: pd.DataFrame) -> pd.DataFrame:
    """Median and raw MAD of ``val_ppm`` per ``(comp_id, atom_id)``.

    Args:
        df: shift rows with ``seq_id``, ``comp_id``, ``atom_id`` and ``val_ppm``
            (any superset, e.g. the whole released shift table).

    Returns:
        One row per group, columns :data:`REFERENCE_COLUMNS`, sorted by group.
    """
    sel = df.loc[_assessable(df), [*_GROUP, "val_ppm"]].copy()
    for col in _GROUP:
        sel[col] = sel[col].astype(str)
    groups = sel.groupby(_GROUP, sort=True)["val_ppm"]
    deviation = (sel["val_ppm"] - groups.transform("median")).abs()
    out = pd.DataFrame(
        {
            "n": groups.size(),
            "median_ppm": groups.median(),
            "mad_ppm": deviation.groupby([sel[c] for c in _GROUP]).median(),
        }
    ).reset_index()
    return out[REFERENCE_COLUMNS]


def annotate_mad(df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``mad_z`` and ``is_far_outlier`` appended.

    ``mad_z`` is float32 rounded to :data:`MAD_Z_DECIMALS`, NaN where not
    assessed; ``is_far_outlier`` is a nullable boolean, NA where not assessed,
    computed from the rounded ``mad_z``. Row order is preserved.
    """
    df = df.copy()
    keys = pd.DataFrame({c: df[c].astype(str).to_numpy() for c in _GROUP})
    stats = keys.merge(
        reference[[*_GROUP, "median_ppm", "mad_ppm"]],
        how="left",
        on=_GROUP,
        validate="many_to_one",
    )
    median = stats["median_ppm"].to_numpy(dtype=float)
    mad = stats["mad_ppm"].to_numpy(dtype=float)
    assessed = _assessable(df) & (mad > 0)  # NaN (no reference) compares False

    with np.errstate(divide="ignore", invalid="ignore"):
        z = (df["val_ppm"].to_numpy(dtype=float) - median) / mad
    z = np.round(np.where(assessed, z, np.nan), MAD_Z_DECIMALS).astype(np.float32)
    limit = keys["atom_id"].map(MAD_K).to_numpy(dtype=float)
    flag = pd.array(np.abs(z) > limit, dtype="boolean")
    flag[~assessed] = pd.NA

    df["mad_z"] = z
    df["is_far_outlier"] = flag
    return df


def write_mad_reference(
    reference: pd.DataFrame, path: Path = MAD_REFERENCE_PATH, *, source: str
) -> Path:
    """Write the reference as CSV under a ``#`` header naming its ``source``."""
    path = Path(path)
    header = [
        "# TriZOD far-outlier reference: median and raw (unscaled) MAD of the "
        "deposited val_ppm per (comp_id, atom_id)",
        f"# source: {source}",
        "# rule: trizod/shifts/outliers.py; written by scripts/build_mad_reference.py; "
        "the N of residue 1 (free amine) is excluded",
    ]
    table = reference[REFERENCE_COLUMNS].round({"median_ppm": 6, "mad_ppm": 6})
    with path.open("w") as fh:
        fh.write("\n".join(header) + "\n")
        table.to_csv(fh, index=False)
    return path


def load_mad_reference(path: Path = MAD_REFERENCE_PATH) -> pd.DataFrame:
    """Read a reference CSV; its ``# source:`` line lands in ``attrs["source"]``."""
    path = Path(path)
    ref = pd.read_csv(
        path,
        comment="#",
        dtype={"comp_id": str, "atom_id": str, "n": "int64"},
    )
    missing = set(REFERENCE_COLUMNS) - set(ref.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    if ref.duplicated(_GROUP).any():
        raise ValueError(f"{path} lists a (comp_id, atom_id) group twice")
    ref = ref[REFERENCE_COLUMNS]
    with path.open() as fh:
        for line in fh:
            if line.startswith("# source:"):
                ref.attrs["source"] = line.split(":", 1)[1].strip()
                break
    return ref
