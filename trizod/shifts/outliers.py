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

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from trizod.constants import AA3TO1
from trizod.io.atomic import atomic_write

#: Thresholds in raw-MAD units, per backbone heavy atom (Sandro Kuppel's).
MAD_K: dict[str, float] = {"C": 8.0, "CA": 8.0, "N": 8.0, "CB": 12.0}

#: MAD -> sigma for a normal distribution; documented, never applied.
MAD_TO_SIGMA = 1.4826

#: ``mad_z`` is rounded to this many decimals. 0.01 MAD is ~0.016 ppm for CA,
#: finer than shifts are deposited, and it shrinks the column from 31 to 7 MB.
MAD_Z_DECIMALS = 2

#: The committed reference table.
MAD_REFERENCE_PATH = Path(__file__).with_name("mad_reference.csv")

REFERENCE_COLUMNS = ["comp_id", "atom_id", "n", "median_ppm", "mad_ppm"]

#: Every group a complete reference covers: 20 residues x 4 atoms, minus GLY CB.
EXPECTED_GROUPS = frozenset(
    (aa, atom) for aa in AA3TO1 for atom in MAD_K if (aa, atom) != ("GLY", "CB")
)

_GROUP = ["comp_id", "atom_id"]


@dataclass(frozen=True, eq=False)
class MadReference:
    """A median/MAD table (:data:`REFERENCE_COLUMNS`) and the build it came from.

    One object serves both the flag and its documentation: the same instance is
    passed to :func:`annotate_mad` and to the shift-Parquet writer, which embeds
    :meth:`metadata`, so a file can never describe a different reference than
    the one its ``mad_z`` was computed against. Treat ``table`` as read-only:
    :attr:`stats` is cached from it on first use.
    """

    table: pd.DataFrame
    source: str = ""

    @cached_property
    def stats(self) -> dict[tuple[str, str], tuple[float, float]]:
        """``(comp_id, atom_id) -> (median_ppm, mad_ppm)``, for per-row lookup."""
        t = self.table
        return {
            (comp, atom): (median, mad)
            for comp, atom, median, mad in zip(
                t["comp_id"], t["atom_id"], t["median_ppm"], t["mad_ppm"]
            )
        }

    def missing_groups(self) -> list[tuple[str, str]]:
        """Canonical ``(comp_id, atom_id)`` groups this reference cannot assess."""
        return sorted(EXPECTED_GROUPS - set(self.stats))

    def metadata(self) -> dict[bytes, bytes]:
        """The reference as Parquet table metadata, keyed ``"<comp_id> <atom_id>"``."""
        groups = {
            f"{comp} {atom}": {
                "n": int(n),
                "median_ppm": float(median),
                "mad_ppm": float(mad),
            }
            for comp, atom, n, median, mad in self.table[REFERENCE_COLUMNS].itertuples(
                index=False
            )
        }
        return {
            b"mad_reference": json.dumps(groups, sort_keys=True).encode(),
            b"mad_reference_source": self.source.encode(),
        }


def _assessable(df: pd.DataFrame) -> np.ndarray:
    """Rows the rule applies to: a flagged atom, and not the N-terminal amine."""
    atom = df["atom_id"]
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
    sel = df.loc[_assessable(df), [*_GROUP, "val_ppm"]]
    median = sel.groupby(_GROUP, observed=True)["val_ppm"].transform("median")
    return (
        sel.assign(deviation=(sel["val_ppm"] - median).abs())
        .groupby(_GROUP, observed=True, sort=True)
        .agg(
            n=("val_ppm", "count"),
            median_ppm=("val_ppm", "median"),
            mad_ppm=("deviation", "median"),
        )
        .reset_index()[REFERENCE_COLUMNS]
    )


def annotate_mad(df: pd.DataFrame, reference: MadReference) -> pd.DataFrame:
    """Return a copy of ``df`` with ``mad_z`` and ``is_far_outlier`` appended.

    ``mad_z`` is float32 rounded to :data:`MAD_Z_DECIMALS`, NaN where not
    assessed; ``is_far_outlier`` is a nullable boolean, NA where not assessed,
    computed from the rounded ``mad_z``. Row order is preserved.
    """
    df = df.copy()
    missing = (np.nan, np.nan)
    stats = np.array(
        [
            reference.stats.get(key, missing)
            for key in zip(df["comp_id"], df["atom_id"])
        ],
        dtype=float,
    ).reshape(-1, 2)
    median, mad = stats[:, 0], stats[:, 1]
    assessed = _assessable(df) & (mad > 0)  # NaN (no reference) compares False

    with np.errstate(divide="ignore", invalid="ignore"):
        z = (df["val_ppm"].to_numpy(dtype=float) - median) / mad
    z = np.round(np.where(assessed, z, np.nan), MAD_Z_DECIMALS).astype(np.float32)
    limit = df["atom_id"].map(MAD_K).to_numpy(dtype=float)
    flag = pd.array(np.abs(z) > limit, dtype="boolean")
    flag[~assessed] = pd.NA

    df["mad_z"] = z
    df["is_far_outlier"] = flag
    return df


def write_mad_reference(
    table: pd.DataFrame, path: Path = MAD_REFERENCE_PATH, *, source: str
) -> Path:
    """Write a reference table as CSV under a ``#`` header naming its ``source``."""
    path = Path(path)
    header = [
        "# TriZOD far-outlier reference: median and raw (unscaled) MAD of the "
        "deposited val_ppm per (comp_id, atom_id)",
        f"# source: {source}",
        "# rule: trizod/shifts/outliers.py; written by scripts/build_mad_reference.py; "
        "the N of residue 1 (free amine) is excluded",
    ]
    table = table[REFERENCE_COLUMNS].round({"median_ppm": 6, "mad_ppm": 6})
    with atomic_write(path, "w") as fh:
        fh.write("\n".join(header) + "\n")
        table.to_csv(fh, index=False)
    return path


def load_mad_reference(path: Path = MAD_REFERENCE_PATH) -> MadReference:
    """Read a reference CSV written by :func:`write_mad_reference`."""
    path = Path(path)
    table = pd.read_csv(
        path,
        comment="#",
        usecols=REFERENCE_COLUMNS,
        dtype={"comp_id": str, "atom_id": str, "n": "int64"},
    )[REFERENCE_COLUMNS]
    if table.duplicated(_GROUP).any():
        raise ValueError(f"{path} lists a (comp_id, atom_id) group twice")
    with path.open() as fh:
        source = next(
            (
                line.split(":", 1)[1].strip()
                for line in fh
                if line.startswith("# source:")
            ),
            "",
        )
    return MadReference(table, source)
