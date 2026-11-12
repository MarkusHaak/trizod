"""Repeated measurements in the release: how often one exact sequence was measured.

The BMRB holds many entries of the same protein. Counted over ALL chains of the release
(bound ones included, one row per chain of ``trizod_dataset.parquet``), an *exact
sequence* is measured in ``n_entries`` distinct BMRB entries. These counts feed Fig. 2d
(the repeat histogram), R2 (\\nSeqsMultiEntry, \\nChainsMultiEntry, \\nEntriesMaxPerSeq) and
R7 (\\nChainsASynExact, \\nEntriesASynExact), all computed by
``scripts/manuscript/make_numbers.py`` with the functions below.

* :func:`sequence_table` -- one row per exact sequence: entries, chains, non-bound chains
  and entries, a readable name.
* :func:`repeat_histogram` -- the Fig. 2d data: a tidy histogram (entries per sequence ->
  number of sequences and their chains) and the most-measured sequences with names.
* :func:`repeat_stats` -- the scalar numbers the text quotes.
* :func:`ubq_most_entries_ordered` -- the check behind the ubiquitin example of Fig. 3c:
  among sequences whose non-bound chains are mostly ordered (median chain-mean G below
  0.2), ubiquitin has the most non-bound entries.

Names come from ``entity_name``; depositors often write a generic name ('entity_1'), so the
most frequent non-generic name is used, and the two sequences the text names get fixed
labels (``LABELS``).
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style_paper as sp  # noqa: E402

# Human ubiquitin (76 aa) and human alpha-synuclein (UniProt P37840, 140 aa; the CheZOD117
# chain 6968_1_1_1 and the Supplementary case-study seed carry exactly this sequence).
UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
ASYN = (
    "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQ"
    "KTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA"
)
LABELS = {UBQ: "ubiquitin", ASYN: "α-synuclein"}
COLUMNS = ["id", "entry_id", "sequence", "entity_name", "is_bound", "gscores"]
ORDERED_MAX_MEDIAN_G = 0.2  # "mostly ordered": median chain-mean G below this
_GENERIC = re.compile(
    r"^(entity|chain|polymer|protein|molecule|mol|peptide|monomer|subunit)?[\s_\-]*([a-z]|\d+)?$",
    re.I,
)


class RepeatHistogram(NamedTuple):
    """``hist``: one row per observed ``n_entries`` (1, 2, ...) with ``n_sequences`` and
    ``n_chains`` (all chains of those sequences) and ``multi`` (n_entries >= 2).
    ``top``: the most-measured sequences (by entries, then chains) with ``label``,
    ``name``, ``length``, ``n_entries``, ``n_chains``, ``n_nonbound_entries`` and
    ``example_id``."""

    hist: pd.DataFrame
    top: pd.DataFrame


def load(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """The release Parquet (``COLUMNS``) unless a frame with those columns is passed."""
    if df is None:
        df = sp.load_main(COLUMNS)
    return df


def readable_name(names) -> str:
    """Most frequent non-generic entity name ('' when all are generic or missing)."""
    c = Counter(
        str(n).strip()
        for n in names
        if n is not None
        and not (isinstance(n, float) and np.isnan(n))
        and str(n).strip()
        and not _GENERIC.fullmatch(str(n).strip())
    )
    return c.most_common(1)[0][0] if c else ""


def sequence_table(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per exact sequence of the release, sorted by entries then chains (desc)."""
    df = load(df)
    bound = df["is_bound"].fillna(False).astype(bool)
    g = df.groupby("sequence", sort=False)
    t = pd.DataFrame(
        {
            "n_entries": g["entry_id"].nunique(),
            "n_chains": g["id"].size(),
            "n_nonbound_chains": (~bound).groupby(df["sequence"], sort=False).sum(),
            "n_nonbound_entries": df[~bound].groupby("sequence")["entry_id"].nunique(),
            "name": g["entity_name"].agg(readable_name),
            "example_id": g["id"].min(),
        }
    )
    t["n_nonbound_entries"] = t["n_nonbound_entries"].fillna(0).astype(int)
    t = t.rename_axis("sequence").reset_index()
    t["length"] = t["sequence"].str.len()
    t["label"] = [LABELS.get(s, n) for s, n in zip(t["sequence"], t["name"])]
    return t.sort_values(
        ["n_entries", "n_chains", "example_id"], ascending=[False, False, True]
    ).reset_index(drop=True)


def repeat_histogram(df: pd.DataFrame | None = None, n_top: int = 5) -> RepeatHistogram:
    """Fig. 2d data: sequences (and their chains) by the number of entries that measured
    them, over all chains, plus the ``n_top`` most-measured sequences with names."""
    t = sequence_table(df)
    hist = (
        t.groupby("n_entries")
        .agg(n_sequences=("sequence", "size"), n_chains=("n_chains", "sum"))
        .reset_index()
        .sort_values("n_entries")
        .reset_index(drop=True)
    )
    hist["multi"] = hist["n_entries"] >= 2
    top = t.head(n_top)[
        ["label", "name", "length", "n_entries", "n_chains", "n_nonbound_entries", "example_id"]
    ].reset_index(drop=True)  # fmt: skip
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    return RepeatHistogram(hist, top)


def repeat_stats(df: pd.DataFrame | None = None) -> dict:
    """Scalar repeat statistics over all chains of the release."""
    df = load(df)
    t = sequence_table(df)
    multi = t[t["n_entries"] >= 2]
    nmax = int(t["n_entries"].max())
    at_max = t[t["n_entries"] == nmax]
    out = {
        "n_chains": int(len(df)),
        "n_sequences": int(len(t)),
        "n_seqs_multi_entry": int(len(multi)),
        "n_chains_multi_entry": int(multi["n_chains"].sum()),
        "n_entries_max_per_seq": nmax,
        "n_seqs_at_max": int(len(at_max)),
        "labels_at_max": list(at_max["label"]),
    }
    for key, seq in (("asyn", ASYN), ("ubq", UBQ)):
        sub = df[df["sequence"] == seq]
        out[f"n_chains_{key}_exact"] = int(len(sub))
        out[f"n_entries_{key}_exact"] = int(sub["entry_id"].nunique())
    return out


def ubq_most_entries_ordered(
    df: pd.DataFrame | None = None, g_max: float = ORDERED_MAX_MEDIAN_G
) -> dict:
    """Among non-bound chains, the sequences whose median chain-mean G (Parquet labels) is
    below ``g_max``, ranked by their number of non-bound entries. Returns the top two and
    whether ubiquitin is the first."""
    df = load(df)
    nb = df[~df["is_bound"].fillna(False).astype(bool)].copy()

    def chain_mean(g):
        g = np.asarray(g, float)
        return float(np.nanmean(g)) if np.isfinite(g).any() else np.nan

    nb["meanG"] = [chain_mean(g) for g in nb["gscores"]]
    s = nb.groupby("sequence").agg(
        n_entries=("entry_id", "nunique"), median_mean_g=("meanG", "median")
    )
    s = s[s["median_mean_g"] < g_max].sort_values(
        ["n_entries", "median_mean_g"], ascending=[False, True]
    )
    first, second = s.iloc[0], s.iloc[1]
    return {
        "top_is_ubq": bool(s.index[0] == UBQ),
        "top_label": LABELS.get(s.index[0], s.index[0][:12]),
        "top_n_entries": int(first["n_entries"]),
        "top_median_mean_g": float(first["median_mean_g"]),
        "runner_up_n_entries": int(second["n_entries"]),
        "runner_up_label": LABELS.get(s.index[1], s.index[1][:12]),
        "n_candidates": int(len(s)),
        "g_max": g_max,
    }


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    D = load()
    h = repeat_histogram(D)
    print(h.hist.to_string())
    print(h.top.to_string())
    print(repeat_stats(D))
    print(ubq_most_entries_ordered(D))
