#!/usr/bin/env python3
"""Re-verify TriZOD parsing/scoring against the CheZOD1325 reference.

Context
-------
The original TriZOD report claims TriZOD's NMR-STAR parser is more correct than
CheZOD's (which mis-associated 11 and errored on 5 entries of the 1325-protein
set, and cannot read ~1700 modern entries). Reproducing the EXACT 11+5 counts
needs CheZOD's own parser (not available here). What we CAN do with the fetched
CheZOD reference Z-scores is a concordance analysis:

  1. Coverage  — how many CheZOD1325 entries does TriZOD also score?
  2. Sequence  — do TriZOD and CheZOD agree on the sequence per entry?
                 (referencing-independent: the direct parsing/association check)
  3. Z-scores  — per-residue concordance on sequence-matched entries.
                 Divergences are split by each entry's max |LACS offset|:
                 large LACS offset => divergence is TriZOD re-referencing
                 (a correction, e.g. alpha-synuclein), not a parse error;
                 small LACS offset + low correlation => candidate genuine
                 discrepancy / CheZOD mis-association.

Inputs
------
* data/chezod/protein_nmr_1325/allseqs1325.txt        (<BMRB_ID> <seq>)
* data/chezod/protein_nmr_1325/allscores1325newest.txt (line-aligned z, 999=NA)
* data/release/unfiltered/scores.json                  (TriZOD scores, JSONL)

Outputs (under docs/260611/data/chezod_verification/, gitignored)
-----------------------------------------------------------------
* per_entry.csv   — one row per CheZOD entry with match + concordance stats
* summary.json    — aggregate counts and distributions
Prints a human-readable summary to stdout.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[3]
CHEZOD = ROOT / "data" / "chezod" / "protein_nmr_1325"
SEQS = CHEZOD / "allseqs1325.txt"
SCORES = CHEZOD / "allscores1325newest.txt"
TRIZOD_SCORES = ROOT / "data" / "release" / "unfiltered" / "scores.json"
OUT = ROOT / "docs" / "260611" / "data" / "chezod_verification"

ATOMS = ["C", "CA", "CB", "H", "HA", "HB", "N"]
NA = 999.0
LOW_CORR = 0.5  # flag matched entries below this Pearson r
SMALL_LACS = 0.5  # |LACS offset| (ppm) below which divergence is NOT re-ref


def load_chezod() -> dict[str, dict]:
    seq_lines = [ln.rstrip("\n") for ln in SEQS.open() if ln.strip()]
    score_lines = [ln.rstrip("\n") for ln in SCORES.open() if ln.strip()]
    assert len(seq_lines) == len(score_lines), "CheZOD seq/score line mismatch"
    out: dict[str, dict] = {}
    for sline, scline in zip(seq_lines, score_lines):
        parts = sline.split(None, 1)
        if len(parts) < 2 or not parts[0].isdigit():
            continue  # skip the 2 non-numeric IDs (dss1, hmbdi)
        bmrb_id, seq = parts[0], parts[1].replace(" ", "")
        vals = [float(v) for v in scline.strip().strip("[]").split(",") if v.strip()]
        if len(vals) != len(seq):
            continue
        z = [None if v == NA else v for v in vals]
        out[bmrb_id] = {"seq": seq, "z": z}
    return out


def load_trizod() -> dict[str, list[dict]]:
    by_entry: dict[str, list[dict]] = {}
    with TRIZOD_SCORES.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            eid = str(r["entryID"])
            max_lacs = max(
                (abs(r.get(f"lacs_off_{a}") or 0.0) for a in ATOMS), default=0.0
            )
            by_entry.setdefault(eid, []).append(
                {"seq": r["seq"] or "", "z": r["zscores"] or [], "max_lacs": max_lacs}
            )
    return by_entry


def best_record(cz_seq: str, recs: list[dict]) -> tuple[dict | None, str]:
    """Pick the TriZOD record best matching the CheZOD sequence."""
    exact = [r for r in recs if r["seq"] == cz_seq]
    if exact:
        return exact[0], "exact"
    same_len = [r for r in recs if len(r["seq"]) == len(cz_seq)]
    if same_len:
        # most identical residues
        best = max(
            same_len,
            key=lambda r: sum(a == b for a, b in zip(r["seq"], cz_seq)),
        )
        ident = sum(a == b for a, b in zip(best["seq"], cz_seq)) / len(cz_seq)
        return best, ("len_match_identical" if ident == 1.0 else "len_match_diff")
    return (recs[0] if recs else None), "len_mismatch"


def concordance(cz_z: list, tz_z: list) -> dict:
    a, b = [], []
    for cz, tz in zip(cz_z, tz_z):
        if cz is not None and tz is not None:
            a.append(cz)
            b.append(tz)
    n = len(a)
    if n < 5:
        return {"n": n, "pearson": None, "spearman": None, "mae": None}
    a_arr, b_arr = np.array(a), np.array(b)
    pear = float(pearsonr(a_arr, b_arr)[0])
    spear = float(spearmanr(a_arr, b_arr)[0])
    return {
        "n": n,
        "pearson": pear,
        "spearman": spear,
        "mae": float(np.mean(np.abs(a_arr - b_arr))),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    chezod = load_chezod()
    trizod = load_trizod()
    print(f"CheZOD1325 entries (numeric BMRB ID): {len(chezod)}")
    print(f"TriZOD unfiltered distinct entryIDs:  {len(trizod)}")

    rows = []
    for bmrb_id, cz in chezod.items():
        recs = trizod.get(bmrb_id)
        if not recs:
            rows.append({"bmrb_id": bmrb_id, "covered": False, "match": "not_scored"})
            continue
        rec, match = best_record(cz["seq"], recs)
        conc = concordance(cz["z"], rec["z"]) if rec else {"n": 0}
        rows.append(
            {
                "bmrb_id": bmrb_id,
                "covered": True,
                "match": match,
                "len_chezod": len(cz["seq"]),
                "len_trizod": len(rec["seq"]) if rec else 0,
                "n_compared": conc.get("n"),
                "pearson": conc.get("pearson"),
                "spearman": conc.get("spearman"),
                "mae": conc.get("mae"),
                "max_lacs_off": round(rec["max_lacs"], 3) if rec else None,
            }
        )

    # write per-entry CSV
    cols = [
        "bmrb_id",
        "covered",
        "match",
        "len_chezod",
        "len_trizod",
        "n_compared",
        "pearson",
        "spearman",
        "mae",
        "max_lacs_off",
    ]
    with (OUT / "per_entry.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})

    covered = [r for r in rows if r.get("covered")]
    not_scored = [r for r in rows if not r.get("covered")]
    exact = [r for r in covered if r["match"] == "exact"]
    len_ident = [r for r in covered if r["match"] == "len_match_identical"]
    len_diff = [r for r in covered if r["match"] == "len_match_diff"]
    len_mis = [r for r in covered if r["match"] == "len_mismatch"]
    seq_ok = exact + len_ident  # sequence agrees residue-for-residue

    with_corr = [r for r in seq_ok if r.get("pearson") is not None]
    pear = np.array([r["pearson"] for r in with_corr])
    low = [r for r in with_corr if r["pearson"] < LOW_CORR]
    low_reref = [r for r in low if (r["max_lacs_off"] or 0) >= SMALL_LACS]
    low_genuine = [r for r in low if (r["max_lacs_off"] or 0) < SMALL_LACS]

    summary = {
        "chezod_entries": len(chezod),
        "covered_by_trizod": len(covered),
        "not_scored_by_trizod": [r["bmrb_id"] for r in not_scored],
        "sequence_match": {
            "exact": len(exact),
            "len_match_identical": len(len_ident),
            "len_match_diff": len(len_diff),
            "len_mismatch": len(len_mis),
        },
        "zscore_concordance_on_seq_matched": {
            "n_entries": len(with_corr),
            "pearson_median": float(np.median(pear)) if len(pear) else None,
            "pearson_q1": float(np.percentile(pear, 25)) if len(pear) else None,
            "pearson_q3": float(np.percentile(pear, 75)) if len(pear) else None,
            "pearson_mean": float(np.mean(pear)) if len(pear) else None,
            "n_low_corr": len(low),
            "n_low_corr_reref_explained": len(low_reref),
            "n_low_corr_candidate_discrepancy": len(low_genuine),
        },
        "low_corr_candidate_discrepancies": sorted(
            (
                {
                    "bmrb_id": r["bmrb_id"],
                    "pearson": round(r["pearson"], 3),
                    "max_lacs_off": r["max_lacs_off"],
                    "n": r["n_compared"],
                }
                for r in low_genuine
            ),
            key=lambda x: x["pearson"],
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Coverage / parsing ===")
    print(f"  CheZOD1325 entries TriZOD also scored: {len(covered)}/{len(chezod)}")
    if not_scored:
        print(f"  NOT scored by TriZOD: {[r['bmrb_id'] for r in not_scored]}")
    print("\n=== Sequence concordance (parsing/association check) ===")
    print(f"  exact sequence match        : {len(exact)}")
    print(f"  length match, identical      : {len(len_ident)}")
    print(f"  length match, differing res  : {len(len_diff)}")
    print(f"  length mismatch              : {len(len_mis)}")
    print("\n=== Z-score concordance on sequence-matched entries ===")
    if len(pear):
        print(
            f"  entries compared: {len(with_corr)} | Pearson median "
            f"{np.median(pear):.3f} (Q1 {np.percentile(pear, 25):.3f}, "
            f"Q3 {np.percentile(pear, 75):.3f})"
        )
    print(
        f"  low-corr (r<{LOW_CORR}): {len(low)} "
        f"-> {len(low_reref)} explained by LACS re-referencing "
        f"(|offset|>={SMALL_LACS} ppm), "
        f"{len(low_genuine)} candidate genuine discrepancies"
    )
    if summary["low_corr_candidate_discrepancies"]:
        print("  candidate discrepancies (low r, small LACS offset):")
        for r in summary["low_corr_candidate_discrepancies"][:15]:
            print(
                f"    bmr{r['bmrb_id']}: r={r['pearson']}, "
                f"maxLACS={r['max_lacs_off']}, n={r['n']}"
            )
    print(f"\nWrote {OUT / 'per_entry.csv'} and {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
