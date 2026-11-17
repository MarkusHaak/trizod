#!/usr/bin/env python3
"""Deep-dive every TriZOD vs CheZOD1325 mismatch; test the 'align by sequence' fix.

The first-pass comparison (verify_chezod_parsing.py) required an EXACT sequence
match before comparing Z-scores, leaving 96 entries as "length mismatch" and
3 unscored. Here we instead align each CheZOD sequence to TriZOD's via
difflib matching blocks and recompute concordance on the aligned overlap — the
candidate "easy fix" — and characterise what the residual differences are.

For each CheZOD entry covered by TriZOD:
  * pick the TriZOD record whose sequence best aligns to CheZOD's,
  * align via SequenceMatcher matching blocks,
  * classify the flanks (N-/C-terminal residues one side has that the other
    lacks), and
  * compute per-residue Pearson on the aligned, mutually-present positions.

Also: a registration scan for bmr27230 (the exact-match, zero-LACS, r=0.05
outlier) to tell a numbering offset from a genuine data difference.

Outputs under docs/260611/data/chezod_verification/ (gitignored):
  per_entry_aligned.csv, mismatch_summary.json
"""

from __future__ import annotations

import csv
import json
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[3]
CHEZOD = ROOT / "data" / "chezod" / "protein_nmr_1325"
TRIZOD_SCORES = ROOT / "data" / "release" / "unfiltered" / "scores.json"
OUT = ROOT / "docs" / "260611" / "data" / "chezod_verification"
NA = 999.0


def load_chezod() -> dict[str, dict]:
    seqs = [ln.rstrip("\n") for ln in (CHEZOD / "allseqs1325.txt").open() if ln.strip()]
    scrs = [
        ln.rstrip("\n")
        for ln in (CHEZOD / "allscores1325newest.txt").open()
        if ln.strip()
    ]
    out = {}
    for sline, scline in zip(seqs, scrs):
        p = sline.split(None, 1)
        if len(p) < 2 or not p[0].isdigit():
            continue
        seq = p[1].replace(" ", "")
        vals = [float(v) for v in scline.strip().strip("[]").split(",") if v.strip()]
        if len(vals) != len(seq):
            continue
        out[p[0]] = {"seq": seq, "z": [None if v == NA else v for v in vals]}
    return out


def load_trizod() -> dict[str, list[dict]]:
    by = {}
    with TRIZOD_SCORES.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            by.setdefault(str(r["entryID"]), []).append(
                {"seq": r["seq"] or "", "z": r["zscores"] or []}
            )
    return by


def aligned_pairs(cz, tz):
    """Return (cz_idx, tz_idx) aligned position pairs over matching blocks."""
    sm = SequenceMatcher(None, cz["seq"], tz["seq"], autojunk=False)
    pairs = []
    for i, j, n in sm.get_matching_blocks():
        for t in range(n):
            pairs.append((i + t, j + t))
    return pairs


def corr_on_pairs(cz, tz, pairs):
    a, b = [], []
    for i, j in pairs:
        cv, tv = cz["z"][i], tz["z"][j]
        if cv is not None and tv is not None:
            a.append(cv)
            b.append(tv)
    if len(a) < 5:
        return None, len(a)
    return float(pearsonr(np.array(a), np.array(b))[0]), len(a)


def best_aligned_record(cz, recs):
    best, best_aligned, best_pairs = None, -1, []
    for r in recs:
        pairs = aligned_pairs(cz, r)
        if len(pairs) > best_aligned:
            best, best_aligned, best_pairs = r, len(pairs), pairs
    return best, best_pairs


def classify(cz_seq, tz_seq, pairs):
    if not pairs:
        return "no_alignment", 0, 0
    matched = len(pairs)
    cz_first, cz_last = pairs[0][0], pairs[-1][0]
    tz_first, tz_last = pairs[0][1], pairs[-1][1]
    n_extra_tz = tz_first + (len(tz_seq) - 1 - tz_last)  # tz residues outside overlap
    n_extra_cz = cz_first + (len(cz_seq) - 1 - cz_last)
    if cz_seq == tz_seq:
        cat = "exact"
    elif matched == len(cz_seq) and len(tz_seq) > len(cz_seq):
        cat = "cz_substring_of_tz"  # TriZOD has extra flanking residues
    elif matched == len(tz_seq) and len(cz_seq) > len(tz_seq):
        cat = "tz_substring_of_cz"
    else:
        cat = "partial_overlap"
    return cat, n_extra_tz, n_extra_cz


def registration_scan(cz, tz):
    """For an exact-length entry, try shifting CheZOD vs TriZOD by -8..8."""
    best = (0, None)
    cz_z, tz_z = cz["z"], tz["z"]
    n = min(len(cz_z), len(tz_z))
    for s in range(-8, 9):
        a, b = [], []
        for i in range(n):
            j = i + s
            if 0 <= j < n and cz_z[i] is not None and tz_z[j] is not None:
                a.append(cz_z[i])
                b.append(tz_z[j])
        if len(a) >= 10:
            r = float(pearsonr(np.array(a), np.array(b))[0])
            if best[1] is None or r > best[1]:
                best = (s, round(r, 3))
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    chezod, trizod = load_chezod(), load_trizod()

    rows = []
    for eid, cz in chezod.items():
        recs = trizod.get(eid)
        if not recs:
            continue
        rec, pairs = best_aligned_record(cz, recs)
        cat, extra_tz, extra_cz = classify(cz["seq"], rec["seq"], pairs)
        r_aligned, n_aligned = corr_on_pairs(cz, rec, pairs)
        rows.append(
            {
                "bmrb_id": eid,
                "category": cat,
                "len_cz": len(cz["seq"]),
                "len_tz": len(rec["seq"]),
                "extra_tz_residues": extra_tz,
                "extra_cz_residues": extra_cz,
                "n_aligned": n_aligned,
                "pearson_aligned": None if r_aligned is None else round(r_aligned, 4),
            }
        )

    with (OUT / "per_entry_aligned.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def has_r(r):
        return r["pearson_aligned"] is not None

    cats = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    scored = [r for r in rows if has_r(r)]
    rvals = np.array([r["pearson_aligned"] for r in scored])
    ge99 = int((rvals >= 0.99).sum())
    ge95 = int((rvals >= 0.95).sum())
    ge90 = int((rvals >= 0.90).sum())
    lt90 = sorted(
        ({k: r[k] for k in r} for r in scored if r["pearson_aligned"] < 0.90),
        key=lambda x: x["pearson_aligned"],
    )

    # focus: the previously length-mismatched (non-exact) entries
    nonexact = [r for r in scored if r["category"] != "exact"]
    ne_r = np.array([r["pearson_aligned"] for r in nonexact])

    summary = {
        "covered_entries": len(rows),
        "categories": cats,
        "aligned_pearson": {
            "n": len(scored),
            "median": float(np.median(rvals)),
            ">=0.99": ge99,
            ">=0.95": ge95,
            ">=0.90": ge90,
            "<0.90": len(scored) - ge90,
        },
        "non_exact_entries_after_alignment": {
            "n": len(nonexact),
            "median_pearson": float(np.median(ne_r)) if len(ne_r) else None,
            ">=0.95": int((ne_r >= 0.95).sum()) if len(ne_r) else 0,
            ">=0.90": int((ne_r >= 0.90).sum()) if len(ne_r) else 0,
        },
        "remaining_low_r_below_0.90": lt90,
        "flank_pattern_non_exact": {
            "tz_has_more_flank": sum(
                1 for r in nonexact if r["extra_tz_residues"] > r["extra_cz_residues"]
            ),
            "cz_has_more_flank": sum(
                1 for r in nonexact if r["extra_cz_residues"] > r["extra_tz_residues"]
            ),
            "median_extra_tz": float(
                np.median([r["extra_tz_residues"] for r in nonexact])
            )
            if nonexact
            else None,
            "median_extra_cz": float(
                np.median([r["extra_cz_residues"] for r in nonexact])
            )
            if nonexact
            else None,
        },
    }

    # bmr27230 registration scan
    if "27230" in chezod and "27230" in trizod:
        rec, _ = best_aligned_record(chezod["27230"], trizod["27230"])
        s, r = registration_scan(chezod["27230"], rec)
        summary["bmr27230_registration_scan"] = {"best_shift": s, "best_pearson": r}

    (OUT / "mismatch_summary.json").write_text(json.dumps(summary, indent=2))

    print("=== Category breakdown (alignment-based) ===")
    for k, v in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {k:24s}: {v}")
    print("\n=== Aligned per-residue Pearson (all covered) ===")
    print(
        f"  n={len(scored)} median={np.median(rvals):.3f} | "
        f">=0.99:{ge99}  >=0.95:{ge95}  >=0.90:{ge90}  <0.90:{len(scored) - ge90}"
    )
    print("\n=== Previously length-mismatched entries, after sequence alignment ===")
    if len(ne_r):
        print(
            f"  n={len(nonexact)} median Pearson={np.median(ne_r):.3f} | "
            f">=0.95:{int((ne_r >= 0.95).sum())}  >=0.90:{int((ne_r >= 0.90).sum())}"
        )
        fp = summary["flank_pattern_non_exact"]
        print(
            f"  flank: TriZOD-longer={fp['tz_has_more_flank']}, "
            f"CheZOD-longer={fp['cz_has_more_flank']}, "
            f"median extra TriZOD={fp['median_extra_tz']}, "
            f"CheZOD={fp['median_extra_cz']}"
        )
    print(f"\n=== Remaining r<0.90 after alignment: {len(lt90)} ===")
    for r in lt90[:15]:
        print(
            f"  bmr{r['bmrb_id']}: r={r['pearson_aligned']} cat={r['category']} "
            f"len_cz={r['len_cz']} len_tz={r['len_tz']} n={r['n_aligned']}"
        )
    if "bmr27230_registration_scan" in summary:
        print(
            f"\n=== bmr27230 registration scan: {summary['bmr27230_registration_scan']}"
        )
    print(
        f"\nWrote {OUT / 'per_entry_aligned.csv'} and {OUT / 'mismatch_summary.json'}"
    )


if __name__ == "__main__":
    main()
