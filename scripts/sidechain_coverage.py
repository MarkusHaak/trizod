#!/usr/bin/env python3
"""Side-chain chemical-shift coverage across the four stringency tiers.

Reproducible source for the manuscript's side-chain numbers: how many of the
shifts the scoring path discards there actually are, how they are distributed
over chains, which nuclei they cover, and how complete the assignments are per
residue type — including the methyl-TROSY (ILVMA) and aromatic-ring probes.

Method: one pass over the unfiltered chain set (a superset of every other
tier), computing per-chain counters from ``get_valid_bbshifts`` (backbone view,
needed for ALA CB, which is a backbone-whitelisted atom) and
``get_sidechain_shifts`` (everything else). Tier tables are then subsets of the
same per-chain counters, so the tiers are strictly comparable.

Denominators are explicit, because "completeness" has no canonical definition:

* expected side-chain atoms = every H/C/N atom of the residue's side chain that
  the backbone whitelist does NOT already consume (so ALA and GLY have none),
  standard BMRB/IUPAC nomenclature. Reported both over all canonical residues
  of the sequence and over the residues that carry any assignment at all, and
  both with and without the labile (exchangeable) side-chain protons.
* methyl-TROSY = ILE CD1, LEU CD1/CD2, VAL CG1/CG2, MET CE, ALA CB (carbons).
* aromatic = ring carbons and nitrogens of PHE/TYR/TRP/HIS.

Usage:
    uv run python scripts/sidechain_coverage.py --output docs/dataset/sidechain-coverage.md
    uv run python scripts/sidechain_coverage.py --cache tmp/sidechain_coverage.npz
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trizod import paths
from trizod.bmrb.bmrb import BB_ATOM_IDS, get_sidechain_shifts, get_valid_bbshifts
from trizod.constants import AA1TO3, AA3TO1, BACKBONE_ATOMS
from trizod.shifts import iter_chains

TIERS = ["unfiltered", "tolerant", "moderate", "strict"]
AA3 = sorted(AA3TO1)
AA_IDX = {aa: i for i, aa in enumerate(AA3)}
CB_COL = BACKBONE_ATOMS.index("CB")

# Side-chain H/C/N atoms per residue, minus the 12 the backbone read consumes
# ({C,CA,CB,H,HA,HB,N,HA2,HA3,HB1,HB2,HB3}) -- hence ALA and GLY are empty.
# Sulfur/oxygen are unobservable in these experiments and are not counted.
EXPECTED_SIDECHAIN = {
    "ALA": (),
    "ARG": (
        "CG",
        "HG2",
        "HG3",
        "CD",
        "HD2",
        "HD3",
        "NE",
        "HE",
        "CZ",
        "NH1",
        "NH2",
        "HH11",
        "HH12",
        "HH21",
        "HH22",
    ),
    "ASN": ("CG", "ND2", "HD21", "HD22"),
    "ASP": ("CG",),
    "CYS": ("HG",),
    "GLN": ("CG", "HG2", "HG3", "CD", "NE2", "HE21", "HE22"),
    "GLU": ("CG", "HG2", "HG3", "CD"),
    "GLY": (),
    "HIS": ("CG", "ND1", "HD1", "CD2", "HD2", "CE1", "HE1", "NE2", "HE2"),
    "ILE": (
        "CG1",
        "HG12",
        "HG13",
        "CG2",
        "HG21",
        "HG22",
        "HG23",
        "CD1",
        "HD11",
        "HD12",
        "HD13",
    ),
    "LEU": ("CG", "HG", "CD1", "HD11", "HD12", "HD13", "CD2", "HD21", "HD22", "HD23"),
    "LYS": (
        "CG",
        "HG2",
        "HG3",
        "CD",
        "HD2",
        "HD3",
        "CE",
        "HE2",
        "HE3",
        "NZ",
        "HZ1",
        "HZ2",
        "HZ3",
    ),
    "MET": ("CG", "HG2", "HG3", "CE", "HE1", "HE2", "HE3"),
    "PHE": ("CG", "CD1", "HD1", "CD2", "HD2", "CE1", "HE1", "CE2", "HE2", "CZ", "HZ"),
    "PRO": ("CG", "HG2", "HG3", "CD", "HD2", "HD3"),
    "SER": ("HG",),
    "THR": ("CG2", "HG21", "HG22", "HG23", "HG1"),
    "TRP": (
        "CG",
        "CD1",
        "HD1",
        "CD2",
        "NE1",
        "HE1",
        "CE2",
        "CE3",
        "HE3",
        "CZ2",
        "HZ2",
        "CZ3",
        "HZ3",
        "CH2",
        "HH2",
    ),
    "TYR": ("CG", "CD1", "HD1", "CD2", "HD2", "CE1", "HE1", "CE2", "HE2", "CZ", "HH"),
    "VAL": ("CG1", "HG11", "HG12", "HG13", "CG2", "HG21", "HG22", "HG23"),
}

# Labile side-chain protons: routinely unobservable in H2O at neutral pH, so
# counting them in the denominator understates the achievable completeness.
EXCHANGEABLE = {
    "ARG": ("HE", "HH11", "HH12", "HH21", "HH22"),
    "ASN": ("HD21", "HD22"),
    "CYS": ("HG",),
    "GLN": ("HE21", "HE22"),
    "HIS": ("HD1", "HE2"),
    "LYS": ("HZ1", "HZ2", "HZ3"),
    "SER": ("HG",),
    "THR": ("HG1",),
    "TRP": ("HE1",),
    "TYR": ("HH",),
}

METHYL_CARBONS = {
    "ALA": ("CB",),  # backbone-whitelisted: counted from the backbone mask
    "ILE": ("CD1",),
    "LEU": ("CD1", "CD2"),
    "MET": ("CE",),
    "VAL": ("CG1", "CG2"),
}

AROMATIC_RING = {
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TRP": ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
}

NONEXCHANGEABLE = {
    a: set(EXPECTED_SIDECHAIN[a]) - set(EXCHANGEABLE.get(a, ())) for a in AA3
}
# carbon/nitrogen skeleton only: the variant usually quoted as "side-chain
# assignment completeness", and the one insensitive to whether a lab deposited
# its protons
HEAVY = {a: {x for x in EXPECTED_SIDECHAIN[a] if x[0] in "CN"} for a in AA3}

# per-residue-type denominators, aligned to AA3
EXPECTED_N = np.array([len(EXPECTED_SIDECHAIN[a]) for a in AA3], dtype=np.int64)
NONEXCH_N = np.array([len(NONEXCHANGEABLE[a]) for a in AA3], dtype=np.int64)
HEAVY_N = np.array([len(HEAVY[a]) for a in AA3], dtype=np.int64)
METHYL_N = np.array([len(METHYL_CARBONS.get(a, ())) for a in AA3], dtype=np.int64)
AROMATIC_N = np.array([len(AROMATIC_RING.get(a, ())) for a in AA3], dtype=np.int64)

# per-chain counter blocks, all (n_chains, 20) except the scalars/nucleus block
BLOCKS = [
    "res_seq",  # canonical residues in the polymer sequence
    "res_assigned",  # residues carrying >=1 deposited shift of any kind
    "sc_obs",  # observed side-chain atoms within the expected set
    "sc_obs_nonexch",  # same, labile protons excluded
    "sc_obs_heavy",  # same, C/N skeleton only
    "methyl_obs",  # observed methyl carbons (ALA CB from the backbone mask)
    "aromatic_obs",  # observed aromatic ring C/N
]
NUCLEI = ["H", "C", "N", "other"]


def chain_counters(seq, shifts):
    """Per-chain counters, or None when the chain has no usable shift table."""
    bb = get_valid_bbshifts(shifts, seq)
    sc = get_sidechain_shifts(shifts, seq)
    if bb is None and sc is None:
        return None

    counts = {name: np.zeros(len(AA3), dtype=np.int64) for name in BLOCKS}
    nucleus = np.zeros(len(NUCLEI), dtype=np.int64)

    for aa1 in seq:
        aa3 = AA1TO3.get(aa1)
        if aa3 is not None:
            counts["res_seq"][AA_IDX[aa3]] += 1

    assigned = np.zeros(len(seq), dtype=bool)
    if bb is not None:
        _, bb_mask = bb
        assigned |= bb_mask.any(axis=1)
        # ALA CB is a methyl-TROSY probe that the backbone read keeps
        ala = np.array([aa1 == "A" for aa1 in seq])
        counts["methyl_obs"][AA_IDX["ALA"]] += int((ala & bb_mask[:, CB_COL]).sum())

    if sc is not None and len(sc):
        assigned[sc["seq_id"].to_numpy() - 1] = True
        for nuc, n in sc["atom_type"].value_counts().items():
            nucleus[NUCLEI.index(nuc) if nuc in NUCLEI else -1] += int(n)
        # coverage counts distinct atoms; conflicting duplicates (19 values
        # corpus-wide) must not inflate it
        uniq = sc.drop_duplicates(subset=["seq_id", "atom_id"])
        for comp_id, group in uniq.groupby("comp_id", sort=False):
            i = AA_IDX[comp_id]
            atoms = group["atom_id"]
            counts["sc_obs"][i] += int(atoms.isin(EXPECTED_SIDECHAIN[comp_id]).sum())
            counts["sc_obs_nonexch"][i] += int(
                atoms.isin(NONEXCHANGEABLE[comp_id]).sum()
            )
            counts["sc_obs_heavy"][i] += int(atoms.isin(HEAVY[comp_id]).sum())
            counts["methyl_obs"][i] += int(
                atoms.isin(METHYL_CARBONS.get(comp_id, ())).sum()
            )
            counts["aromatic_obs"][i] += int(
                atoms.isin(AROMATIC_RING.get(comp_id, ())).sum()
            )

    for i, aa1 in enumerate(seq):
        aa3 = AA1TO3.get(aa1)
        if aa3 is not None and assigned[i]:
            counts["res_assigned"][AA_IDX[aa3]] += 1

    counts["n_sidechain"] = len(sc) if sc is not None else 0
    # same rows before get_sidechain_shifts() drops fully identical re-statements
    counts["n_sidechain_raw"] = (
        sum(
            1
            for s in shifts
            if s[3] in AA3TO1 and s[4] not in BB_ATOM_IDS  # Comp_ID, Atom_ID
        )
        if sc is not None
        else 0
    )
    counts["nucleus"] = nucleus
    return counts


def collect(ids, pkl_dir):
    """Run :func:`chain_counters` over every chain; returns (ids, arrays)."""
    kept, rows, nucleus = [], {name: [] for name in BLOCKS}, []
    n_sc, n_sc_raw = [], []
    for n, (cid, seq, shifts) in enumerate(iter_chains(ids, pkl_dir), start=1):
        if n % 2000 == 0:
            print(f"  {n} chains processed", flush=True)
        counts = chain_counters(seq, shifts)
        if counts is None:
            continue
        kept.append(cid)
        for name in BLOCKS:
            rows[name].append(counts[name])
        nucleus.append(counts["nucleus"])
        n_sc.append(counts["n_sidechain"])
        n_sc_raw.append(counts["n_sidechain_raw"])
    arrays = {name: np.array(rows[name], dtype=np.int64) for name in BLOCKS}
    arrays["nucleus"] = np.array(nucleus, dtype=np.int64)
    arrays["n_sidechain"] = np.array(n_sc, dtype=np.int64)
    arrays["n_sidechain_raw"] = np.array(n_sc_raw, dtype=np.int64)
    return np.array(kept), arrays


def read_ids(scored_dir: Path, tier: str):
    path = scored_dir / tier / "scores.json"
    if not path.exists():
        raise SystemExit(f"missing scored tier: {path}")
    with path.open() as fh:
        return [json.loads(ln)["ID"] for ln in fh if ln.strip()]


def pct(num, den):
    return f"{100.0 * num / den:.1f} %" if den else "n/a"


def frac(num, den):
    return f"{num / den:.3f}" if den else "n/a"


def report(ids, arrays, tier_ids) -> str:
    idx = {cid: i for i, cid in enumerate(ids)}
    out = ["# Side-chain chemical-shift coverage", ""]
    out.append(
        "Values are AS DEPOSITED in BMRB: no re-referencing and no offset "
        "correction is applied to side-chain shifts."
    )
    out.append("")

    sel = {}
    for tier in TIERS:
        rows = np.array([idx[c] for c in tier_ids[tier] if c in idx], dtype=int)
        sel[tier] = rows

    # --- bulk ------------------------------------------------------------
    out.append("## Bulk")
    out.append("")
    out.append(
        "| tier | chains | chains with side chains | total side-chain shifts | "
        "median / chain | IQR | 1H | 13C | 15N |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|")
    for tier in TIERS:
        rows = sel[tier]
        n_sc = arrays["n_sidechain"][rows]
        nuc = arrays["nucleus"][rows].sum(axis=0)
        q1, med, q3 = np.percentile(n_sc, [25, 50, 75]) if len(n_sc) else (0, 0, 0)
        out.append(
            f"| {tier} | {len(rows)} | {int((n_sc > 0).sum())} "
            f"({pct(int((n_sc > 0).sum()), len(rows))}) | {int(n_sc.sum())} | "
            f"{med:.0f} | {q1:.0f}–{q3:.0f} | {nuc[0]} | {nuc[1]} | {nuc[2]} |"
        )
    out.append("")
    dropped = int(
        (arrays["n_sidechain_raw"] - arrays["n_sidechain"])[sel["unfiltered"]].sum()
    )
    out.append(
        f"Fully identical re-statements of the same value dropped corpus-wide: "
        f"{dropped} (unfiltered). Genuinely conflicting values for the same "
        f"(chain, seq_id, atom_id) are kept."
    )
    out.append("")

    # --- completeness ----------------------------------------------------
    out.append("## Per-residue side-chain completeness")
    out.append("")
    out.append(
        "`all residues` divides by every canonical residue of the sequence; "
        "`assigned` divides only by residues carrying at least one deposited "
        "shift. `non-exchangeable` drops the labile side-chain protons from "
        "both numerator and denominator; `C/N only` keeps just the side-chain "
        "carbon/nitrogen skeleton."
    )
    out.append("")
    out.append(
        "| tier | all residues | all, non-exchangeable | all, C/N only | "
        "assigned residues | assigned, non-exchangeable | assigned, C/N only |"
    )
    out.append("|---|---|---|---|---|---|---|")
    for tier in TIERS:
        rows = sel[tier]
        res_seq = arrays["res_seq"][rows].sum(axis=0)
        res_asg = arrays["res_assigned"][rows].sum(axis=0)
        obs = arrays["sc_obs"][rows].sum(axis=0)
        obs_ne = arrays["sc_obs_nonexch"][rows].sum(axis=0)
        obs_hv = arrays["sc_obs_heavy"][rows].sum(axis=0)
        out.append(
            f"| {tier} | {frac(obs.sum(), res_seq @ EXPECTED_N)} | "
            f"{frac(obs_ne.sum(), res_seq @ NONEXCH_N)} | "
            f"{frac(obs_hv.sum(), res_seq @ HEAVY_N)} | "
            f"{frac(obs.sum(), res_asg @ EXPECTED_N)} | "
            f"{frac(obs_ne.sum(), res_asg @ NONEXCH_N)} | "
            f"{frac(obs_hv.sum(), res_asg @ HEAVY_N)} |"
        )
    out.append("")

    out.append("### By residue type (all residues, full expected set)")
    out.append("")
    out.append("| residue | expected atoms | " + " | ".join(TIERS) + " |")
    out.append("|---|---|" + "---|" * len(TIERS))
    for i, aa in enumerate(AA3):
        if EXPECTED_N[i] == 0:
            continue
        cells = []
        for tier in TIERS:
            rows = sel[tier]
            den = arrays["res_seq"][rows, i].sum() * EXPECTED_N[i]
            cells.append(frac(arrays["sc_obs"][rows, i].sum(), den))
        out.append(f"| {aa} | {EXPECTED_N[i]} | " + " | ".join(cells) + " |")
    out.append("")

    # --- methyl-TROSY ----------------------------------------------------
    out.append("## Methyl-TROSY (ILVMA) 13C coverage")
    out.append("")
    out.append(
        "ILE CD1, LEU CD1/CD2, VAL CG1/CG2, MET CE, ALA CB. ALA CB is a "
        "backbone-whitelisted atom, so it is counted from the backbone mask."
    )
    out.append("")
    out.append("| tier | overall | " + " | ".join(sorted(METHYL_CARBONS)) + " |")
    out.append("|---|---|" + "---|" * len(METHYL_CARBONS))
    for tier in TIERS:
        rows = sel[tier]
        obs = arrays["methyl_obs"][rows].sum(axis=0)
        res_seq = arrays["res_seq"][rows].sum(axis=0)
        cells = [pct(obs.sum(), res_seq @ METHYL_N)]
        for aa in sorted(METHYL_CARBONS):
            i = AA_IDX[aa]
            cells.append(pct(obs[i], res_seq[i] * METHYL_N[i]))
        out.append(f"| {tier} | " + " | ".join(cells) + " |")
    out.append("")

    # --- aromatics -------------------------------------------------------
    out.append("## Aromatic ring 13C/15N coverage")
    out.append("")
    out.append("| tier | overall | " + " | ".join(sorted(AROMATIC_RING)) + " |")
    out.append("|---|---|" + "---|" * len(AROMATIC_RING))
    for tier in TIERS:
        rows = sel[tier]
        obs = arrays["aromatic_obs"][rows].sum(axis=0)
        res_seq = arrays["res_seq"][rows].sum(axis=0)
        cells = [frac(obs.sum(), res_seq @ AROMATIC_N)]
        for aa in sorted(AROMATIC_RING):
            i = AA_IDX[aa]
            cells.append(frac(obs[i], res_seq[i] * AROMATIC_N[i]))
        out.append(f"| {tier} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scored-dir",
        type=Path,
        default=paths.INTERIM_SCORED,
        help="per-tier scores.json dir (default: data/interim/scored)",
    )
    ap.add_argument(
        "--pkl-dir",
        type=Path,
        default=paths.PKL_DIR,
        help="BMRB pickle cache (default: tmp/bmrb_entries)",
    )
    ap.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="npz of per-chain counters; reused when present, written otherwise",
    )
    ap.add_argument("--output", type=Path, default=None, help="write markdown here")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # per-chain rejections are expected in bulk and are counted, not printed
    logging.getLogger("trizod.bmrb").setLevel(logging.CRITICAL)

    tier_ids = {t: read_ids(args.scored_dir, t) for t in TIERS}
    for tier in TIERS:
        print(f"{tier:>10}: {len(tier_ids[tier])} scored chains")

    if args.cache and args.cache.exists():
        print(f"reusing per-chain counters from {args.cache}")
        cached = np.load(args.cache, allow_pickle=False)
        ids = cached["ids"]
        arrays = {k: cached[k] for k in cached.files if k != "ids"}
    else:
        ids, arrays = collect(tier_ids["unfiltered"], args.pkl_dir)
        print(f"collected counters for {len(ids)} chains")
        if args.cache:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.cache, ids=ids, **arrays)
            print(f"cached per-chain counters -> {args.cache}")

    md = report(ids, arrays, tier_ids)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md)
        print(f"\nwrote {args.output}")
    else:
        print()
        print(md)


if __name__ == "__main__":
    main()
