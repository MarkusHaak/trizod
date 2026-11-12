#!/usr/bin/env python3
"""Build the final, de-duplicated TriZOD dataset for downstream clustering.

Inputs
------
* ``data/interim/scored/<tier>/scores.json`` — one JSON line per scored
  (entryID, stID, entity_assemID, entityID) tuple.
* ``tmp/bmrb_entries/<entryID>.pkl`` — pickled ``BmrbEntry`` objects from
  which we read assembly composition (used to flag bound complexes).

Steps
-----
1. Load the per-entry assembly composition from BMRB pkls and mark each
   entry as "single-entity protein" or "multi-molecule (bound)".  Bound
   means any of:
     * an assembly references more than one distinct non-water Entity
       (homo-oligomers reference ONE Entity repeatedly and count as one;
       a ``water`` Entity is not a binding partner), or
     * the entry contains a ``non-polymer`` entity (ligand, drug), or
     * the entry contains a nucleic-acid entity (DNA, RNA, or the
       DNA/RNA hybrid polymer type).
   Metal, ligand and oligomer metadata ride along as labels; see
   ``trizod.dataset.composition``.
2. Drop every row whose BMRB entry is on the committed withdrawn/unlisted
   list (``trizod/dataset/data/withdrawn_entries.tsv``, see
   :mod:`trizod.dataset.withdrawn`; audit I-06). This happens before
   deduplication, so a withdrawn chain lands in no tier pool, no training set
   and no test set, and a live entry with the identical sequence takes over
   as representative. The build reads the file only; it never calls BMRB.
3. For every row in the per-tier scores.json, attach the bound flag and
   compute a per-row quality score:
       quality = (bbshift_positions_post * bbshift_types_post)
                  − (max |POTENCI residual offset|, in sigma)
   Higher is better.  In ties, prefer the strictest tier.
   The offsets come out of ``off_<atom>_sigma`` (sigma) and
   ``total_off_<atom>_ppm`` (ppm) via :mod:`trizod.offsets`; the two are in
   different units and are never mixed.
4. Drop rows with no sequence / sequence < 20 residues, rows that are
   flagged "bound", and placeholder sequences whose literal-"X" fraction
   exceeds ``MAX_UNKNOWN_FRACTION`` (audit I-08). The drop counts are
   reported in that order and are mutually exclusive.
5. Within each tier, group by exact ``seq`` and pick the highest-quality
   representative per sequence.  Carry the ranked list along so an
   external pipeline (mmseqs cluster representative override) can pick
   the best member of a similarity cluster too.
6. Write to ``<work-dir>/final_dataset/<tier>/``:
       <tier>.fasta            — deduplicated sequences, one per cluster
       <tier>_all_ranked.tsv   — every entry kept, with quality_score,
                                  cluster_repr flag, and metadata
       <tier>_summary.json     — counts (kept, dropped, dedup ratio)

Usage
-----
    uv run python -m trizod.dataset.build [--work-dir DIR] [--root DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from trizod.dataset.composition import detect_bound
from trizod.dataset.paths import resolve_paths
from trizod.dataset.withdrawn import WITHDRAWN_TSV, load_withdrawn
from trizod.offsets import max_abs_offset, off_sigma_col, total_off_ppm_col

TIERS = ["unfiltered", "tolerant", "moderate", "strict"]
TIER_RANK = {"strict": 4, "moderate": 3, "tolerant": 2, "unfiltered": 1}
ATOMS = ["C", "CA", "CB", "H", "HA", "HB", "N"]

#: Offset columns read out of ``scores.json``, built from the shared helpers so
#: a rename cannot desynchronise this reader from the emitter (it did once: the
#: reader kept the pre-rename spelling, ``.get()`` returned None, ``or 0.0``
#: turned every offset into a perfect zero, and 58 of 12,745 exact-sequence
#: groups silently elected a different representative).
POTENCI_SIGMA_COLS = [off_sigma_col(a) for a in ATOMS]
TOTAL_PPM_COLS = [total_off_ppm_col(a) for a in ATOMS]

MIN_SEQ_LEN = 20

#: Placeholder gate (NMR-STAR audit I-08). Many 1990s depositions give the
#: polymer sequence as the literal unknown residue "X" everywhere except the
#: assigned residues (bmr1712: 250 X then SRDVHSKVISEE). ``MIN_SEQ_LEN`` counts
#: those X's, and MMseqs2 cannot align them, so a mostly-X chain both passes the
#: length screen and slips through leakage control (505_1_1_1, a Cu/Zn SOD that
#: is 111 of 118 X, hides a homolog of the CheZOD117 chain 15711_1_1_1). A row
#: whose X fraction EXCEEDS this value is dropped from every tier; exactly 0.2
#: is kept. 0.2 is the tolerant tier's ``max-x-fraction`` (``filter_defaults``
#: in ``trizod/trizod.py``), counted the same way (literal upper-case "X" over
#: the full sequence length), so the gate only changes the unfiltered tier.
#: Other non-amino-acid characters ("*", "Z", "'") are not counted and are
#: passed through unchanged.
MAX_UNKNOWN_FRACTION = 0.2
UNKNOWN_RESIDUE = "X"


def unknown_fraction(seq: str) -> float:
    """Fraction of ``seq`` that is the literal unknown residue "X" (0 if empty)."""
    return seq.count(UNKNOWN_RESIDUE) / len(seq) if seq else 0.0


def is_placeholder(seq: str, max_fraction: float = MAX_UNKNOWN_FRACTION) -> bool:
    """True when ``seq`` is a placeholder: X fraction strictly above the limit."""
    return unknown_fraction(seq) > max_fraction


def resolve_withdrawn_list(path) -> Path:
    """The withdrawn/unlisted-entry list to apply (``None`` = the committed one)."""
    return WITHDRAWN_TSV if path is None else Path(path)


def drop_withdrawn(
    df: pd.DataFrame, withdrawn_ids
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (kept, dropped) on BMRB entry ID membership."""
    hit = df["entryID"].astype(str).isin(set(withdrawn_ids))
    return df.loc[~hit].copy(), df.loc[hit].copy()


def build_composition_cache(pkl_dir: Path) -> dict:
    """Load every BMRB pkl once and return entryID -> composition dict.

    Classifier failures are collected and re-raised at the end rather than
    swallowed. The previous bare ``except Exception`` wrote
    ``{"is_bound": True, "error": ...}`` per failing entry, which meant a stale
    ``tmp/bmrb_entries/`` cache produced a ``_composition_cache.csv`` with NaN
    on every composition column for thousands of entries — and the build still
    exited 0.
    """
    cache: dict[str, dict] = {}
    failures: list[tuple[str, str]] = []
    pkls = list(pkl_dir.glob("*.pkl"))
    print(f"Loading composition info from {len(pkls)} BMRB pkl files...")
    for i, pkl_path in enumerate(pkls):
        eid = pkl_path.stem
        try:
            with pkl_path.open("rb") as f:
                entry = pickle.load(f)
            cache[eid] = detect_bound(entry)
        except Exception as exc:
            failures.append((eid, f"{type(exc).__name__}: {exc}"))
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1} / {len(pkls)} processed")
    if failures:
        shown = "\n".join(f"    {eid}: {msg}" for eid, msg in failures[:10])
        more = f"\n    ... and {len(failures) - 10} more" if len(failures) > 10 else ""
        raise RuntimeError(
            f"composition classifier failed on {len(failures)} of {len(pkls)} "
            f"BMRB pickles in {pkl_dir}:\n{shown}{more}\n"
            "A stale pickle cache is the usual cause — delete the directory and "
            "let the pipeline re-parse the .str files."
        )
    return cache


def load_tier_scores(tier: str, release: Path) -> pd.DataFrame:
    rows = []
    with open(release / tier / "scores.json") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            # Both raise if the column is absent -- a scores.json without it is
            # schema drift, and the only safe reaction is to stop. They return
            # None only when every atom's offset is null (nothing measurable),
            # which is a real "no penalty" and is spelled out below rather than
            # arrived at by coercing a missing key.
            max_potenci_off = max_abs_offset(r, POTENCI_SIGMA_COLS)
            max_total_off_ppm = max_abs_offset(r, TOTAL_PPM_COLS)
            rows.append(
                {
                    "ID": r["ID"],
                    "entryID": r["entryID"],
                    "stID": r["stID"],
                    "entity_assemID": r["entity_assemID"],
                    "entityID": r["entityID"],
                    "seq": r["seq"] or "",
                    "len": len(r["seq"] or ""),
                    "n_bb_pos": r["bbshift_positions_post"] or 0,
                    "n_bb_types": r["bbshift_types_post"] or 0,
                    "total_bbshifts": r["total_bbshifts"] or 0,
                    # sigma, and the tie-break term of quality_score below.
                    "max_potenci_off": (
                        0.0 if max_potenci_off is None else max_potenci_off
                    ),
                    # ppm, and the honest answer to "how badly was this chain
                    # re-referenced": the POTENCI residual is part of the
                    # correction too, so the LACS term alone understates it.
                    "max_total_off_ppm": (
                        0.0 if max_total_off_ppm is None else max_total_off_ppm
                    ),
                    "entity_name": r.get("entity_name") or "",
                    "ionic_strength": r.get("ionic_strength"),
                    "pH": r.get("pH"),
                    "temperature": r.get("temperature"),
                }
            )
    df = pd.DataFrame(rows)
    df["tier"] = tier
    df["tier_rank"] = TIER_RANK[tier]
    return df


def compute_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Lexicographic quality: higher tier_rank → more shifts → smaller residual.

    We expose all three components plus a flat composite score that is
    monotone in each (with sensible scaling) so it can be used downstream.
    """
    df = df.copy()
    df["shift_volume"] = df["n_bb_pos"] * df["n_bb_types"]
    df["quality_score"] = (
        df["tier_rank"] * 1_000_000.0
        + df["shift_volume"].astype(float)
        - df["max_potenci_off"].astype(float)
    )
    return df


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="dataset build dir (default: <root>/data/interim/build)",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="repository root (default: auto-detected)",
    )
    ap.add_argument(
        "--exclude-homo-oligomers",
        action="store_true",
        help=(
            "also drop rows whose entity appears on more than one _Entity_assembly "
            "record (n_copies >= 2). OFF by default: oligomeric state is annotated, "
            "not filtered, and n_copies is a conservative lower bound."
        ),
    )
    ap.add_argument(
        "--withdrawn-list",
        type=Path,
        default=None,
        help=(
            "TSV of withdrawn/unlisted BMRB entries to drop before dedup "
            f"(default: the committed {WITHDRAWN_TSV.name}; refresh it with "
            "`python -m trizod.dataset.withdrawn`)"
        ),
    )
    ap.add_argument(
        "--max-unknown-fraction",
        type=float,
        default=MAX_UNKNOWN_FRACTION,
        help=(
            "drop placeholder chains whose literal-'X' fraction exceeds this "
            f"(default: {MAX_UNKNOWN_FRACTION}; 1.0 disables the gate)"
        ),
    )
    return ap


def main(argv=None):
    args = make_parser().parse_args(argv)
    paths = resolve_paths(args.work_dir, args.root)
    out = paths.final_dataset

    out.mkdir(parents=True, exist_ok=True)

    comp = build_composition_cache(paths.pkl_dir)
    print(f"Composition cache built: {len(comp)} entries")
    if not comp:
        raise SystemExit(f"no BMRB pkl files found in {paths.pkl_dir}")

    comp_df = pd.DataFrame([{"entryID": eid, **c} for eid, c in comp.items()])
    comp_df.to_csv(out / "_composition_cache.csv", index=False)
    print(f"Wrote {out / '_composition_cache.csv'} (n={len(comp_df)})")

    n_bound_global = int(comp_df["is_bound"].sum())
    n_total_global = len(comp_df)
    print(
        f"  global: {n_bound_global}/{n_total_global} "
        f"({n_bound_global / n_total_global:.1%}) flagged bound/multi-molecule"
    )

    # Load every tier and concatenate, marking each row with its tier.
    # Note: the tiers are nested (strict ⊂ moderate ⊂ tolerant ⊂
    # unfiltered) so a single (entryID, stID, entityAssemID, entityID)
    # row appears in multiple per-tier scores.json files — that
    # multiplicity is what lets us pick the strictest tier each row
    # belongs to.
    all_rows = []
    raw_counts = {}
    for tier in TIERS:
        df_t = load_tier_scores(tier, paths.scored)
        raw_counts[tier] = len(df_t)
        all_rows.append(df_t)
    all_df = pd.concat(all_rows, ignore_index=True)

    # Withdrawn / unlisted BMRB entries (audit I-06) go before anything else,
    # in particular before dedup: a live entry with the identical sequence then
    # becomes the representative instead of the sequence vanishing with it.
    withdrawn_path = resolve_withdrawn_list(args.withdrawn_list)
    withdrawn = load_withdrawn(withdrawn_path)
    all_df, dropped_wd = drop_withdrawn(all_df, withdrawn)
    n_dropped_withdrawn_global = len(dropped_wd)
    withdrawn_dropped_ids = sorted(dropped_wd["ID"].unique())
    n_withdrawn_by_tier = dropped_wd.groupby("tier").size().to_dict()
    withdrawn_list_info = {
        "file": withdrawn_path.name,
        "sha256": hashlib.sha256(withdrawn_path.read_bytes()).hexdigest(),
        "n_entries": len(withdrawn),
        "date_checked": max((e.date_checked for e in withdrawn.values()), default=None),
    }
    print(
        f"  withdrawn/unlisted BMRB entries: {len(withdrawn)} on "
        f"{withdrawn_path.name} (checked {withdrawn_list_info['date_checked']}); "
        f"dropped {n_dropped_withdrawn_global} rows "
        f"({len(withdrawn_dropped_ids)} chains) before dedup"
        + (f": {withdrawn_dropped_ids}" if withdrawn_dropped_ids else "")
    )

    all_df = all_df.merge(comp_df, on="entryID", how="left", suffixes=("", "_comp"))
    # The left merge upcasts is_bound to object dtype whenever a scored entryID
    # has no composition match (NaN introduced). ``~`` on an object column does
    # Python bitwise invert (~True == -2, ~False == -1 — both truthy), which
    # would silently defeat the bound-complex filter below. Coerce back to bool.
    all_df["is_bound"] = all_df["is_bound"].fillna(True).astype(bool)
    all_df = compute_quality(all_df)

    # Row screens, counted in order (short -> bound -> placeholder) so each
    # count is what that screen adds on top of the ones before it.
    is_short = all_df["len"] < MIN_SEQ_LEN
    universal_keep = ~is_short & ~all_df["is_bound"]
    n_dropped_short_global = int(is_short.sum())
    n_dropped_bound_global = int((~is_short & all_df["is_bound"]).sum())
    max_unknown = args.max_unknown_fraction
    is_placeholder_row = (
        all_df["seq"].map(lambda seq: is_placeholder(seq, max_unknown)).astype(bool)
    )
    dropped_placeholder = universal_keep & is_placeholder_row
    n_dropped_placeholder_global = int(dropped_placeholder.sum())
    placeholder_dropped_ids = sorted(all_df.loc[dropped_placeholder, "ID"].unique())
    universal_keep &= ~is_placeholder_row
    print(
        f"  placeholder sequences (X fraction > {max_unknown}): dropped "
        f"{n_dropped_placeholder_global} rows ({len(placeholder_dropped_ids)} chains)"
    )
    n_dropped_oligomer_global = 0
    if args.exclude_homo_oligomers:
        is_oligomer = all_df["n_copies"].fillna(1).astype(float) >= 2
        n_dropped_oligomer_global = int((universal_keep & is_oligomer).sum())
        universal_keep &= ~is_oligomer
        print(
            f"  --exclude-homo-oligomers: dropping {n_dropped_oligomer_global} "
            "rows with n_copies >= 2"
        )
    kept_all = all_df.loc[universal_keep].copy()

    # Reduce each (ID) appearance to its strictest tier — quality_score
    # already has tier_rank * 1e6 baked in, so the highest-quality row
    # per ID is necessarily in the strictest tier where it was scored.
    kept_all = kept_all.sort_values("quality_score", ascending=False)
    kept_all = kept_all.drop_duplicates(subset=["ID"], keep="first")

    # Per-sequence representative: the highest-quality ID across all
    # tiers carries the canonical name used in every per-tier FASTA so
    # that mmseqs clusterupdate sees a consistent identifier set.
    # ID is a deterministic final tiebreak so ties on quality_score resolve
    # reproducibly (pandas' sort is not stable) rather than by chance.
    kept_all = kept_all.sort_values(
        ["seq", "quality_score", "ID"], ascending=[True, False, True]
    )
    kept_all["seq_rank_global"] = kept_all.groupby("seq").cumcount() + 1
    kept_all["is_global_seq_repr"] = kept_all["seq_rank_global"] == 1
    global_repr = kept_all.loc[
        kept_all["is_global_seq_repr"], ["seq", "ID", "tier"]
    ].rename(columns={"ID": "global_repr_ID", "tier": "global_repr_tier"})
    kept_all = kept_all.merge(global_repr, on="seq", how="left")

    overall_summary = {
        "composition_cache_size": int(len(comp_df)),
        "composition_flagged_bound": int(n_bound_global),
        "withdrawn_list": withdrawn_list_info,
        "global_dropped_withdrawn": n_dropped_withdrawn_global,
        "withdrawn_dropped_ids": withdrawn_dropped_ids,
        "global_dropped_short": n_dropped_short_global,
        "global_dropped_bound": n_dropped_bound_global,
        "max_unknown_fraction": max_unknown,
        "global_dropped_placeholder": n_dropped_placeholder_global,
        "placeholder_dropped_ids": placeholder_dropped_ids,
        "global_dropped_homo_oligomer": n_dropped_oligomer_global,
        "global_unique_sequences": int(global_repr["seq"].nunique()),
        "tiers": {},
    }

    for tier in TIERS:
        print(f"\n=== Processing tier: {tier} ===")
        # Tier is the cumulative (nested) set: every entry that passes
        # at least this tier.  We achieve that by selecting rows whose
        # strictest-tier rank is ≥ TIER_RANK[tier].
        tier_threshold = TIER_RANK[tier]
        df_t = kept_all[kept_all["tier_rank"] >= tier_threshold].copy()
        n_initial = raw_counts[tier]
        in_tier = all_df["tier"] == tier
        n_withdrawn = int(n_withdrawn_by_tier.get(tier, 0))
        n_short = int((in_tier & is_short).sum())
        n_bound = int((in_tier & ~is_short & all_df["is_bound"]).sum())
        n_placeholder = int((in_tier & dropped_placeholder).sum())

        df_t = df_t.sort_values(
            ["seq", "quality_score", "ID"], ascending=[True, False, True]
        )
        df_t["seq_rank_tier"] = df_t.groupby("seq").cumcount() + 1
        df_t["is_seq_repr_tier"] = df_t["seq_rank_tier"] == 1

        tier_dir = out / tier
        tier_dir.mkdir(parents=True, exist_ok=True)

        keep_cols = [
            "ID", "entryID", "stID", "entity_assemID", "entityID",
            "tier", "tier_rank", "len", "n_bb_pos", "n_bb_types",
            "total_bbshifts", "shift_volume", "max_potenci_off",
            "max_total_off_ppm", "quality_score",
            "seq_rank_tier", "is_seq_repr_tier",
            "global_repr_ID", "global_repr_tier",
            "is_bound", "has_non_polymer", "has_nucleic",
            "has_metal", "has_metal_ion", "has_metal_cofactor",
            "has_water", "has_other_ligand",
            "metal_comp_ids", "ligand_comp_ids", "ligand_names",
            "multi_protein_assembly", "n_entities",
            "n_entity_assembly_rows", "n_copies", "has_conformational_isomer",
            "entity_name",
            "ionic_strength", "pH", "temperature", "seq",
        ]  # fmt: skip
        df_t[keep_cols].to_csv(
            tier_dir / f"{tier}_all_ranked.tsv", sep="\t", index=False
        )

        # FASTA: one record per sequence.  Header = global_repr_ID
        # (stable across tiers) but the row carrying it is the
        # highest-quality entry inside THIS tier.  Subsequent
        # clusterupdate steps therefore see the same identifier
        # for the same sequence in old (strict) and new (next-tier) DBs.
        repr_rows = df_t[df_t["is_seq_repr_tier"]].copy()
        repr_rows = repr_rows.drop_duplicates(subset=["seq"], keep="first")
        fasta_path = tier_dir / f"{tier}.fasta"
        with fasta_path.open("w") as fh:
            for _, row in repr_rows.iterrows():
                fh.write(
                    f">{row['global_repr_ID']} "
                    f"tier={row['tier']} "
                    f"best_in_tier={row['ID']} "
                    f"quality={row['quality_score']:.1f} "
                    f"n_bb_pos={int(row['n_bb_pos'])} "
                    f"n_bb_types={int(row['n_bb_types'])}\n{row['seq']}\n"
                )

        n_unique_seq = repr_rows["seq"].nunique()
        tier_summary = {
            "tier": tier,
            "initial_rows": int(n_initial),
            "dropped_withdrawn": n_withdrawn,
            "dropped_short": n_short,
            "dropped_bound": n_bound,
            "dropped_placeholder": n_placeholder,
            "kept_rows_after_filters": int(len(df_t)),
            "unique_sequences_kept": int(n_unique_seq),
            "fraction_dedup": (1 - n_unique_seq / len(df_t)) if len(df_t) else 0.0,
            "min_seq_len": MIN_SEQ_LEN,
            "max_unknown_fraction": max_unknown,
        }
        with open(tier_dir / f"{tier}_summary.json", "w") as fh:
            json.dump(tier_summary, fh, indent=2)
        overall_summary["tiers"][tier] = tier_summary
        print(
            f"  initial rows               : {n_initial}\n"
            f"  dropped (withdrawn entry)  : {n_withdrawn}\n"
            f"  dropped (seq < {MIN_SEQ_LEN} aa)      : {n_short}\n"
            f"  dropped (bound/multi-mol)  : {n_bound}\n"
            f"  dropped (X frac > {max_unknown})  : {n_placeholder}\n"
            f"  kept rows                  : {len(df_t)}\n"
            f"  unique sequences (FASTA)   : {n_unique_seq} "
            f"({(n_unique_seq / len(df_t)) if len(df_t) else 0:.1%} of kept rows)\n"
            f"  wrote {fasta_path.name} and {tier}_all_ranked.tsv"
        )

    with open(out / "final_dataset_summary.json", "w") as fh:
        json.dump(overall_summary, fh, indent=2)
    print(f"\nWrote {out / 'final_dataset_summary.json'}")


if __name__ == "__main__":
    main()
