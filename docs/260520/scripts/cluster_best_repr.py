#!/usr/bin/env python3
"""Override mmseqs cluster representatives with the best quality member.

After ``run_mmseqs_pipeline.py`` finishes, mmseqs assigns the cluster
representative based on its internal similarity-graph reasoning.  The
clusterupdate ordering already biases the choice towards strict-tier
entries (as the original TriZOD report does), but within a tier the
choice is not score-aware.

This script reads:
  * ``train_<tier>_clu.tsv``           — two-column mmseqs cluster file
                                          (cluster_repr, member).
  * ``final_dataset/<tier>/<tier>_all_ranked.tsv`` — per-entry quality
                                          scores written by
                                          build_final_dataset.py.

…and writes for every tier:
  * ``train_<tier>_clu_best.tsv``      — same as input plus columns:
        best_repr   — the highest-quality_score member of the cluster,
        quality_diff — quality(best) − quality(input_repr),
  * ``train_<tier>_best.fasta``        — FASTA whose headers carry the
        highest-quality entry of each cluster (one sequence per
        cluster).
  * ``cluster_repr_overrides_<tier>.tsv`` — only the clusters where the
        score-best representative differs from mmseqs' pick (useful for
        understanding what changed).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MMSEQS = ROOT / "docs" / "260520" / "data" / "mmseqs"
FINAL = ROOT / "docs" / "260520" / "data" / "final_dataset"
TIERS = ["strict", "moderate", "tolerant", "unfiltered"]


def load_ranked(tier: str) -> pd.DataFrame:
    p = FINAL / tier / f"{tier}_all_ranked.tsv"
    return pd.read_csv(p, sep="\t")


def load_clusters(tier: str) -> pd.DataFrame:
    p = MMSEQS / f"train_{tier}_clu.tsv"
    return pd.read_csv(p, sep="\t", header=None, names=["cluster_repr", "member"])


def main():
    for tier in TIERS:
        print(f"=== {tier} ===")
        clu = load_clusters(tier)
        ranked = load_ranked(tier)

        # We keyed FASTAs by global_repr_ID; mmseqs uses that as both
        # cluster_repr and member labels.  Map to the row's quality
        # score via global_repr_ID.
        quality = (
            ranked.loc[ranked["is_seq_repr_tier"]]
            .drop_duplicates(subset=["global_repr_ID"], keep="first")
            [["global_repr_ID", "ID", "tier", "quality_score", "seq",
              "n_bb_pos", "n_bb_types", "max_potenci_off", "max_lacs_off"]]
            .rename(columns={
                "global_repr_ID": "member",
                "ID": "member_ID",
                "tier": "member_tier",
                "quality_score": "member_quality",
            })
        )

        merged = clu.merge(quality, on="member", how="left")
        if merged["member_quality"].isna().any():
            n_miss = int(merged["member_quality"].isna().sum())
            print(f"  WARNING: {n_miss} cluster members not in ranked table")

        # Pick best member per cluster
        merged = merged.sort_values(
            ["cluster_repr", "member_quality"], ascending=[True, False]
        )
        best = merged.drop_duplicates(subset=["cluster_repr"], keep="first")
        rename = best.set_index("cluster_repr")["member"].to_dict()
        rename_quality = best.set_index("cluster_repr")["member_quality"].to_dict()

        # Annotate every member row with the cluster's best
        merged["best_repr"] = merged["cluster_repr"].map(rename)
        merged["best_repr_quality"] = merged["cluster_repr"].map(rename_quality)
        merged["quality_diff_vs_mmseqs"] = (
            merged["best_repr_quality"] - merged["member_quality"]
        )
        # quality_diff only meaningful on the mmseqs cluster_repr row
        mmseqs_repr_rows = merged[merged["member"] == merged["cluster_repr"]].copy()
        mmseqs_repr_rows["overrides_to"] = mmseqs_repr_rows.apply(
            lambda r: None if r["member"] == r["best_repr"] else r["best_repr"], axis=1,
        )

        # Save augmented clusters
        merged_cols = [
            "cluster_repr", "member", "best_repr",
            "member_quality", "best_repr_quality", "quality_diff_vs_mmseqs",
            "member_tier", "member_ID", "n_bb_pos", "n_bb_types",
            "max_potenci_off", "max_lacs_off",
        ]
        merged[merged_cols].to_csv(
            MMSEQS / f"train_{tier}_clu_best.tsv", sep="\t", index=False
        )

        # Save best-representative FASTA
        # Build a lookup member -> sequence (one of the rows in `quality`)
        seq_by_member = quality.set_index("member")["seq"].to_dict()
        out_fa = MMSEQS / f"train_{tier}_best.fasta"
        with out_fa.open("w") as fh:
            for cluster_repr, row in best.set_index("cluster_repr").iterrows():
                m = row["member"]
                seq = seq_by_member.get(m)
                if seq is None:
                    continue
                fh.write(
                    f">{m} cluster_repr={cluster_repr} "
                    f"quality={row['member_quality']:.1f} "
                    f"tier={row['member_tier']}\n{seq}\n"
                )

        # Save overrides (cases where best != mmseqs)
        overrides = mmseqs_repr_rows[mmseqs_repr_rows["overrides_to"].notna()][
            ["cluster_repr", "best_repr", "member_quality",
             "best_repr_quality", "quality_diff_vs_mmseqs", "member_tier"]
        ].rename(columns={
            "cluster_repr": "mmseqs_repr",
            "best_repr": "score_best_repr",
            "member_quality": "mmseqs_repr_quality",
            "best_repr_quality": "score_best_quality",
            "member_tier": "mmseqs_repr_tier",
        })
        overrides.to_csv(
            MMSEQS / f"cluster_repr_overrides_{tier}.tsv", sep="\t", index=False
        )
        n_clu = best["cluster_repr"].nunique()
        n_overrides = len(overrides)
        print(
            f"  clusters: {n_clu}, "
            f"overrides (best != mmseqs pick): {n_overrides} "
            f"({(n_overrides / n_clu):.1%})"
        )
        if n_overrides:
            top = overrides.sort_values("quality_diff_vs_mmseqs", ascending=False).head(3)
            print("  top quality_diff overrides:")
            for _, r in top.iterrows():
                print(
                    f"    {r['mmseqs_repr']} (q={r['mmseqs_repr_quality']:.1f}) "
                    f"→ {r['score_best_repr']} (q={r['score_best_quality']:.1f}), "
                    f"Δ={r['quality_diff_vs_mmseqs']:.1f}"
                )


if __name__ == "__main__":
    main()
