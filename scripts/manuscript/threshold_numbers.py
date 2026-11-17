"""Write numbers_threshold.tex: the macros of the G-score threshold framework.

Reads results.json of scripts/validation/threshold_calibration/framework/compute.py:
  --framework-dir / $TRIZOD_THRESHOLD_FRAMEWORK_DIR
      default <data root>/data/interim/threshold-framework-port-2026-11-17
and writes one \\newcommand per number, so the text never types a calibration value.

The framework replaces the single-cut MCC calibration at 0.40 of 16 Nov 2026: three cuts from one
rule (posterior 0.10 / 0.50 / 0.90 at the stated reference-consensus prevalence of the strict-tier
unbound chains), the consensus likelihood ratio (LR) at each published value, and a 95 % joint
cluster-bootstrap interval for everything. Estimates ("Est") are those of the primary estimator (log
LR interpolated between isotonic block midpoints); "Step" macros are the isotonic step sensitivity.
Paper values: \\thrPaperOrd / \\thrPaperBin / \\thrPaperDis; \\thr<Edge>InCI says whether the paper value
lies inside the 95 % interval of its estimate (the frozen reporting rule).

Macros of the former calibration that no longer exist are listed in DROPPED (printed on every run).

Run: uv run python scripts/manuscript/threshold_numbers.py --out <draft>/numbers_threshold.tex
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from trizod import paths

DEFAULT_DIR = "threshold-framework-port-2026-11-17"
EDGES = (
    ("Ord", "ordered_edge", "0.10", "0.15"),
    ("Bin", "binary_cut", "0.50", "0.35"),
    ("Dis", "disordered_edge", "0.90", "0.70"),
)
TIER_TAGS = (
    ("Unfiltered", "tier_unfiltered+"),
    ("Tolerant", "tier_tolerant+"),
    ("Moderate", "tier_moderate+"),
    ("Strict", "tier_strict+"),
    ("Cal", "analysis_set"),
    ("Train", "train"),
    ("Test", "test_trizod"),
)
# macros of the MCC-at-0.40 calibration (threshold-calibration-2026-11-16) that this file no longer
# writes; nThrNoeChains, nThrNoeResidues and rhoThrNoe keep their names and meaning (all same-entry
# hetNOE residues, now re-joined to the build).
DROPPED = (
    "nThrFilteredChains", "nThrMappedChains",
    "nThrSetAChains", "nThrSetAResidues", "thrSetATstar", "thrSetATstarLo", "thrSetATstarHi", "aucThrSetA",
    "mccThrSetATrainNew", "mccThrSetATrainOld", "mccThrSetATestNew", "mccThrSetATestOld",
    "nThrSetBChains", "nThrSetBResidues", "thrSetBTstar", "thrSetBTstarLo", "thrSetBTstarHi", "aucThrSetB",
    "mccThrSetBTrainNew", "mccThrSetBTrainOld", "mccThrSetBTestNew", "mccThrSetBTestOld",
    "thrValley", "thrMixture",
    "thrKappaAllTstar", "thrKappaAllTstarLo", "thrKappaAllTstarHi", "nThrKappaAllPairs",
    "thrKappaSameTstar", "thrKappaSameTstarLo", "thrKappaSameTstarHi", "nThrKappaSamePairs",
    "kappaThrSameNew", "kappaThrSameOld", "pctThrDisorderedTrainNew",
    "thrNoeTstar", "thrNoeTstarLo", "thrNoeTstarHi", "aucThrNoe", "mccThrNoeNew", "mccThrNoeOld",
)  # fmt: skip


def _root() -> Path:
    for p in [paths.ROOT, *paths.ROOT.parents]:
        if (p / "data" / "interim").is_dir():
            return p
    return paths.ROOT


def _dir(flag: Path | None) -> Path:
    if flag:
        return flag.expanduser()
    v = os.environ.get("TRIZOD_THRESHOLD_FRAMEWORK_DIR")
    return Path(v).expanduser() if v else _root() / "data" / "interim" / DEFAULT_DIR


def num(x, d: int = 2) -> str:
    if x is None:
        return "NA"
    s = f"{x:.{d}f}"
    return s.replace("-", "\\textminus{}") if s.startswith("-") else s


def f2(x) -> str:
    return num(x, 2)


def f3(x) -> str:
    return num(x, 3)


def pct(x, d: int = 1) -> str:
    return "NA" if x is None else num(100 * x, d)


def lr(x) -> str:
    """Likelihood ratios: 2 decimals below 1, 1 decimal below 10, else integer."""
    if x is None:
        return "NA"
    return num(x, 2 if x < 1 else 1 if x < 10 else 0)


def n(x) -> str:
    return "NA" if x is None else f"{int(x):,}"


def yes(b) -> str:
    return "yes" if b else "no"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--framework-dir", type=Path, help="output dir of framework/compute.py"
    )
    args = ap.parse_args()
    fdir = _dir(args.framework_dir)
    R = json.loads((fdir / "results.json").read_text())
    C = R["calibration"]
    est = C["primary_estimator"]
    E = C["estimators"]
    m: list[tuple[str, str, str]] = []

    def add(name: str, value: str, what: str) -> None:
        m.append((name, value, what))

    def add3(name: str, v: list, fmt, what: str) -> None:
        """[point, lo, hi] -> name, nameLo, nameHi."""
        add(name, fmt(v[0]), what)
        add(f"{name}Lo", fmt(v[1]), "95 % CI, lower")
        add(f"{name}Hi", fmt(v[2]), "95 % CI, upper")

    # ---- analysis set and references
    S = R["analysis_set"]
    acc = {a["step"]: a for a in S["accounting"]}
    first = S["accounting"][0]
    add(
        "nThrStrictTierChains",
        n(first["chains"]),
        f"chains scored in the calibration tier(s) ({R['settings']['tier']} and stricter)",
    )
    add(
        "nThrStrictBoundRemoved",
        n(acc.get("is_bound False", {}).get("removed")),
        "of these, bound chains removed",
    )
    add("nThrStrictStateRemoved",
        n(sum(acc.get(k, {}).get("removed", 0) for k in ("physical_state not denatured/unfolded", "cosolvent_evidence False"))),
        "removed as denatured/unfolded or with cosolvent evidence")  # fmt: skip
    add(
        "nThrCalChains",
        n(S["chains"]),
        "analysis set: unbound strict-tier chains, all splits",
    )
    add("nThrCalFitChains", n(S["fit_chains"]), "analysis set, fit (non-test) chains")
    add("nThrCalTestChains", n(S["test_chains"]), "analysis set, held-out test chains")
    pi_counts = C["counts_fit"]["PI"]
    add(
        "nThrCalFitResidues",
        n(pi_counts["residues"]),
        "scored residues of the fit chains (prevalence histogram)",
    )
    add(
        "nThrCalFitClusters",
        n(pi_counts["clusters"]),
        "sequence clusters of the fit chains (bootstrap units)",
    )
    for r in ("A", "B", "C"):
        c = C["counts_fit"][r]
        add(
            f"nThrRef{r}Chains", n(c["chains"]), f"fit chains with reference-{r} labels"
        )
        add(
            f"nThrRef{r}Clusters",
            n(c["clusters"]),
            f"sequence clusters with reference-{r} labels",
        )
        add(
            f"nThrRef{r}Residues",
            n(c["residues"]),
            f"labelled fit residues, reference {r}",
        )
        add(
            f"nThrRef{r}Positives",
            n(c["positives"]),
            f"of these disordered, reference {r}",
        )
        add(
            f"aucThrRef{r}",
            f2(C["auc"][r][0]),
            f"ROC AUC of G, reference {r} (fit residues)",
        )
        t = R.get("heldout", {}).get("test", {}).get(r, {})
        if "clusters" in t:
            add(
                f"nThrRef{r}TestClusters",
                n(t["clusters"]),
                f"held-out test clusters, reference {r}",
            )
            add(
                f"mccThrRef{r}Test",
                f3(t["MCC_binary"][0]),
                f"MCC of G >= \\thrPaperBin against reference {r}, test chains",
            )
    j = R["join"]
    add(
        "nThrJoinXrayResidues",
        n(j["xray"]["rows_joined"]),
        "X-ray/DisProt-labelled residues joined to the build (chain id + sequence)",
    )
    add(
        "nThrJoinXrayChains",
        n(j["xray"]["chains_joined"]),
        "chains with X-ray/DisProt labels joined to the build",
    )
    add(
        "nThrJoinNoeResidues",
        n(j["hetnoe"]["rows_joined"]),
        "hetNOE-labelled residues joined to the build",
    )
    add(
        "nThrNoeChains",
        n(j["hetnoe"]["chains_joined"]),
        "chains with hetNOE in the same entry (joined to the build)",
    )
    add(
        "nThrNoeResidues",
        n(j["hetnoe"]["rows_joined"]),
        "residues with hetNOE (joined to the build)",
    )
    sp = R.get("hetnoe_spearman", {})
    if "all_joined" in sp and "rho" in sp["all_joined"]:
        add3(
            "rhoThrNoe",
            [sp["all_joined"]["rho"], *sp["all_joined"]["CI"]],
            f2,
            "Spearman rho of G and hetNOE, all joined residues",
        )
    if "analysis_set_fit" in sp and "rho" in sp["analysis_set_fit"]:
        add(
            "rhoThrNoeCal",
            f2(sp["analysis_set_fit"]["rho"]),
            "Spearman rho of G and hetNOE, analysis-set fit residues",
        )

    # ---- prevalence
    cons = E[est]["consensus"]
    add3(
        "thrPrior",
        cons["pi"],
        f2,
        "reference-consensus prevalence pi (prior-shift MLE, strict unbound fit chains)",
    )
    add("thrPriorThree", f3(cons["pi"][0]), "the same, three decimals")
    alt = C["prior_alternatives"]
    add(
        "thrPriorRG",
        f3(alt["rogan_gladen"][0]),
        "Rogan-Gladen prevalence (numerical check of the MLE)",
    )
    if "logit_mixture" in alt:
        add(
            "thrPriorMixture",
            f3(alt["logit_mixture"][0]),
            "logit-G two-component mixture weight (the former, replaced prior)",
        )
        mx = cons["thresholds_at_alternative_priors"].get("logit_mixture", {})
        for tag, _, lev, _ in EDGES:
            if lev in mx:
                add(
                    f"thr{tag}MixturePrior",
                    f2(mx[lev]["t"]),
                    f"cut at posterior {lev} under the mixture prior",
                )

    # ---- cuts, paper values and the reporting rule
    pc = C["paper_check"][est]
    for tag, key, lev, paper in EDGES:
        add(f"thrPaper{tag}", paper, f"published value at posterior {lev}")
        add3(
            f"thr{tag}Est",
            [cons["cuts"][lev]["t"], *cons["cuts"][lev]["CI"]],
            f2,
            f"G where the posterior reaches {lev} ({est})",
        )
        add(f"thr{tag}EstThree", f3(cons["cuts"][lev]["t"]), "the same, three decimals")
        st = E["step"]["consensus"]["cuts"][lev]
        add3(
            f"thr{tag}Step",
            [st["t"], *st["CI"]],
            f2,
            "the same, isotonic step estimator (sensitivity)",
        )
        add(
            f"thr{tag}InCI",
            yes(pc[key]["inside_CI"]),
            f"paper value {paper} inside the 95 % CI of its estimate",
        )
        add(
            f"thr{tag}Rule",
            f2(pc[key]["rule_value"]),
            "value under the frozen rule (paper value if inside the CI)",
        )
        add(
            f"pctThr{tag}RoundsPaper",
            pct(pc[key]["share_rounds_to_paper"], 0),
            f"% of bootstrap cuts that round (0.05) to {paper}",
        )
    add(
        "pctThrDisBootLow",
        pct(cons.get("t_dis_boot_share_le_0.60"), 0),
        "% of bootstrap disordered edges <= 0.60",
    )
    add(
        "pctThrDisBootHigh",
        pct(cons.get("t_dis_boot_share_ge_0.85"), 0),
        "% of bootstrap disordered edges >= 0.85",
    )
    lv95 = cons["cuts"].get("0.95", {})
    add("thrNinetyFiveEst", f2(lv95.get("t")), "G where the posterior reaches 0.95")
    add(
        "pctThrNinetyFiveNA",
        pct(lv95.get("NA_share"), 1),
        "% of bootstraps in which 0.95 is never reached",
    )

    # ---- LR and posterior at the published values (LR prevalence-free; posterior at the stated pi)
    for tag, _, lev, paper in EDGES:
        v = cons["at_paper_values"][paper]
        add3(f"thrLR{tag}", v["LR"], lr, f"consensus LR at G = {paper}")
        add3(
            f"thrPost{tag}",
            v["posterior"],
            f2,
            f"posterior P(dis | G = {paper}) at the stated prevalence \\thrPrior",
        )
        vs = E["step"]["consensus"]["at_paper_values"][paper]
        add3(
            f"thrLRStep{tag}",
            vs["LR"],
            lr,
            f"consensus LR at G = {paper}, step estimator",
        )
        for r in ("A", "B", "C"):
            add(
                f"thrLR{tag}Ref{r}",
                lr(v["LR_per_reference"][r][0]),
                f"LR of reference {r} alone at G = {paper}",
            )
        add(
            f"thrLRLevel{tag}",
            lr(cons["lr_levels_at_pi"][lev]),
            f"LR that gives posterior {lev} at the stated prevalence",
        )
    add("thrLRStrong", "18.7", "ACMG/ClinGen 'strong' odds (Tavtigian 2018)")
    add("thrLRDisAboveStrong", yes((cons["at_paper_values"]["0.70"]["LR"][1] or 0) >= 18.7),
        "lower 95 % bound of the LR at 0.70 at or above 18.7")  # fmt: skip
    add3(
        "thrNeutralG",
        cons["neutral_point_LR1"],
        f2,
        "G where the consensus LR crosses 1",
    )
    ev = cons["evidence_edges"]
    for lab, tag in (
        ("supporting", "Supporting"),
        ("moderate", "Moderate"),
        ("strong", "Strong"),
    ):
        add(
            f"thrEvid{tag}G",
            f2(ev[lab]["G_ge"][0]),
            f"first G with consensus LR >= {ev[lab]['odds']}",
        )
        if ev[lab]["G_lt"][0] is not None:
            add(
                f"thrEvid{tag}OrdG",
                f2(ev[lab]["G_lt"][0]),
                f"consensus LR <= 1/{ev[lab]['odds']} below this G",
            )
    z = C["zone_lr"]["consensus"]
    for tag, zn in (
        ("Ord", "ordered"),
        ("Ind", "indeterminate"),
        ("Dis", "disordered"),
    ):
        add3(f"thrZoneLR{tag}", z[zn]["LR"], lr, f"consensus LR of the {zn} zone")
        add(
            f"thrZonePost{tag}",
            f2(z[zn]["posterior_at_pi"][0]),
            f"posterior of the {zn} zone at \\thrPrior",
        )

    # ---- sensitivity: pools, hetNOE field-aware cut
    for pool, tag in (
        ("structural_only", "StructOnly"),
        ("hetnoe_only", "NoeOnly"),
        ("equal_ABC", "EqualW"),
        ("B_excluded", "NoB"),
        ("hetnoe_field_aware", "FieldNoe"),
    ):
        d = E[est].get(pool)
        if not d:
            continue
        for etag, _, lev, _ in EDGES:
            add3(
                f"thr{etag}{tag}",
                [d["cuts"][lev]["t"], *d["cuts"][lev]["CI"]],
                f2,
                f"cut at posterior {lev}, pool {pool}",
            )
        add(f"thrPrior{tag}", f2(d["pi"][0]), f"prevalence under pool {pool}")
    for r in ("A", "B", "C"):
        d = E[est].get(f"{r}_only") if r != "C" else E[est].get("hetnoe_only")
        if d:
            add(
                f"thrRef{r}OwnCuts",
                " / ".join(f2(d["cuts"][lev]["t"]) for _, _, lev, _ in EDGES),
                f"reference {r} alone, at its own prevalence: cuts at 0.10 / 0.50 / 0.90",
            )

    # ---- precision check (looser tier, own LR and prevalence)
    P = R.get("precision_check")
    if P:
        add("thrPrecisionTier", P["tier"], "tier of the precision check (and stricter)")
        add(
            "nThrPrecisionFitChains",
            n(P["counts_fit"]["PI"]["chains"]),
            "fit chains of the precision check",
        )
        add3(
            "thrPriorPrecision",
            P[est]["pi"],
            f2,
            "prevalence of the precision-check set",
        )
        for tag, key, _, _ in EDGES:
            add3(
                f"thr{tag}Precision",
                [P[est][key]["t"], *P[est][key]["CI"]],
                f2,
                f"cut of the precision-check set ({key})",
            )

    # ---- transport: the same LR at other prevalences
    tr = {t["set"]: t for t in R.get("transport", [])}
    if "test_trizod" in tr:
        t = tr["test_trizod"]
        add(
            "thrPriorTest",
            f2(t["pi_em"]),
            "prevalence of the TriZOD test set under the calibration LR",
        )
        for tag, _, _, paper in EDGES:
            add(
                f"thrPost{tag}Test",
                f2(t[f"posterior_at_{paper}"]),
                f"posterior at G = {paper} on the TriZOD test set",
            )

    # ---- zone shares per tier at the published values
    zs = {(z["set"], z["cuts_from"]): z for z in R.get("zone_shares", [])}
    for tag, key in TIER_TAGS:
        z = zs.get((key, "paper_values"))
        if not z:
            continue
        add(f"pctThrZoneOrd{tag}", pct(z["ordered"]), f"% residues G < 0.15, {key}")
        add(
            f"pctThrZoneInd{tag}",
            pct(z["indeterminate"]),
            f"% residues 0.15 <= G < 0.70, {key}",
        )
        add(f"pctThrZoneDis{tag}", pct(z["disordered"]), f"% residues G >= 0.70, {key}")
        add(
            f"pctThrBinary{tag}",
            pct(z["binary_disordered"]),
            f"% residues G >= 0.35, {key}",
        )

    # ---- replicate agreement (same conditions) at the published values
    rp = R.get("replicates", {})
    for tag, key in (("", "analysis_set"), ("All", "all_unbound_chains")):
        v = rp.get(key, {}).get("paper_values")
        if not v:
            continue
        add3(
            f"kappaThrBin{tag}",
            v["kappa_binary"],
            f2,
            f"replicate kappa of G >= 0.35, same conditions, {key}",
        )
        add(
            f"kappaThrConfident{tag}",
            f3(v["kappa_confident_only"][0]),
            "kappa among pairs with both residues in a confident zone",
        )
        add(
            f"nThrRepPairs{tag}",
            n(v["n_pairs"]),
            "same-condition replicate entry pairs",
        )
        add(f"nThrRepResiduePairs{tag}", n(v["n_residue_pairs"]), "residue pairs")
        for ztag, zn in (
            ("Ord", "ordered"),
            ("Ind", "indeterminate"),
            ("Dis", "disordered"),
        ):
            add(
                f"pctThrFlip{ztag}{tag}",
                pct(v[f"{zn}_flip"][0]),
                f"% binary-call flips between replicates, {zn} zone",
            )

    # ---- k >= 10 usage check
    k = R.get("k_check")
    if k:
        add("thrKCheckTier", k["tier"], "tier of the k check (and stricter)")
        add(
            "pctThrLowK",
            pct(k["share_k_lt_10"]["release_all_chains"]),
            "% of release residues with k < 10",
        )
        add(
            "pctThrLowKCal",
            pct(k["share_k_lt_10"]["analysis_set"]),
            "% of analysis-set residues with k < 10",
        )
        kr = {r["k"]: r for r in k["replicates"]}
        kz = {r["k"]: r for r in k["zone"]}
        for tag, key in (("VeryLowK", "k1-4"), ("LowK", "k1-9"), ("HighK", "k10-21")):
            add3(
                f"kappaThr{tag}",
                kr[key]["kappa_binary"],
                f2,
                f"replicate kappa at 0.35, min k of the pair in {key}",
            )
            add(
                f"pctThrFlipDis{tag}",
                pct(kr[key]["disordered_flip"][0]),
                f"% flips in the disordered zone, {key}",
            )
            add(
                f"thrPostDis{tag}",
                f2(kz[key]["post_disordered_at_stated_pi"][0]),
                f"posterior of G >= 0.70 at \\thrPrior, {key}",
            )

    lines = [
        "% Generated by scripts/manuscript/threshold_numbers.py -- do not edit by hand.",
        f"% Input: {fdir.name} (release {'/'.join(Path(R['inputs']['release']).parts[-3:])},"
        f" nboot {R['settings']['nboot']}, seed {R['settings']['seed']}, primary estimator {est})",
    ]
    lines += [f"\\newcommand{{\\{k_}}}{{{v}}}  % {w}" for k_, v, w in m]
    args.out.write_text("\n".join(lines) + "\n")
    names = {k_ for k_, _, _ in m}
    assert len(names) == len(m), "duplicate macro names"
    print(f"wrote {len(m)} macros to {args.out}")
    flags = [v["flag"] for v in pc.values() if v["flag"]]
    print(
        "paper values inside their 95 % CI:", "all" if not flags else "; ".join(flags)
    )
    print(f"dropped since the 0.40 calibration ({len(DROPPED)}): {', '.join(DROPPED)}")


if __name__ == "__main__":
    main()
