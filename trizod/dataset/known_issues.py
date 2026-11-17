"""Changes against the previous build and the known issues of a release, per chain.

``trizod dataset finalize`` (:mod:`trizod.dataset.finalize`) writes the
deposit's ``known_issues.tsv`` (``chain_id``, ``issue``, ``evidence``) and fills
the CHANGELOG's change and known-issue sections from the functions here (release
audit 2026-11-17b, items 7 and C13/D18/F28/G39/B6/G43). Every number comes from
the release Parquet, the package bundles of this and the previous build, the
build's BMRB pickle cache, or a table finalize is given; none is typed:

* :func:`tier_moves` -- chains that entered or left a tier against the previous
  build, by the rule that moved them;
* :func:`label_changes` -- the 273 K re-score statistics and the chains whose
  sequence changed (the alphabet fix), from the two builds' unfiltered records;
* :func:`condition_issues` -- pH* / pD readings, temperatures read as Celsius
  or Fahrenheit although deposited as kelvin, pressures without a unit, read
  from the build's ``BmrbEntry`` pickles;
* :func:`snapshot_drift`, :func:`construct_caveats`, :func:`reviewed_issues` --
  per-chain tables from an audit (snapshot drift, construct and sample caveats)
  or reviewed by hand (``docs/dataset/known_issues_reviewed.tsv``), checked
  against the release;
* :func:`pin_trimmed`, :func:`few_scored_training` -- from the pin provenance
  and the Parquet.
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from trizod.dataset import release_info as ri
from trizod.dataset.tiers import TIER_ORDER

#: Columns of ``known_issues.tsv``.
KNOWN_ISSUES_COLUMNS = ("chain_id", "issue", "evidence")
#: The committed table of hand-reviewed known issues.
REVIEWED_TSV = Path("docs") / "dataset" / "known_issues_reviewed.tsv"
STRICT_IONIC_CAP_M = 0.5
#: |ΔG| below this is no change (float32 storage of the labels).
DG_TOL = 1e-6
#: Group sizes up to which the CHANGELOG names the chains of a cause inline.
NAME_UP_TO = 5
#: A pressure deposited in Pa below this value is a placeholder for one
#: atmosphere (release validation 2026-11-17c, G39: 86 chains deposit "1 Pa",
#: one "1.013 Pa"); it is stored as deposited, about 1e-5 bar.
PRESSURE_PLACEHOLDER_PA_MAX = 10.0
#: Where finalize ships the per-chain tables it was given (G36 e): every
#: known_issues.tsv row that cites one names its path in the deposit.
INPUTS_DIR = "known_issues_inputs"


def _fmt(n) -> str:
    return f"{n:,}"


def _ids(ids, limit=None) -> str:
    ids = sorted(ids, key=_id_key)
    if limit is not None and len(ids) > limit:
        return ", ".join(ids[:limit]) + f", … ({len(ids)} in all)"
    return ", ".join(ids)


def _id_key(cid: str):
    return tuple(int(p) if p.isdigit() else p for p in str(cid).split("_"))


# --------------------------------------------------------------------------- #
# build records
# --------------------------------------------------------------------------- #


def scores_root(path) -> Path:
    """The directory holding ``<tier>/scores.json``: ``path`` itself, or its
    ``scores/`` (a package bundle)."""
    path = Path(path)
    if (path / "scores" / "unfiltered" / "scores.json").is_file():
        return path / "scores"
    if (path / "unfiltered" / "scores.json").is_file():
        return path
    raise SystemExit(
        f"{path}: neither a package bundle (scores/<tier>/scores.json) nor a "
        "scored directory (<tier>/scores.json)"
    )


_ID = re.compile(r'"ID":\s*"([^"]*)"')


def _tier_ids(path: Path) -> set[str]:
    with open(path) as fh:
        return {m.group(1) for line in fh if (m := _ID.search(line))}


def _fasta_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with open(path) as fh:
        return {line[1:].split()[0] for line in fh if line.startswith(">")}


def load_build(path, labels: bool = False) -> dict:
    """``{"ids": {tier: chain IDs}}`` of a build (a package bundle or a scored
    directory); with ``labels``, also ``{"labels": {id: {"seq", "gscores",
    "tier"}}}``: each chain's released labels, read from the record of the tier
    that selects it (``LABEL_SOURCE_RULE``: its ``train_tier`` if it is a
    training representative of the bundle, else the strictest tier holding
    it), so they compare with a release Parquet's ``gscores``."""
    root = scores_root(path)
    ids = {tier: _tier_ids(root / tier / "scores.json") for tier in TIER_ORDER}
    out = {"ids": ids, "path": root}
    if not labels:
        return out
    train = root.parent / "train"
    reps = {t: _fasta_ids(train / t / f"train_{t}_best.fasta") for t in TIER_ORDER}
    select = {}
    for cid in ids["unfiltered"]:
        trained = [t for t in TIER_ORDER if cid in reps[t]]
        held = [t for t in TIER_ORDER if cid in ids[t]]
        select[cid] = trained[-1] if trained else held[-1]
    recs = {}
    for tier in TIER_ORDER:
        with open(root / tier / "scores.json") as fh:
            for line in fh:
                m = _ID.search(line)
                if m is None or select.get(m.group(1)) != tier:
                    continue
                rec = json.loads(line)
                recs[rec["ID"]] = {
                    "seq": rec.get("seq"),
                    "gscores": rec.get("gscores") or [],
                    "tier": tier,
                }
    out["labels"] = recs
    return out


# --------------------------------------------------------------------------- #
# tier moves
# --------------------------------------------------------------------------- #


def _move_causes(tier: str, row: dict | None, seq_changed: bool) -> str:
    if row is None:
        return "not in this release"
    causes = []
    if tier != "unfiltered" and row.get("high_pressure") is True:
        causes.append("pressure")
    if tier in ("moderate", "strict") and row.get("membrane_mimetic_sample") is True:
        causes.append("membrane")
    ionic = row.get("ionic_strength")
    if tier == "strict" and ionic is not None and ionic > STRICT_IONIC_CAP_M:
        causes.append("ionic")
    if row.get("temperature_placeholder") is True:
        causes.append("273 K" if tier == "strict" else "273 K re-score")
    if seq_changed:
        causes.append("sequence alphabet")
    return " + ".join(causes) or "other"


def tier_moves(prev: dict, cur: dict, rows_by_id: dict) -> dict:
    """Per tier: record counts before and after, and the chains lost and gained,
    grouped by the rule that explains the move (from the chain's row in this
    release: ``high_pressure``, ``membrane_mimetic_sample``, ``ionic_strength``
    above the strict cap, ``temperature_placeholder``, a changed sequence)."""
    out = {}
    for tier in TIER_ORDER:
        before, after = prev["ids"][tier], cur["ids"][tier]
        lost, gained = defaultdict(list), defaultdict(list)
        for cid in before - after:
            changed = _seq_changed(prev, rows_by_id, cid)
            lost[_move_causes(tier, rows_by_id.get(cid), changed)].append(cid)
        for cid in after - before:
            row = rows_by_id.get(cid)
            if row is not None and row.get("temperature_placeholder") is True:
                cause = "273 K re-score"
            elif _seq_changed(prev, rows_by_id, cid):
                cause = "sequence alphabet"
            else:
                cause = "other"
            gained[cause].append(cid)
        out[tier] = {
            "before": len(before),
            "after": len(after),
            "lost": {k: sorted(v, key=_id_key) for k, v in lost.items()},
            "gained": {k: sorted(v, key=_id_key) for k, v in gained.items()},
        }
    return out


def _seq_changed(prev, rows_by_id, cid) -> bool:
    a = prev.get("labels", {}).get(cid, {}).get("seq")
    b = (rows_by_id.get(cid) or {}).get("sequence")
    return a is not None and b is not None and a != b


def _causes_text(groups: dict) -> str:
    n = sum(len(v) for v in groups.values())
    if not n:
        return "0"
    parts = []
    for cause, ids in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        named = f" ({_ids(ids)})" if len(ids) <= NAME_UP_TO else ""
        parts.append(f"{cause} {_fmt(len(ids))}{named}")
    return f"{_fmt(n)}: " + ", ".join(parts)


def tier_moves_table(moves: dict) -> str:
    lines = [
        "| Tier | Records before → after | Lost, by rule | Gained, by cause |",
        "|---|---|---|---|",
    ]
    for tier in TIER_ORDER:
        m = moves[tier]
        lines.append(
            f"| {tier} | {_fmt(m['before'])} → {_fmt(m['after'])} "
            f"| {_causes_text(m['lost'])} | {_causes_text(m['gained'])} |"
        )
    return "\n".join(lines)


def tier_move_rows(moves: dict) -> list[tuple[str, str, str]]:
    per_chain = defaultdict(list)
    for tier in TIER_ORDER:
        for kind in ("lost", "gained"):
            verb = "left" if kind == "lost" else "entered"
            for cause, ids in moves[tier][kind].items():
                for cid in ids:
                    per_chain[cid].append(f"{verb} {tier} ({cause})")
    return [
        (cid, "tier_move", "; ".join(per_chain[cid]))
        for cid in sorted(per_chain, key=_id_key)
    ]


# --------------------------------------------------------------------------- #
# label changes: 273 K re-scores, sequence alphabet
# --------------------------------------------------------------------------- #


def _finite(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def compare_labels(old_g, new_g, cuts=(ri.DISORDER_CUT, ri.CONFIDENT_DISORDER_CUT)):
    """Per-chain label change over the common length: max |ΔG| and residues
    whose zone at each cut flips, over residues scored in both; mask flips."""
    n = min(len(old_g), len(new_g))
    both, max_dg, mask_flips = 0, 0.0, 0
    cross = dict.fromkeys(cuts, 0)
    changed_pos = []
    for i in range(n):
        a, b = old_g[i], new_g[i]
        fa, fb = _finite(a), _finite(b)
        if fa != fb:
            mask_flips += 1
            changed_pos.append(i + 1)
            continue
        if not fa:
            continue
        both += 1
        d = abs(a - b)
        if d > DG_TOL:
            changed_pos.append(i + 1)
        max_dg = max(max_dg, d)
        for c in cuts:
            cross[c] += (a >= c) != (b >= c)
    return {
        "scored_in_both": both,
        "max_abs_dg": max_dg,
        "mask_flips": mask_flips,
        "crossings": cross,
        "changed_positions": changed_pos,
        "changed": bool(changed_pos) or len(old_g) != len(new_g),
    }


def _quantile(values, q):
    xs = sorted(values)
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def label_changes(prev: dict, rows: list[dict]) -> dict:
    """The 273 K re-score statistics (chains with ``temperature_placeholder``
    released by both builds) and the chains whose sequence changed: this
    release's labels (the Parquet's ``gscores``) against the previous build's
    released labels (:func:`load_build` with ``labels``)."""
    old = prev["labels"]
    placeholder = [
        r["id"]
        for r in rows
        if r.get("temperature_placeholder") is True and r["id"] in old
    ]
    per_chain = {}
    for r in rows:
        if r["id"] in placeholder:
            per_chain[r["id"]] = compare_labels(
                old[r["id"]]["gscores"], list(r["gscores"])
            )
    maxes = [v["max_abs_dg"] for v in per_chain.values()]
    both = sum(v["scored_in_both"] for v in per_chain.values())
    cross = {
        c: sum(v["crossings"][c] for v in per_chain.values())
        for c in (ri.DISORDER_CUT, ri.CONFIDENT_DISORDER_CUT)
    }
    alphabet = {}
    for r in rows:
        before = old.get(r["id"])
        if before is None or before["seq"] is None or before["seq"] == r["sequence"]:
            continue
        alphabet[r["id"]] = {
            "old_length": len(before["seq"]),
            "new_length": len(r["sequence"]),
            **compare_labels(before["gscores"], list(r["gscores"])),
            "row": r,
        }
    return {
        "placeholder_chains": len(placeholder),
        "placeholder_changed": sum(v["changed"] for v in per_chain.values()),
        "max_abs_dg_median": _quantile(maxes, 0.5),
        "max_abs_dg_q90": _quantile(maxes, 0.9),
        "max_abs_dg_max": max(maxes) if maxes else None,
        "residues_scored_in_both": both,
        "crossings": cross,
        "per_chain": per_chain,
        "alphabet": alphabet,
    }


def rescore_text(lab: dict, prev_name: str) -> str:
    n = lab["placeholder_chains"]
    if not n:
        return f"No chain carries `temperature_placeholder` in both {prev_name} and this release."
    both = lab["residues_scored_in_both"]
    c1, c2 = ri.DISORDER_CUT, ri.CONFIDENT_DISORDER_CUT
    pct = (lambda k: f"{100 * k / both:.1f} %") if both else (lambda k: "–")
    return (
        f"Against {prev_name}, {_fmt(lab['placeholder_changed'])} of the "
        f"{_fmt(n)} such chains in both builds changed their labels; the "
        f"per-chain maximum |ΔG| has median {lab['max_abs_dg_median']:.3f}, 90th "
        f"percentile {lab['max_abs_dg_q90']:.3f} and maximum "
        f"{lab['max_abs_dg_max']:.3f}, and {pct(lab['crossings'][c1])} of their "
        f"{_fmt(both)} residues scored in both builds crossed G = "
        f"{ri.cut_label(c1)} ({pct(lab['crossings'][c2])} crossed "
        f"{ri.cut_label(c2)}). Released labels of both builds compared (each "
        "chain's selecting-tier record); `known_issues.tsv` lists every chain "
        "(`temperature_placeholder_rescore`)."
    )


def chezod_placeholder_text(lab: dict, rows: list[dict]) -> str:
    ids = [
        r["id"]
        for r in rows
        if r["split"] == "test_chezod117" and r.get("temperature_placeholder") is True
    ]
    retired = Counter(
        r.get("retired_reason") or "unknown"
        for r in rows
        if r["split"] == "test_retired" and r.get("temperature_placeholder") is True
    )
    c1, c2 = ri.DISORDER_CUT, ri.CONFIDENT_DISORDER_CUT
    parts = []
    for cid in sorted(ids, key=_id_key):
        v = lab["per_chain"].get(cid)
        if v is None:
            parts.append(f"{cid} (not in the previous build)")
            continue
        parts.append(
            f"{cid} (max |ΔG| {v['max_abs_dg']:.3f}; {v['crossings'][c1]} "
            f"residue(s) cross {ri.cut_label(c1)}, {v['crossings'][c2]} cross "
            f"{ri.cut_label(c2)})"
        )
    text = (
        f"{_fmt(len(ids))} CheZOD117 chains carry `temperature_placeholder` and "
        "were re-scored at 298 K" + (": " + "; ".join(parts) if parts else "") + "."
    )
    n_ret = sum(retired.values())
    if n_ret:
        text += (
            f" {_fmt(n_ret)} placeholder chains of the pin are `test_retired` ("
            + ", ".join(f"{_fmt(n)} {k}" for k, n in sorted(retired.items()))
            + ")."
        )
    return text


def rescore_rows(lab: dict) -> list[tuple[str, str, str]]:
    c1, c2 = ri.DISORDER_CUT, ri.CONFIDENT_DISORDER_CUT
    return [
        (
            cid,
            "temperature_placeholder_rescore",
            f"deposited 273 K read as not deposited, re-scored at 298 K: "
            f"max |ΔG| {v['max_abs_dg']:.3f}; {v['crossings'][c1]} residue(s) "
            f"cross {ri.cut_label(c1)}, {v['crossings'][c2]} cross "
            f"{ri.cut_label(c2)}; {v['mask_flips']} scored/unscored flips",
        )
        for cid, v in sorted(lab["per_chain"].items(), key=lambda kv: _id_key(kv[0]))
    ]


def _alphabet_evidence(v: dict) -> str:
    row = v.get("row") or {}
    where = row.get("split", "not released")
    if row.get("train_tier"):
        where += f", train_tier {row['train_tier']}"
    pos = v["changed_positions"]
    span = f"positions {pos[0]}–{pos[-1]}" if len(pos) > 1 else f"position {pos[0]}"
    labels = (
        f"labels changed at {len(pos)} residue(s) ({span})"
        if pos
        else ("labels unchanged")
    )
    return (
        f"sequence {v['old_length']} → {v['new_length']} residues; {labels}; "
        f"now {where}"
    )


def alphabet_text(lab: dict) -> str:
    if not lab["alphabet"]:
        return "No released chain changed its sequence."
    return (
        "Chains whose sequence changed: "
        + "; ".join(
            f"{cid} ({_alphabet_evidence(v)})"
            for cid, v in sorted(lab["alphabet"].items(), key=lambda kv: _id_key(kv[0]))
        )
        + "."
    )


def alphabet_rows(lab: dict) -> list[tuple[str, str, str]]:
    return [
        (cid, "sequence_alphabet_change", _alphabet_evidence(v))
        for cid, v in sorted(lab["alphabet"].items(), key=lambda kv: _id_key(kv[0]))
    ]


# --------------------------------------------------------------------------- #
# sample conditions, from the build's BmrbEntry pickles
# --------------------------------------------------------------------------- #


def _condition_of(entry, row):
    key = (str(row["st_id"]), str(row["entity_assem_id"]), str(row["entity_id"]))
    hit = entry.get_peptide_shifts().get(key)
    if hit is None:
        return None
    return entry.conditions.get(hit[1])


def classify_conditions(cond) -> list[tuple[str, str]]:
    """``(issue, evidence)`` for one chain's sample-condition list: a pH* or pD
    reading, a temperature read as Celsius or Fahrenheit although deposited as
    kelvin (or without a unit), a pressure without a (known) unit."""
    import numpy as np

    from trizod.bmrb.bmrb import (
        HPA_AS_BAR_RANGE,
        PD_TO_PH_STAR_OFFSET,
        PRESSURE_UNITS_TO_BAR,
        _key,
    )

    out = []
    scale = cond.get_pH_scale()
    if scale in ("pH*", "pD"):
        val, unit = cond.pH
        stored = cond.get_pH()
        rule = (
            f"converted to pD − {PD_TO_PH_STAR_OFFSET:g}"
            if scale == "pD"
            else "used as is"
        )
        out.append(
            (
                "ph_scale_" + ("pd" if scale == "pD" else "ph_star"),
                f"{scale} {val} deposited (unit {unit or 'none'}), {rule}: ph "
                f"{stored:g}",
            )
        )
    if not cond.is_temperature_placeholder():
        raw, unit = cond.temperature
        fixed = cond.get_temperature()
        unfixed = cond.get_temperature(fix_outliers=False)
        num = cond.convert_val(raw)
        same = (fixed == unfixed) or (
            isinstance(fixed, float)
            and isinstance(unfixed, float)
            and np.isnan(fixed)
            and np.isnan(unfixed)
        )
        if not same and not np.isnan(num):
            kind = "celsius" if num < 50 else "fahrenheit"
            out.append(
                (
                    f"temperature_{kind}_as_kelvin",
                    f"temperature {raw} deposited with unit {unit or 'none'} "
                    f"(below 100 K, impossible), read as °{kind[0].upper()}: "
                    f"{fixed:.2f} K",
                )
            )
    pval, punit = cond.pressure
    text = str(pval if pval is not None else "").strip()
    if text and text not in (".", "?") and _key(punit) not in PRESSURE_UNITS_TO_BAR:
        what = "without a unit" if not _key(punit).strip(".?") else f"in unit {punit}"
        out.append(
            (
                "pressure_unit_missing",
                f"pressure {text} deposited {what}: pressure_bar and "
                "high_pressure are null (passes as ambient)",
            )
        )
    elif text and text not in (".", "?"):
        num = cond.convert_val(text)
        bar = cond.get_pressure_bar()
        if _key(punit) == "pa" and 0 < num < PRESSURE_PLACEHOLDER_PA_MAX:
            out.append(
                (
                    "pressure_placeholder_pa",
                    f"pressure {text} Pa deposited (almost certainly one "
                    f"atmosphere), stored as deposited: pressure_bar {bar:.3g}; far "
                    "below the 400 bar cap, so no tier is affected",
                )
            )
        elif _key(punit) == "bar" and HPA_AS_BAR_RANGE[0] <= num <= HPA_AS_BAR_RANGE[1]:
            out.append(
                (
                    "pressure_hpa_as_bar",
                    f"pressure {text} deposited in bar, read as hPa (one "
                    f"atmosphere): pressure_bar {bar:.4g}",
                )
            )
    return out


def condition_issues(rows: list[dict], pkl_dir, bmrb_dir=None) -> list[tuple]:
    """``(chain_id, issue, evidence)`` of :func:`classify_conditions` for every
    release row, from the build's pickle cache (``bmrb_dir`` parses a missing
    pickle). A chain whose entry cannot be read is reported as such."""
    from trizod.shifts.extract import load_entry

    by_entry = defaultdict(list)
    for r in rows:
        by_entry[str(r["entry_id"])].append(r)
    out = []
    for eid in sorted(by_entry, key=_id_key):
        try:
            entry = load_entry(eid, Path(pkl_dir), bmrb_dir)
        except Exception as exc:  # noqa: BLE001 - recorded, never fatal
            for r in by_entry[eid]:
                out.append((r["id"], "conditions_unread", f"{type(exc).__name__}"))
            continue
        for r in by_entry[eid]:
            cond = _condition_of(entry, r)
            if cond is None:
                out.append((r["id"], "conditions_unread", "no such shift table"))
                continue
            for issue, evidence in classify_conditions(cond):
                out.append((r["id"], issue, evidence))
    return out


def condition_text(issues: list[tuple], cur: dict, rows_by_id: dict) -> dict:
    """The CHANGELOG sentences on pH* / pD, Celsius-as-kelvin and pressure
    units, with tier membership read from this build's bundle (``cur``)."""
    from trizod.bmrb.bmrb import PD_TO_PH_STAR_OFFSET

    by_issue = defaultdict(set)
    for cid, issue, _ in issues:
        by_issue[issue].add(cid)
    strict = cur["ids"]["strict"]
    tolerant = cur["ids"]["tolerant"]
    ph = by_issue["ph_scale_ph_star"] | by_issue["ph_scale_pd"]
    ph_strict = sorted(ph & strict, key=_id_key)
    pd_all = by_issue["ph_scale_pd"]
    celsius = by_issue["temperature_celsius_as_kelvin"]
    below_100 = celsius | by_issue["temperature_fahrenheit_as_kelvin"]
    entries_below_100 = {
        str(rows_by_id[c]["entry_id"]) for c in below_100 if c in rows_by_id
    }
    pressure = by_issue["pressure_unit_missing"]
    placeholder_pa = by_issue["pressure_placeholder_pa"]
    hpa = by_issue["pressure_hpa_as_bar"]
    unread = by_issue["conditions_unread"]
    return {
        "ph": (
            f"{_fmt(len(ph))} released chains were measured in pH* or pD "
            f"({_fmt(len(ph_strict))} of them strict: {_ids(ph_strict)}). A pH* "
            f"reading is used as is; a pD is converted to "
            f"pD − {PD_TO_PH_STAR_OFFSET:g} ({_fmt(len(pd_all))} chains: "
            f"{_ids(pd_all)})."
        ),
        "temperature": (
            f"{_fmt(len(celsius & tolerant))} tolerant chains deposited a "
            "Celsius temperature as kelvin and were read as °C "
            f"({_ids(celsius & tolerant)}); {_fmt(len(entries_below_100))} "
            "released entries deposited a kelvin value below 100 K "
            f"({_ids(entries_below_100)})."
        ),
        "pressure": (
            f"{_fmt(len(pressure))} released chains deposited a pressure without "
            "a unit or in an unknown one; it reads as not recorded"
            + (f" ({_ids(pressure, 12)})" if pressure else "")
            + ". "
            f"{_fmt(len(placeholder_pa))} released chains deposited a pressure of "
            "about 1 Pa (1 or 1.013 Pa, almost certainly one atmosphere); it is "
            "stored as deposited, about 1e-5 bar (`pressure_placeholder_pa`"
            + (f": {_ids(placeholder_pa, 12)}" if placeholder_pa else "")
            + f"). {_fmt(len(hpa))} deposited 1013 bar, read as hPa, i.e. one "
            "atmosphere (`pressure_hpa_as_bar`"
            + (f": {_ids(hpa)}" if hpa else "")
            + "). Every one of these values is far below the 400 bar cap, so no "
            "tier is affected."
        ),
        "unread": len(unread),
    }


# --------------------------------------------------------------------------- #
# tables finalize is given, and the reviewed table
# --------------------------------------------------------------------------- #


def _read_tsv(path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def changed_entries(old_dir, new_dir) -> dict[str, set[str]]:
    """Entries of two BMRB snapshots (``bmr<ID>/bmr<ID>_3.str``): ``changed``
    (in both, file bytes differ), ``added`` and ``withdrawn``."""
    import filecmp

    def entries(root):
        root = Path(root)
        return {
            d.name[3:]
            for d in root.iterdir()
            if re.fullmatch(r"bmr\d+", d.name) and (d / f"{d.name}_3.str").is_file()
        }

    old, new = entries(old_dir), entries(new_dir)
    changed = {
        e
        for e in old & new
        if not filecmp.cmp(
            Path(old_dir) / f"bmr{e}" / f"bmr{e}_3.str",
            Path(new_dir) / f"bmr{e}" / f"bmr{e}_3.str",
            shallow=False,
        )
    }
    return {"changed": changed, "added": new - old, "withdrawn": old - new}


def _drift_fields(per_tier: dict) -> set[str]:
    out = set()
    for v in per_tier.values():
        if isinstance(v, dict):
            out.update(v.get("fields") or ())
    return out


def snapshot_drift(path, rows_by_id: dict, changed: dict | None = None) -> dict:
    """Rows of a snapshot-drift table (an audit's ``id``, ``released``,
    ``label_or_tier_change``, ``per_tier`` JSON; one row per chain whose
    release record differs between the two snapshots): the chains whose labels
    or tiers changed. ``changed`` (:func:`changed_entries`) adds every chain
    of the changed entries, so the text says what the counts are of."""
    recs = _read_tsv(path)
    out_rows, changed_ids, test_hit = [], [], []
    meta_only, meta_fields = [], set()
    for r in recs:
        if str(r.get("label_or_tier_change")).lower() != "true":
            meta_only.append(r["id"])
            with contextlib.suppress(ValueError):
                meta_fields |= _drift_fields(json.loads(r.get("per_tier") or "{}"))
            continue
        cid = r["id"]
        changed_ids.append(cid)
        row = rows_by_id.get(cid)
        if row is not None and str(row.get("split")).startswith("test"):
            test_hit.append(cid)
        try:
            per_tier = json.loads(r.get("per_tier") or "{}")
        except ValueError:
            per_tier = {}
        bits = []
        for tier, v in per_tier.items():
            if isinstance(v, str):
                bits.append(f"{tier}: {v.replace('_', ' ')}")
            elif isinstance(v, dict) and "gscores" in v:
                g = v["gscores"]
                bits.append(
                    f"{tier}: max |ΔG| {g.get('max_abs_dG')}, "
                    f"{g.get('mask_flips')} mask flips"
                )
        where = (
            row.get("split", "released") if row is not None else "no longer released"
        )
        out_rows.append(
            (cid, "snapshot_drift", f"{'; '.join(bits) or 'changed'}; now {where}")
        )
    out = {
        "n_rows": len(recs),
        "changed": sorted(changed_ids, key=_id_key),
        "metadata_only": sorted(meta_only, key=_id_key),
        "metadata_fields": sorted(meta_fields),
        "test_affected": sorted(test_hit, key=_id_key),
        "rows": out_rows,
        "entries": None,
    }
    if changed is not None:
        ents = set(changed["changed"])
        released = {
            cid: row
            for cid, row in rows_by_id.items()
            if str((row or {}).get("entry_id")) in ents
        }
        gone = sorted({r["id"] for r in recs if r["id"] not in rows_by_id}, key=_id_key)
        out["entries"] = {
            "changed": len(ents),
            "added": len(changed["added"]),
            "withdrawn": len(changed["withdrawn"]),
            "released_chains": len(released),
            "released_by_split": dict(
                sorted(Counter(r.get("split") for r in released.values()).items())
            ),
            "gone": gone,
            "test_chains": sorted(
                (
                    c
                    for c, r in released.items()
                    if str(r.get("split")).startswith("test")
                ),
                key=_id_key,
            ),
        }
    return out


def snapshot_drift_text(drift: dict | None) -> str:
    if drift is None:
        return (
            "Per-chain snapshot drift was not assessed for this deposit "
            "(finalize ran without --snapshot-drift)."
        )
    fields = ", ".join(drift["metadata_fields"])
    differ = (
        f"{_fmt(len(drift['metadata_only']))} only in metadata"
        + (f" ({fields})" if fields else "")
        + f" and {_fmt(len(drift['changed']))} in labels or tiers"
        + (f" ({_ids(drift['changed'])})" if drift["changed"] else "")
    )
    e = drift.get("entries")
    if e is None:
        return (
            f"Of {_fmt(drift['n_rows'])} chains whose record differs between the "
            f"two snapshots, {differ}; "
            + (
                f"test chains with changed labels: {_ids(drift['test_affected'])}."
                if drift["test_affected"]
                else "no test chain changed its labels."
            )
            + " The chains of the changed entries were not counted (finalize ran "
            "without --previous-bmrb-dir)."
        )
    n_chains = e["released_chains"] + len(e["gone"])
    splits = ", ".join(f"{_fmt(n)} {s}" for s, n in e["released_by_split"].items())
    tests = e["test_chains"]
    labelled = set(drift["test_affected"])
    test_text = (
        (
            f"The {_fmt(len(tests))} test chains of changed entries ({_ids(tests)}) "
            + (
                "have identical labels."
                if not labelled
                else f"include {_ids(labelled)}, whose labels or tiers changed."
            )
        )
        if tests
        else "No test chain belongs to a changed entry."
    )
    return (
        f"Between the two snapshots {_fmt(e['changed'])} entries changed in place "
        f"({_fmt(e['added'])} added, {_fmt(e['withdrawn'])} withdrawn). The changed "
        f"entries hold {_fmt(n_chains)} chains: {_fmt(e['released_chains'])} "
        f"released now ({splits})"
        + (f", and {_ids(e['gone'])}, no longer released" if e["gone"] else "")
        + f". {_fmt(drift['n_rows'])} of them have a record that differs: "
        f"{differ}. {test_text}"
    )


#: Construct caveat classes that concern single residues: their share is
#: reported at residue level (affected labelled residues over the labelled
#: residues assessed), never as the scored residues of the chains listed
#: (release validation 2026-11-17c, G43). The other classes concern whole
#: chains (a deuterated or paramagnetic sample).
RESIDUE_LEVEL_CLASSES = (
    "outside_uniprot_span_sifts",
    "nonstandard_letter_in_sequence",
    "nonstandard_residue",
)
CONSTRUCT_CLASS_DESCRIPTIONS = {
    "deut_uniform_2H": "uniformly 2H-labelled protein (isotope shifts not corrected)",
    "heme_spin_state_dependent": "heme protein whose shifts depend on the heme "
    "spin state",
    "nonstandard_letter_in_sequence": "the sequence holds X, U or Z",
    "nonstandard_residue": "the deposited polymer holds a non-standard residue",
    "outside_uniprot_span_sifts": "labelled residues outside the SIFTS UniProt "
    "span (tags, linkers, fusion partners)",
    "paramagnetic_definite": "paramagnetic sample",
}
CONSTRUCT_INPUT = f"{INPUTS_DIR}/construct_caveats.tsv"


def _nonstandard_positions(row) -> tuple[list[int], int]:
    std = set("ACDEFGHIKLMNPQRSTVWY")
    pos = [i + 1 for i, ch in enumerate(row["sequence"]) if ch not in std]
    mask = row.get("mask") or []
    scored = sum(1 for p in pos if p - 1 < len(mask) and mask[p - 1])
    return pos, scored


def construct_caveats(
    path, rows: list[dict], residue_counts: dict[str, dict] | None = None
) -> dict:
    """Counts of an audit's construct / sample caveat table (``class``,
    ``id``): per class, in all training representatives, the tolerant
    training set and the test set, the chains listed and -- for a whole-chain
    class -- the share of scored residues they hold, or -- for a residue-level
    class (``RESIDUE_LEVEL_CLASSES``) -- the affected labelled residues over
    the labelled residues assessed, from ``residue_counts`` (``{class: {chain:
    (affected, assessed)}}``; :mod:`trizod.dataset.construct_span` measures
    ``outside_uniprot_span_sifts``, the X/U/Z positions come from the
    sequence). Every row's evidence names the class, what was measured for the
    chain and the shipped input table."""
    recs = _read_tsv(path)
    rows_by_id = {r["id"]: r for r in rows}
    residue_counts = dict(residue_counts or {})
    if "nonstandard_letter_in_sequence" not in residue_counts:
        residue_counts["nonstandard_letter_in_sequence"] = {
            r["id"]: (_nonstandard_positions(r)[1], sum(1 for m in r["mask"] if m))
            for r in rows
            if r["split"] in ("train", "test_trizod")
        }
    groups = {
        "training": lambda r: r["split"] == "train",
        "tolerant training": lambda r: (
            r["split"] == "train" and r.get("train_tier") in TIER_ORDER[1:]
        ),
        "test": lambda r: r["split"] == "test_trizod",
    }

    def scored(r):
        return sum(1 for m in r["mask"] if m)

    denom = {g: sum(scored(r) for r in rows if f(r)) for g, f in groups.items()}
    by_class = defaultdict(set)
    unknown = set()
    for rec in recs:
        cid = rec.get("id")
        if cid in rows_by_id:
            by_class[rec.get("class", "unknown")].add(cid)
        else:
            unknown.add(cid)
    summary = {}
    for cls, ids in sorted(by_class.items()):
        d = {}
        counts = residue_counts.get(cls) if cls in RESIDUE_LEVEL_CLASSES else None
        for g, f in groups.items():
            sel = [rows_by_id[i] for i in ids if f(rows_by_id[i])]
            res = sum(scored(r) for r in sel)
            d[g] = {
                "chains": len(sel),
                "share": res / denom[g] if denom[g] else None,
            }
            if counts is not None:
                assessed = [r for r in rows if f(r) and r["id"] in counts]
                d[g]["assessed_chains"] = len(assessed)
                d[g]["affected"] = sum(counts[r["id"]][0] for r in assessed)
                d[g]["assessed"] = sum(counts[r["id"]][1] for r in assessed)
                d[g]["affected_chains"] = sum(
                    1 for r in assessed if counts[r["id"]][0] > 0
                )
        summary[cls] = {
            "level": "residue" if cls in RESIDUE_LEVEL_CLASSES else "chain",
            "measured": counts is not None,
            **d,
        }

    def evidence(cls, cid):
        desc = CONSTRUCT_CLASS_DESCRIPTIONS.get(cls, cls.replace("_", " "))
        bits = [desc]
        row = rows_by_id[cid]
        if cls == "nonstandard_letter_in_sequence":
            pos, sc = _nonstandard_positions(row)
            bits.append(
                f"X/U/Z at position(s) {', '.join(map(str, pos[:12]))}"
                + (f", … ({len(pos)} in all)" if len(pos) > 12 else "")
                + f"; {sc} of them scored"
            )
        counts = residue_counts.get(cls) if cls in RESIDUE_LEVEL_CLASSES else None
        if counts is not None and cls != "nonstandard_letter_in_sequence":
            if cid in counts:
                bits.append(
                    f"{counts[cid][0]} of {counts[cid][1]} labelled residues "
                    "outside the span"
                )
            else:
                bits.append("not assessed at residue level")
        bits.append(f"listed by {CONSTRUCT_INPUT}")
        return "; ".join(bits)

    out_rows = [
        (cid, f"construct_{cls}", evidence(cls, cid))
        for cls, ids in sorted(by_class.items())
        for cid in sorted(ids, key=_id_key)
    ]
    return {"summary": summary, "unknown": sorted(unknown), "rows": out_rows}


def nonstandard_counts(rows: list[dict]) -> dict:
    """Training and test chains whose sequence holds X (or U, Z), from the
    Parquet alone."""
    std = set("ACDEFGHIKLMNPQRSTVWY")
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test_trizod"]

    def odd(r):
        return set(r["sequence"]) - std

    odd_train = [r for r in train if odd(r)]
    return {
        "train_chains": len(odd_train),
        "train_positions": sum(len(_nonstandard_positions(r)[0]) for r in odd_train),
        "train_scored_positions": sum(_nonstandard_positions(r)[1] for r in odd_train),
        "train_chain_residues": sum(len(r["sequence"]) for r in odd_train),
        "train_residues": sum(len(r["sequence"]) for r in train),
        "test_chains": sum(1 for r in test if odd(r)),
    }


def _pct(a, b) -> str:
    if not b:
        return "–"
    share = 100 * a / b
    return f"{share:.2f} %" if 0 < share < 0.1 else f"{share:.1f} %"


def construct_text(caveats: dict | None, nonstd: dict) -> str:
    x = (
        f"{_fmt(nonstd['train_chains'])} training representatives hold a "
        f"non-standard letter (X, U or Z): {_fmt(nonstd['train_positions'])} X/U/Z "
        f"positions ({_fmt(nonstd.get('train_scored_positions', 0))} scored) in the "
        f"{_fmt(nonstd.get('train_chain_residues', 0))} residues of these chains; "
        f"{_fmt(nonstd['test_chains'])} test chains"
    )
    if caveats is None:
        return (
            f"{x}. Deuteration, residues outside the UniProt-aligned span and "
            "paramagnetic samples were not counted for this deposit (finalize "
            "ran without --construct-caveats). No flag column marks any of "
            "these yet; flag columns are deferred to a later release."
        )
    parts = []
    for cls, d in caveats["summary"].items():
        bits = []
        groups = ("training", "tolerant training", "test")
        if d["level"] == "chain":
            for g in groups:
                v = d[g]
                share = f"{100 * v['share']:.1f} %" if v["share"] is not None else "–"
                bits.append(
                    f"{g} {_fmt(v['chains'])} chains ({share} of scored residues)"
                )
            parts.append(f"`{cls}` (whole chains): " + ", ".join(bits))
        elif d["measured"]:
            for g in groups:
                v = d[g]
                bits.append(
                    f"{g} {_fmt(v['affected'])} of {_fmt(v['assessed'])} labelled "
                    f"residues ({_pct(v['affected'], v['assessed'])}) of the "
                    f"{_fmt(v['assessed_chains'])} assessed chains, in "
                    f"{_fmt(v['affected_chains'])} chains"
                )
            parts.append(f"`{cls}` (residue level): " + ", ".join(bits))
        else:
            for g in groups:
                bits.append(f"{g} {_fmt(d[g]['chains'])} chains")
            parts.append(
                f"`{cls}` (chains listed; residue-level share not measured): "
                + ", ".join(bits)
            )
    return (
        f"{x}. Per caveat class (`known_issues.tsv`, issue `construct_<class>`; "
        "a whole-chain class gives the share of scored residues in the chains "
        "listed, a residue-level class the affected labelled residues): "
        + "; ".join(parts)
        + ". No flag column marks these yet; flag columns are deferred to a "
        "later release."
    )


#: ``physical_state`` values of a CheZOD117 chain deposited as not native.
UNFOLDED_STATES = ("denatured", "unfolded", "partially unfolded")


def chezod117_conditions(rows: list[dict]) -> dict:
    """CheZOD117 chains whose sample conditions a TriZOD tier would reject:
    CheZOD117 is the external benchmark and is not filtered on conditions."""
    groups = {"cosolvent": [], "bound": [], "placeholder": [], "unfolded": []}
    out = []
    for r in sorted(rows, key=lambda r: _id_key(r["id"])):
        if r["split"] != "test_chezod117":
            continue
        bits = []
        if r.get("cosolvent_evidence") is True:
            groups["cosolvent"].append(r["id"])
            bits.append("names a perturbing cosolvent (cosolvent_evidence)")
        if r.get("is_bound") is True:
            groups["bound"].append(r["id"])
            bits.append("bound complex (is_bound)")
        if r.get("temperature_placeholder") is True:
            groups["placeholder"].append(r["id"])
            bits.append("273 K placeholder, re-scored at 298 K")
        if (r.get("physical_state") or "") in UNFOLDED_STATES:
            groups["unfolded"].append(r["id"])
            bits.append(f"physical_state {r['physical_state']}")
        if bits:
            out.append(
                (
                    r["id"],
                    "chezod117_condition",
                    "; ".join(bits) + " (CheZOD117 is not filtered on conditions)",
                )
            )
    g = groups
    text = (
        "CheZOD117 is not filtered on sample conditions: "
        f"{_fmt(len(g['cosolvent']))} chains name a perturbing cosolvent "
        f"({_ids(g['cosolvent'])}), {_fmt(len(g['bound']))} are bound complexes "
        f"({_ids(g['bound'])}), {_fmt(len(g['placeholder']))} carry the 273 K "
        f"placeholder ({_ids(g['placeholder'])}) and {_fmt(len(g['unfolded']))} "
        "are deposited as denatured or (partially) unfolded "
        f"({_ids(g['unfolded'])})."
    )
    return {"rows": out, "text": text, "groups": groups}


def ionic_drivers(path, rows_by_id: dict) -> dict:
    """Chains whose deposited ionic strength puts them out of strict while
    their own sample components imply <= the strict cap, from the audit's
    driver log (``chain``, ``flag`` starting ``INCONSISTENT``,
    ``I_deposited_strict_M``, ``I_implied_M``, ``implied_parts``)."""
    out = []
    for rec in _read_tsv(path):
        if not str(rec.get("flag", "")).startswith("INCONSISTENT"):
            continue
        cid = rec["chain"]
        row = rows_by_id.get(cid)
        where = row["split"] if row is not None else "not released"
        out.append(
            (
                cid,
                "ionic_strength_contradicts_components",
                f"ionic strength {rec.get('I_deposited_strict_M')} M deposited, "
                f"its sample components imply {rec.get('I_implied_M')} M "
                f"({rec.get('implied_parts') or 'components'}); used as deposited, "
                f"so the chain is out of strict (split {where})",
            )
        )
    ids = [cid for cid, _, _ in out]
    entries = {_entry_of(c) for c in ids}
    text = (
        f"Ionic strengths are used as deposited: {_fmt(len(ids))} chains "
        f"({_fmt(len(entries))} entries) are out of strict on a deposited value "
        f"above {STRICT_IONIC_CAP_M:g} M although their own sample components imply "
        f"at most that ({_ids(ids)})."
    )
    return {"rows": out, "text": text}


def _entry_of(cid: str) -> str:
    return str(cid).split("_", 1)[0]


def reviewed_issues(path, rows_by_id: dict) -> list[tuple[str, str, str]]:
    """The hand-reviewed known issues, each chain checked to be in the release
    (a stale entry is an error, not a silent row)."""
    out, missing = [], []
    for rec in _read_tsv(path):
        cid = rec["chain_id"]
        row = rows_by_id.get(cid)
        if row is None:
            missing.append(cid)
            continue
        out.append((cid, rec["issue"], f"{rec['evidence']} (split {row['split']})"))
    if missing:
        raise SystemExit(
            f"{path}: {missing} are not in the release Parquet; update the table"
        )
    return out


def reviewed_text(rows: list[tuple]) -> str:
    if not rows:
        return "No reviewed per-chain issue."
    return "; ".join(f"{cid}: {evidence}" for cid, _, evidence in rows) + "."


def pin_trimmed(provenance: dict | None, rows_by_id: dict) -> dict:
    """The chains dropped when the pin was trimmed, with their status now."""
    if not provenance:
        return {"n": 0, "rows": [], "text": "No pin provenance."}
    dropped = provenance.get("dropped") or []
    frm = provenance.get("trimmed_from") or {}
    out = []
    for cid in dropped:
        row = rows_by_id.get(cid)
        if row is None:
            where = "not released"
        else:
            where = row["split"]
            if row.get("train_tier"):
                where += f", train_tier {row['train_tier']}"
            elif row.get("label_tier_source"):
                where += f", record from {row['label_tier_source']}"
        out.append(
            (
                cid,
                "pin_trimmed",
                f"dropped when the {frm.get('count', '?')}-chain pin of "
                f"{frm.get('written', '?')} was trimmed to "
                f"{provenance.get('count', '?')} ({provenance.get('written', '?')}); "
                f"now {where}",
            )
        )
    statuses = Counter(e.split("now ", 1)[1].split(",")[0] for _, _, e in out)
    text = (
        f"{_fmt(len(dropped))} chains were trimmed from the "
        f"{frm.get('count', '?')}-chain pin ({_ids(dropped)}); they are now "
        + ", ".join(f"{_fmt(n)} {s}" for s, n in sorted(statuses.items()))
        + " (`known_issues.tsv`, `pin_trimmed`; reasons in "
        "`test/TriZOD_test_set.provenance.json`)."
    )
    return {"n": len(dropped), "rows": out, "text": text}


def few_scored_training(rows: list[dict]) -> list[tuple[str, str, str]]:
    return [
        (
            r["id"],
            "few_scored_residues",
            f"training representative ({r.get('train_tier')}) with "
            f"{sum(1 for m in r['mask'] if m)} scored residues (< "
            f"{ri.K_MIN_RELIABLE})",
        )
        for r in sorted(rows, key=lambda r: _id_key(r["id"]))
        if r["split"] == "train" and sum(1 for m in r["mask"] if m) < ri.K_MIN_RELIABLE
    ]


def write_tsv(rows, path) -> Path:
    """``known_issues.tsv``: one row per (chain, issue), sorted, tab-separated
    (tabs and newlines in the evidence replaced by spaces)."""
    path = Path(path)
    seen = sorted(set(rows), key=lambda r: (_id_key(r[0]), r[1], r[2]))
    with path.open("w", newline="") as fh:
        fh.write("\t".join(KNOWN_ISSUES_COLUMNS) + "\n")
        for row in seen:
            fh.write("\t".join(" ".join(str(c).split()) for c in row) + "\n")
    return path
