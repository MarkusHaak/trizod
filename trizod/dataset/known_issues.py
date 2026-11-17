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

    from trizod.bmrb.bmrb import PD_TO_PH_STAR_OFFSET, PRESSURE_UNITS_TO_BAR, _key

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
            + "."
        ),
        "unread": len(unread),
    }


# --------------------------------------------------------------------------- #
# tables finalize is given, and the reviewed table
# --------------------------------------------------------------------------- #


def _read_tsv(path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def snapshot_drift(path, rows_by_id: dict) -> dict:
    """Rows of a snapshot-drift table (an audit's ``id``, ``released``,
    ``label_or_tier_change``, ``per_tier`` JSON): the chains whose labels or
    tiers changed between the two snapshots."""
    recs = _read_tsv(path)
    out_rows, changed, test_hit = [], [], []
    for r in recs:
        if str(r.get("label_or_tier_change")).lower() != "true":
            continue
        cid = r["id"]
        changed.append(cid)
        row = rows_by_id.get(cid)
        if row is not None and row["split"].startswith("test"):
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
        where = row["split"] if row is not None else "no longer released"
        out_rows.append(
            (cid, "snapshot_drift", f"{'; '.join(bits) or 'changed'}; now {where}")
        )
    return {
        "n_rows": len(recs),
        "changed": sorted(changed, key=_id_key),
        "test_affected": sorted(test_hit, key=_id_key),
        "rows": out_rows,
    }


def snapshot_drift_text(drift: dict | None) -> str:
    if drift is None:
        return (
            "Per-chain snapshot drift was not assessed for this deposit "
            "(finalize ran without --snapshot-drift)."
        )
    return (
        f"Of {_fmt(drift['n_rows'])} released or formerly released chains of "
        "entries whose file changed between the two snapshots, "
        f"{_fmt(len(drift['changed']))} changed labels or tiers "
        f"({_ids(drift['changed'])}); "
        + (
            f"test chains affected: {_ids(drift['test_affected'])}."
            if drift["test_affected"]
            else "no test chain is affected."
        )
    )


def construct_caveats(path, rows: list[dict]) -> dict:
    """Counts of an audit's construct / sample caveat table (``class``,
    ``id``): chains and the share of scored residues per class, in all
    training representatives, the tolerant training set and the test set."""
    recs = _read_tsv(path)
    rows_by_id = {r["id"]: r for r in rows}
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
        for g, f in groups.items():
            sel = [rows_by_id[i] for i in ids if f(rows_by_id[i])]
            res = sum(scored(r) for r in sel)
            d[g] = {
                "chains": len(sel),
                "share": res / denom[g] if denom[g] else None,
            }
        summary[cls] = d
    out_rows = [
        (cid, f"construct_{cls}", f"listed by {Path(path).name}")
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

    def pos(rs):
        return sum(sum(1 for ch in r["sequence"] if ch not in std) for r in rs)

    return {
        "train_chains": sum(1 for r in train if odd(r)),
        "train_positions": pos(train),
        "train_residues": sum(len(r["sequence"]) for r in train),
        "test_chains": sum(1 for r in test if odd(r)),
    }


def construct_text(caveats: dict | None, nonstd: dict) -> str:
    x = (
        f"{_fmt(nonstd['train_chains'])} training representatives hold a "
        f"non-standard letter (X, U or Z; {_fmt(nonstd['train_positions'])} of "
        f"their {_fmt(nonstd['train_residues'])} residues), "
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
        for g in ("training", "tolerant training", "test"):
            v = d[g]
            share = f"{100 * v['share']:.1f} %" if v["share"] is not None else "–"
            bits.append(f"{g} {_fmt(v['chains'])} chains ({share} of scored residues)")
        parts.append(f"`{cls}`: " + ", ".join(bits))
    return (
        f"{x}. Per caveat class (`known_issues.tsv`, issue `construct_<class>`): "
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
