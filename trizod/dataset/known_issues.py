"""The known issues of a release, per chain.

``trizod dataset finalize`` (:mod:`trizod.dataset.finalize`) writes the
deposit's ``known_issues.tsv`` (``chain_id``, ``issue``, ``evidence``) and fills
the CHANGELOG's known-issue section from the functions here. Every row states a
property of a released chain, never a change against another build. Every
number comes from the release Parquet, the package bundle, the build's BMRB
pickle cache, or a table finalize is given; none is typed:

* :func:`temperature_placeholder_rows` -- chains whose deposited 273 K was read
  as not deposited (``temperature_placeholder``) and scored at the 298 K
  default;
* :func:`condition_issues` -- pH* / pD readings, temperatures read as Celsius
  or Fahrenheit although deposited as kelvin, pressures without a unit, read
  from the build's ``BmrbEntry`` pickles;
* :func:`construct_caveats`, :func:`ionic_drivers`, :func:`reviewed_issues` --
  per-chain tables finalize is given (construct and sample caveats, ionic
  strengths that contradict the sample's components) or reviewed by hand
  (``docs/dataset/known_issues_reviewed.tsv``), checked against the release;
* :func:`chezod117_conditions`, :func:`few_scored_training` -- from the
  Parquet.
"""

from __future__ import annotations

import csv
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
#: A pressure deposited in Pa below this value is a placeholder for one
#: atmosphere (86 chains deposit "1 Pa", one "1.013 Pa"); it is stored as
#: deposited, about 1e-5 bar.
PRESSURE_PLACEHOLDER_PA_MAX = 10.0
#: Where finalize ships the per-chain tables it was given: every
#: known_issues.tsv row that cites one names its path in the deposit.
INPUTS_DIR = "known_issues_inputs"
#: Evidence of a ``temperature_placeholder_rescore`` row.
TEMPERATURE_PLACEHOLDER_EVIDENCE = (
    "deposited 273 K read as not deposited; scored at the 298 K default"
)


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
# tier membership of the build
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


def load_build(path) -> dict:
    """``{"ids": {tier: chain IDs}, "path": scores root}`` of a build (a
    package bundle or a scored directory)."""
    root = scores_root(path)
    ids = {tier: _tier_ids(root / tier / "scores.json") for tier in TIER_ORDER}
    return {"ids": ids, "path": root}


# --------------------------------------------------------------------------- #
# the 273 K placeholder
# --------------------------------------------------------------------------- #


def temperature_placeholder_rows(rows: list[dict]) -> list[tuple[str, str, str]]:
    """One row per released chain whose deposited 273 K was read as not
    deposited (``temperature_placeholder``): scored at the 298 K default."""
    return [
        (r["id"], "temperature_placeholder_rescore", TEMPERATURE_PLACEHOLDER_EVIDENCE)
        for r in sorted(rows, key=lambda r: _id_key(r["id"]))
        if r.get("temperature_placeholder") is True
    ]


def chezod_placeholder_text(rows: list[dict]) -> str:
    """The CheZOD117 and retired test chains that carry the 273 K
    placeholder."""
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
    text = (
        f"{_fmt(len(ids))} CheZOD117 chains carry `temperature_placeholder` and "
        "were re-scored at 298 K" + (f" ({_ids(ids)})" if ids else "") + "."
    )
    n_ret = sum(retired.values())
    if n_ret:
        text += (
            f" {_fmt(n_ret)} retired test chains carry it ("
            + ", ".join(f"{_fmt(n)} {k}" for k, n in sorted(retired.items()))
            + ")."
        )
    return text


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
#: Residue-level classes about labelled residues: a listed chain measured with
#: none gets no known-issue row (the audit table counted residues outside the
#: span, labelled or not; release validation 2026-11-17d, G39/G43).
LABELLED_RESIDUE_CLASSES = ("outside_uniprot_span_sifts",)
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

    def unaffected(cls, cid):
        counts = residue_counts.get(cls)
        return (
            cls in LABELLED_RESIDUE_CLASSES
            and counts is not None
            and cid in counts
            and counts[cid][0] == 0
        )

    out_rows, dropped = [], defaultdict(list)
    for cls, ids in sorted(by_class.items()):
        for cid in sorted(ids, key=_id_key):
            if unaffected(cls, cid):
                dropped[cls].append(cid)
            else:
                out_rows.append((cid, f"construct_{cls}", evidence(cls, cid)))
    for cls, ids in dropped.items():
        summary[cls]["listed_without_labelled_residue"] = len(ids)
    return {
        "summary": summary,
        "unknown": sorted(unknown),
        "rows": out_rows,
        "dropped": dict(sorted(dropped.items())),
    }


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
            none = d.get("listed_without_labelled_residue")
            parts.append(
                f"`{cls}` (residue level): "
                + ", ".join(bits)
                + (
                    f" ({_fmt(none)} listed chains have no labelled residue "
                    "there and no known-issue row)"
                    if none
                    else ""
                )
            )
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
