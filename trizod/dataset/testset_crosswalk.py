#!/usr/bin/env python3
"""Crosswalk of every TriZOD test set and its overlap with the final release.

TriZOD's test set has been redrawn or trimmed several times, and each change
turned former test chains into training data. This module lines the sets up
(checks G38 and E21) and writes, into ``--out``:

* ``test_set_crosswalk.tsv`` -- one row per sequence that was ever in a TriZOD
  test set (the earlier sets plus the final ``split == "test_trizod"``):
  ``in_<set>`` / ``<set>_ids`` for every set, and the sequence's status in the
  final release (``release_status``: test / chezod117 / train / redundant /
  retired / excluded / absent, with the splits, IDs, ``train_tier`` and
  ``cluster_repr`` behind it);
* ``test_set_overlap.tsv`` -- per set, how many of its sequences hold each
  status in the final release, training representatives broken down by tier;
* ``test_set_crosswalk.json`` -- the sources (with checksums or git blob IDs),
  the overlap table, set-by-set shared-sequence counts and the match rule.

Matching is by **exact full-length sequence** (case-insensitive), because IDs
change between BMRB snapshots (e.g. a withdrawn entry re-deposited under a new
ID). It therefore undercounts overlap: a homologue at >= 30 % identity / 80 %
coverage, a construct that differs by a tag, or a fragment does not match.

Statuses come from the release ``split`` column. Where several release rows
share a sequence (duplicate depositions), ``release_status`` takes the first of
:data:`STATUS_PRIORITY` among their splits and ``release_splits`` lists all of
them. ``redundant`` means a training representative heads the sequence's
50 %/80 % cluster (``release_cluster_repr``), so a near-homologue trains.

The earlier sets (:func:`default_set_specs`):

* ``trizod2024_348`` -- the 2024 figshare release test set
  (10.6084/m9.figshare.25792035), "Trizod348" in the UdonPred preprint; local
  copy ``data/external/chezod117/TriZOD_test_set.fasta``;
* ``v0.2.0_344`` / ``v0.3.0_365`` -- the Zenodo releases' ``test_trizod`` rows;
* ``pin_2026-07-13_342``, ``pin_2026-08-11_460``, ``pin_2026-11-10_452`` -- the
  committed pin ``trizod/dataset/pinned/TriZOD_test_set.fasta`` at the commit
  that wrote each (v0.3.0's 365 is the pin of 2026-07-14, identical by ID).

Usage
-----
    uv run --with pyarrow trizod dataset test-crosswalk \\
        --parquet <final trizod_dataset.parquet> [--out DIR] [--root DIR] \\
        [--set NAME=fasta:PATH|parquet:PATH|git:REV:PATH ...]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from trizod import paths
from trizod.dataset.export_jsonl import natural_key, read_release
from trizod.dataset.tiers import TIER_ORDER

#: Release ``split`` -> crosswalk status.
SPLIT_STATUS = {
    "test_trizod": "test",
    "test_chezod117": "chezod117",
    "train": "train",
    "redundant": "redundant",
    "test_retired": "retired",
    "excluded": "excluded",
}
#: Which status wins when release rows sharing a sequence disagree: the one
#: that matters most for leakage first.
STATUS_PRIORITY = ("test", "chezod117", "train", "redundant", "retired", "excluded")
STATUSES = (*STATUS_PRIORITY, "absent")
FINAL_SET = "final"
PIN_PATH = "trizod/dataset/pinned/TriZOD_test_set.fasta"
MATCH_RULE = (
    "exact full-length sequence (case-insensitive); a lower bound on overlap "
    "under the 30 %/80 % leakage rule (homologues, tagged constructs and "
    "fragments do not match)"
)
_RELEASE_COLUMNS = ("id", "sequence", "split", "train_tier", "cluster_repr")


@dataclass(frozen=True)
class HeldOutSet:
    """A named test set: ``members`` maps chain ID -> sequence; ``source`` is
    the spec it was loaded from plus its checksum or git commit/blob."""

    name: str
    description: str
    source: dict
    members: dict[str, str]


def _norm(seq: str) -> str:
    return "".join(seq.split()).upper()


def parse_fasta_text(text: str) -> dict[str, str]:
    """``{id: sequence}`` from FASTA text (first header token is the ID)."""
    recs: dict[str, str] = {}
    cur = None
    for line in text.splitlines():
        if line.startswith(">"):
            cur = line[1:].split()[0]
            recs[cur] = ""
        elif cur is not None:
            recs[cur] += line.strip()
    return recs


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


def load_source(spec: str, repo: Path | None = None) -> dict[str, str]:
    """Members of a test set from ``fasta:PATH``, ``parquet:PATH`` (its
    ``test_trizod`` rows) or ``git:REV:PATH`` (a file at a commit of ``repo``)."""
    kind, _, loc = spec.partition(":")
    if kind == "fasta":
        return parse_fasta_text(Path(loc).read_text())
    if kind == "parquet":
        rows, _ = read_release(loc, columns=("id", "sequence", "split"))
        return {r["id"]: r["sequence"] for r in rows if r["split"] == "test_trizod"}
    if kind == "git":
        rev, _, path = loc.partition(":")
        return parse_fasta_text(
            _git(Path(repo or paths.repo_root()), "show", f"{rev}:{path}")
        )
    raise ValueError(f"unknown set source {spec!r} (fasta:, parquet: or git:)")


def describe_source(spec: str, repo: Path | None = None) -> dict:
    """Checksum (file) or commit + blob ID (git) of a set source."""
    kind, _, loc = spec.partition(":")
    if kind == "git":
        rev, _, path = loc.partition(":")
        repo = Path(repo or paths.repo_root())
        return {
            "commit": _git(repo, "rev-parse", f"{rev}^{{commit}}").strip(),
            "blob": _git(repo, "rev-parse", f"{rev}:{path}").strip(),
            "path": path,
        }
    p = Path(loc)
    return {
        "path": str(p.resolve()),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }


def default_set_specs(root) -> list[tuple[str, str, str]]:
    """``(name, source spec, description)`` of every earlier TriZOD test set.

    ``root`` holds ``data/`` (the main checkout; a worktree has none). The pin
    revisions are commits of this repository.
    """
    root = Path(root)
    rel = root / "data" / "release"
    return [
        (
            "trizod2024_348",
            f"fasta:{root / 'data' / 'external' / 'chezod117' / 'TriZOD_test_set.fasta'}",
            "2024 figshare release test set (10.6084/m9.figshare.25792035); "
            "'Trizod348' in the UdonPred preprint",
        ),
        (
            "v0.2.0_344",
            f"parquet:{rel / 'trizod-dataset-v0.2.0' / 'trizod_dataset.parquet'}",
            "Zenodo v0.2.0 (2026-06), test_trizod rows",
        ),
        (
            "pin_2026-07-13_342",
            f"git:0b0b151:{PIN_PATH}",
            "first committed pin (post-fix rebuild of 2026-07-13)",
        ),
        (
            "v0.3.0_365",
            f"parquet:{rel / 'trizod-dataset-v0.3.0' / 'trizod_dataset.parquet'}",
            "Zenodo v0.3.0 (2026-07), test_trizod rows; = pin 7bff12c of 2026-07-14",
        ),
        (
            "pin_2026-08-11_460",
            f"git:cc5b006:{PIN_PATH}",
            "seeded redraw from the final strict pool (2026-08-11)",
        ),
        (
            "pin_2026-11-10_452",
            f"git:0dfe124:{PIN_PATH}",
            "trim to the chains that still resolve (2026-11-10)",
        ),
    ]


def release_status(splits) -> str:
    """The crosswalk status of a sequence held by release rows of ``splits``."""
    found = set()
    for s in splits:
        if s not in SPLIT_STATUS:
            raise ValueError(f"unknown release split {s!r}")
        found.add(SPLIT_STATUS[s])
    for status in STATUS_PRIORITY:
        if status in found:
            return status
    return "absent"


def _strictest(tiers) -> str:
    known = [t for t in tiers if t in TIER_ORDER]
    return max(known, key=TIER_ORDER.index) if known else ""


def build_crosswalk(sets, release_rows) -> list[dict]:
    """One row per sequence of any set or of the final test set."""
    final = {
        r["id"]: r["sequence"] for r in release_rows if r["split"] == "test_trizod"
    }
    named = [(s.name, s.members) for s in sets] + [(FINAL_SET, final)]
    names = [n for n, _ in named]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate set names: {names}")

    index: dict[str, list[dict]] = {}
    for r in release_rows:
        index.setdefault(_norm(r["sequence"]), []).append(r)

    # sequence -> set name -> ids, in order of first appearance
    by_seq: dict[str, dict[str, list[str]]] = {}
    for name, members in named:
        for rid in sorted(members, key=natural_key):
            by_seq.setdefault(_norm(members[rid]), {}).setdefault(name, []).append(rid)

    out = []
    for seq, membership in by_seq.items():
        rel = index.get(seq, [])
        row = {
            "seq_hash": hashlib.sha256(seq.encode()).hexdigest()[:16],
            "length": len(seq),
            "n_sets": len(membership),
        }
        for name in names:
            ids = membership.get(name, [])
            row[f"in_{name}"] = int(bool(ids))
            row[f"{name}_ids"] = ";".join(ids)
        splits = sorted({r["split"] for r in rel})
        row["release_status"] = release_status(splits)
        row["release_splits"] = ";".join(splits)
        row["release_ids"] = ";".join(sorted((r["id"] for r in rel), key=natural_key))
        row["release_train_tier"] = _strictest(
            r.get("train_tier") for r in rel if r["split"] == "train"
        )
        row["release_cluster_repr"] = ";".join(
            sorted(
                {r["cluster_repr"] for r in rel if r.get("cluster_repr")},
                key=natural_key,
            )
        )
        row["sequence"] = seq
        out.append(row)
    return out


def _set_names(sets) -> list[str]:
    return [s.name for s in sets] + [FINAL_SET]


def overlap_summary(rows, sets) -> list[dict]:
    """Per set: sequences by final-release status, training reps by tier."""
    desc = {s.name: s.description for s in sets}
    desc[FINAL_SET] = "final release test set (split == test_trizod)"
    tol = TIER_ORDER.index("tolerant")
    out = []
    for name in _set_names(sets):
        mine = [r for r in rows if r[f"in_{name}"]]
        rec = {
            "set": name,
            "n_ids": sum(len(r[f"{name}_ids"].split(";")) for r in mine),
            "n_sequences": len(mine),
        }
        for status in STATUSES:
            rec[status] = sum(r["release_status"] == status for r in mine)
        for tier in reversed(TIER_ORDER):
            rec[f"train_{tier}"] = sum(
                r["release_status"] == "train" and r["release_train_tier"] == tier
                for r in mine
            )
        rec["in_tolerant_training"] = sum(
            r["release_status"] == "train"
            and TIER_ORDER.index(r["release_train_tier"]) >= tol
            for r in mine
        )
        rec["description"] = desc[name]
        out.append(rec)
    return out


def pairwise_shared(rows, names) -> dict[str, dict[str, int]]:
    """Number of sequences shared by every pair of sets (diagonal: set size)."""
    return {
        a: {b: sum(1 for r in rows if r[f"in_{a}"] and r[f"in_{b}"]) for b in names}
        for a in names
    }


def write_tsv(rows, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        w.writeheader()
        w.writerows(rows)


def run(sets, release_rows, out, release_source: dict) -> dict:
    """Build and write the crosswalk, overlap table and provenance JSON."""
    out = Path(out)
    rows = build_crosswalk(sets, release_rows)
    summary = overlap_summary(rows, sets)
    write_tsv(rows, out / "test_set_crosswalk.tsv")
    write_tsv(summary, out / "test_set_overlap.tsv")
    payload = {
        "match": MATCH_RULE,
        "release": release_source,
        "statuses": {
            "test": "split == test_trizod (the final TriZOD test set)",
            "chezod117": "split == test_chezod117",
            "train": "a training representative (train_tier: strictest tier)",
            "redundant": "in a training pool, clustered (50 %/80 %) under a "
            "training representative (release_cluster_repr)",
            "retired": "split == test_retired: pinned test chain that no longer "
            "meets strict; neither test nor training",
            "excluded": "scored but in no tier's training pool",
            "absent": "no release row has this exact sequence",
            "priority": list(STATUS_PRIORITY),
        },
        "sets": [
            {"name": s.name, "description": s.description, "source": s.source}
            for s in sets
        ],
        "n_sequences": len(rows),
        "overlap": summary,
        "pairwise_shared_sequences": pairwise_shared(rows, _set_names(sets)),
    }
    (out / "test_set_crosswalk.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, required=True, help="final release Parquet")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: <parquet dir>/crosswalk)",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="directory holding data/ for the default sets (default: repo root)",
    )
    ap.add_argument(
        "--set",
        dest="sets",
        action="append",
        metavar="NAME=SPEC",
        help="replace the default earlier sets; SPEC is fasta:PATH, parquet:PATH "
        "or git:REV:PATH (repeatable)",
    )
    args = ap.parse_args(argv)

    repo = paths.repo_root()
    if args.sets:
        specs = []
        for item in args.sets:
            name, sep, spec = item.partition("=")
            if not sep:
                raise SystemExit(f"--set expects NAME=SPEC: {item!r}")
            specs.append((name, spec, spec))
    else:
        specs = default_set_specs(args.root or repo)

    sets = []
    for name, spec, desc in specs:
        kind, _, loc = spec.partition(":")
        if kind in ("fasta", "parquet") and not Path(loc).is_file():
            raise SystemExit(
                f"{name}: {loc} not found -- pass --root <checkout with data/> "
                "or --set NAME=SPEC"
            )
        source = {"spec": spec, **describe_source(spec, repo)}
        sets.append(HeldOutSet(name, desc, source, load_source(spec, repo)))

    release_rows, meta = read_release(args.parquet, columns=_RELEASE_COLUMNS)
    release_source = {
        "path": str(args.parquet.resolve()),
        "sha256": hashlib.sha256(args.parquet.read_bytes()).hexdigest(),
        "split_values": meta.get("split_values"),
    }
    out = args.out or args.parquet.parent / "crosswalk"
    payload = run(sets, release_rows, out, release_source)

    print(f"Wrote test-set crosswalk ({payload['n_sequences']} sequences) -> {out}")
    print(f"  match: {MATCH_RULE}")
    cols = (
        "n_sequences",
        "test",
        "train",
        "in_tolerant_training",
        "redundant",
        "retired",
        "excluded",
        "absent",
    )
    print("  " + f"{'set':22s}" + "".join(f"{c[:12]:>13s}" for c in cols))
    for rec in payload["overlap"]:
        print("  " + f"{rec['set']:22s}" + "".join(f"{rec[c]:13d}" for c in cols))


if __name__ == "__main__":
    main()
