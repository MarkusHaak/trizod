#!/usr/bin/env python3
"""Construct or emit the TriZOD test set.

DEFAULT mode: emit the committed pin. Loads the pinned test set from
``trizod/dataset/pinned/TriZOD_test_set.fasta`` and resolves it against the
*current* STRICT-tier pool (``trizod.dataset.testset.resolve_pinned_testset``,
no mmseqs involved) so entry IDs stay valid as the dataset snapshot evolves.

Why strict: the TriZOD test set is defined as strict-tier sequences (2026-11).
It was drawn from the strict tier, and its labels are meant to carry the
strictest guarantees the release makes -- no membrane mimetic, no kilobar
pressure, deposited ambient conditions. Its defining property beyond that,
CheZOD-disjointness at 30 % id / 80 % cov, belongs to the sequences and holds
under any filter change. Test metrics are thus only comparable across
releases on the chains both test sets share. (From 2026-08 to 2026-11 the pin
was resolved against the tolerant pool, to keep the v0.3.0 test set whole;
``--resolve-tier`` still allows that.) Each resolved chain carries a
``label_tier`` -- the strictest tier its sequence still satisfies -- which is
``strict`` for every chain under the default.

RETIRED test chains: a pinned sequence that no longer meets strict after a rule
change or rescore leaves the test set, but it is not forgotten. If it still
resolves against the ``--retire-tier`` pool (default tolerant) it is RETIRED:
written to ``TriZOD_test_retired.fasta`` (``reason=not_strict``, with its
``label_tier``), which ``trizod.dataset.redundancy`` adds to the leakage
targets, ``package_release`` stages under ``test/`` (so its leakage gate checks
it) and the Parquet marks as ``split = test_retired``. Training is therefore
exactly what it would be with the full pin; only the test set shrinks. This is
why the committed pin is NOT trimmed when chains leave strict: the full pin is
what defines the retired set. Only a pinned sequence absent from the retire
pool too (e.g. a withdrawn entry with no live twin) is dropped outright.

Three more reasons retire a chain that is still strict (release audits
2026-11-17 and 2026-11-17b), in this order:

* ``chezod117_overlap`` (E23): the chain meets the leakage criterion of
  ``trizod.dataset.mmseqs`` (>= 30 % identity over >= 80 % of either chain,
  >= 27 bits, either search direction) against a CheZOD117 sequence -- a
  released CheZOD117 chain or one of the 117 original sequences SETH published
  (two of which, "hmbdi" and "6078", are no release chain). 19481_1_1_1
  contains CheZOD117 19482 as an exact 91-residue substring.
* ``shared_construct`` (re-audit 2026-11-17b, item 9 / E26) -- a reviewed
  decision, not a rule: the chain meets no leakage criterion on its deposited
  sequence, but is a near-copy of training chains once its expression tag or
  fusion partner is set aside (52047_1_1_1, a GB1 + N-myc TAD fusion whose
  N-myc part is 97.6 % identical to training 52066_1_1_1). Such chains are
  listed, with the evidence, in the committed table
  ``trizod/dataset/data/test_retirements.tsv`` (:data:`TEST_RETIREMENTS_TSV`,
  :func:`load_test_retirements`), matched to the resolved pin by sequence
  (:func:`match_test_retirements`). The pin itself is never edited for them.
* ``within_test_duplicate`` (E24): the remaining test chains are searched
  against each other at 50 % identity over 80 % of EITHER chain (the coverage
  the release uses everywhere else; the 50 % is the test set's own clustering
  threshold). Of every such pair the shorter chain -- the contained one for an
  exact substring, 6439_1_1_1 in 6440_1_1_1 -- is retired, the lower entry ID
  kept on a tie, greedily from the longest chain down
  (:func:`select_within_test_duplicates`), so no two remaining test chains
  form a pair and no chain is retired for a partner that is itself retired.

The two automatic checks run mmseqs and need the original CheZOD117 FASTA.

``--redraw`` mode (requires ``--confirm-redraw``): reproduces the original
TriZOD test-set recipe (Senoner & Heinzinger, 2024) on the *current* dataset
snapshot, with a FIXED SEED so it is reproducible (the original 2024 draw was
unseeded and could not be regenerated), and OVERWRITES the committed pin with
the result:

  1. Cluster the strict unique sequences together with all CheZOD sequences
     (CheZOD117 + CheZOD1325) at 30% id / 80% cov.
  2. Keep the clusters that contain NO CheZOD sequence (CheZOD-free) — this is
     what makes the resulting test set disjoint from CheZOD.
  3. Randomly sample 25% of those CheZOD-free clusters (seed=SEED).
  4. Recluster the member sequences of the sampled clusters at 50% id / 80%
     cov; the representatives are the TriZOD test set.

CheZOD117 (the external published benchmark) is NOT rebuilt here — only the
in-distribution TriZOD test set is. The result is written to
``<work-dir>/testset/`` and consumed by ``trizod.dataset.redundancy`` as the
TriZOD-test leakage target.

Outputs (``<work-dir>/testset/``):
  TriZOD_test_set.fasta      one record per representative (pinned or redrawn)
  TriZOD_test_retired.fasta  retired pinned chains (leakage targets only),
                             header label_tier=... pinned_id=... reason=...
  TriZOD_test_set_labels.tsv per chain: current ID, pinned (v0.3.0) ID,
                             whether the ID was substituted, label_tier,
                             status (test / retired) and the retire reason
                             (not_strict / chezod117_overlap /
                             shared_construct / within_test_duplicate;
                             empty for a test chain) —
                             this file *is* the released-ID mapping table
  build_test_set_summary.json
  (``--redraw`` additionally writes TriZOD_test_set_clu.tsv: 50/80 cluster
  membership, and overwrites the committed pin + its provenance sidecar)

Usage
-----
    uv run python -m trizod.dataset.testset [--work-dir DIR] [--root DIR]
    uv run python -m trizod.dataset.testset --redraw --confirm-redraw ...
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from trizod.dataset import mmseqs as mm
from trizod.dataset.mmseqs import COMMON, cluster_tsv_groups, run
from trizod.dataset.paths import resolve_paths
from trizod.io.fasta import read_fasta, write_fasta

SEED = 42
SAMPLE_FRACTION = 0.25
CHEZOD_PREFIX = "CHEZOD__"

# Strictest first: the first tier whose pool holds a sequence is its label_tier.
TIER_ORDER = ("strict", "moderate", "tolerant", "unfiltered")
# Tier the pin is resolved against — see the module docstring for why.
RESOLVE_TIER = "strict"
# Looser pool a pinned sequence that left RESOLVE_TIER is retired against: it
# stays a leakage target (see the module docstring), written to RETIRED_FASTA.
RETIRE_TIER = "tolerant"
RETIRED_FASTA = "TriZOD_test_retired.fasta"
UNKNOWN_TIER = "unknown"
# Retire reasons beside f"not_{resolve tier}" (release audit 2026-11-17).
CHEZOD117_OVERLAP = "chezod117_overlap"
WITHIN_TEST_DUPLICATE = "within_test_duplicate"
# A reviewed decision (re-audit 2026-11-17b, item 9): see TEST_RETIREMENTS_TSV.
SHARED_CONSTRUCT = "shared_construct"

#: The committed, reviewed table of pinned test chains retired by decision
#: rather than by a rule (:func:`load_test_retirements`).
TEST_RETIREMENTS_TSV = Path(__file__).resolve().parent / "data" / "test_retirements.tsv"
RETIREMENT_COLUMNS = ("id", "sequence", "reason", "partners", "evidence")
#: Reasons the table may give.
TABLE_REASONS = (SHARED_CONSTRUCT,)
#: The code checkout (``trizod/dataset/testset.py`` -> ``parents[2]``).
_CODE_ROOT = Path(__file__).resolve().parents[2]
_CHAIN_ID = re.compile(r"^\d+_\d+_\d+_\d+$")
_SEQUENCE = re.compile(r"^[A-Z]+$")

REDRAW_WARNING = """\
!! --redraw DESTROYS COMPARABILITY WITH EVERY PUBLISHED TriZOD RELEASE !!
The seeded draw is not stable under a pool change: rng.sample() depends on the
number of CheZOD-free clusters, which moves with every rescore or filter
change. A redraw therefore yields a DIFFERENT test set, invalidating the
published test numbers, anything trained against the current split, and every
DisProt/ROC figure in the manuscript. The default (pinned) mode is what the
release chain uses; --redraw exists only to establish a NEW pin on purpose."""


def _entry_sort_key(entry_id: str):
    """Deterministic key so the lowest-numbered entry ID sharing a sequence is
    chosen. Splits on '_'; numeric parts sort before non-numeric parts."""
    key = []
    for part in entry_id.split("_"):
        key.append((0, int(part)) if part.isdigit() else (1, part))
    return tuple(key)


def load_tier_pools(final_dataset: Path, tiers=TIER_ORDER) -> dict[str, dict[str, str]]:
    """Read the per-tier deduplicated pools, strictest first.

    Returns {tier: {entry_id: sequence}} for the ``<tier>/<tier>.fasta`` files
    that exist; tiers whose FASTA is missing are skipped (they only affect
    ``label_tier`` granularity, so a partial build still resolves).
    """
    pools: dict[str, dict[str, str]] = {}
    for tier in tiers:
        fa = final_dataset / tier / f"{tier}.fasta"
        if fa.exists():
            pools[tier] = read_fasta(fa)
    return pools


def _label_tier(seq: str, tier_seqs: dict[str, set]) -> str:
    """Strictest tier (per ``TIER_ORDER``) whose pool contains ``seq``."""
    for tier, seqs in tier_seqs.items():
        if seq in seqs:
            return tier
    return UNKNOWN_TIER


def resolve_pinned_testset(
    pinned: dict[str, str],
    pool: dict[str, str],
    tier_pools: dict[str, dict[str, str]] | None = None,
) -> tuple[dict[str, str], dict]:
    """Resolve pinned test sequences against the current ``pool`` (strict).

    Each pinned sequence maps to its pinned entry ID if that entry is still in
    the pool, else to the lowest-numbered current entry sharing the identical
    sequence. A pinned sequence absent from the pool is dropped.

    ``tier_pools`` ({tier: {id: seq}}, strictest first — see
    :func:`load_tier_pools`) supplies each resolved chain's ``label_tier``: the
    strictest tier its sequence still satisfies. It never affects membership.

    ``info["id_map"]`` is the full pinned-ID -> current-ID table (one row per
    resolved chain, ``substituted`` flagging the rows where the ID moved), so a
    holder of an older release can repair ID-based joins. The pin matches by
    *sequence*, so a pool-side representative change would otherwise rename a
    test chain silently.
    """
    seq_to_ids: dict[str, list[str]] = {}
    for eid, seq in pool.items():
        seq_to_ids.setdefault(seq, []).append(eid)
    # Sort by TIER_ORDER so the label is the strictest satisfied tier whatever
    # order the caller assembled the pools in; unknown names rank last.
    tier_seqs = {
        t: set((tier_pools or {})[t].values())
        for t in sorted(
            tier_pools or {},
            key=lambda t: TIER_ORDER.index(t) if t in TIER_ORDER else len(TIER_ORDER),
        )
    }

    test_recs: dict[str, str] = {}
    dropped: list[str] = []
    substitutions: list[list[str]] = []
    label_tiers: dict[str, str] = {}
    id_map: list[dict] = []
    for pid, pseq in pinned.items():
        ids = seq_to_ids.get(pseq)
        if not ids:
            dropped.append(pid)
            continue
        chosen = pid if pid in ids else min(ids, key=_entry_sort_key)
        if chosen != pid:
            substitutions.append([pid, chosen])
        if chosen in test_recs:
            raise ValueError(
                f"two pinned sequences resolved to the same entry {chosen}; "
                f"the pin file must have unique sequences"
            )
        test_recs[chosen] = pseq
        label_tiers[chosen] = _label_tier(pseq, tier_seqs)
        id_map.append(
            {
                "pinned_id": pid,
                "test_id": chosen,
                "substituted": chosen != pid,
                "label_tier": label_tiers[chosen],
            }
        )

    counts: dict[str, int] = {}
    for tier in [*tier_seqs, UNKNOWN_TIER]:
        n = sum(1 for t in label_tiers.values() if t == tier)
        if n:
            counts[tier] = n

    info = {
        "pinned_total": len(pinned),
        "resolved": len(test_recs),
        "dropped": dropped,
        "substitutions": substitutions,
        "label_tiers": label_tiers,
        "label_tier_counts": counts,
        "id_map": id_map,
    }
    return test_recs, info


def chezod1325_records(path: Path) -> dict[str, str]:
    """Parse allseqs1325.txt ('<BMRB_ID> <sequence>' per line) into id->seq."""
    recs: dict[str, str] = {}
    for line in path.open():
        parts = line.split()
        if len(parts) >= 2:
            recs[parts[0]] = parts[1]
    return recs


def _easy_cluster(in_fa, pref, work, min_seq_id, log, *, append=False) -> None:
    """Run ``mmseqs easy-cluster`` at ``min_seq_id`` identity and 80% coverage."""
    run(
        [
            "mmseqs",
            "easy-cluster",
            str(in_fa),
            str(pref),
            str(work),
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            "0.8",
            *COMMON,
        ],
        log,
        append=append,
    )


def _redraw(paths, out) -> dict[str, str]:
    """Seeded 30/80 -> sample -> 50/80 recipe. Writes the fasta/clu/summary and
    returns {entry_id: sequence} for the redrawn test set."""
    strict_fasta = paths.final_dataset / "strict" / "strict.fasta"
    tmp = out / "_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    strict = read_fasta(strict_fasta)
    chezod = read_fasta(paths.chezod117)
    chezod.update(chezod1325_records(paths.chezod1325_txt))
    print(f"strict unique seqs: {len(strict)}; CheZOD seqs (117+1325): {len(chezod)}")

    # ---- Step 1: cluster strict + CheZOD @30/80 ----
    step1_in = tmp / "strict_plus_chezod.fasta"
    step1_recs = {
        **strict,
        **{f"{CHEZOD_PREFIX}{rid}": seq for rid, seq in chezod.items()},
    }
    write_fasta(step1_recs, step1_in)
    pref1 = tmp / "c30"
    _easy_cluster(step1_in, pref1, tmp / "w30", 0.3, out / "build_test_set.log")
    groups30 = cluster_tsv_groups(Path(str(pref1) + "_cluster.tsv"))

    # ---- Step 2: CheZOD-free clusters + their strict members ----
    free_clusters: dict[str, list[str]] = {}
    n_chezod_touch = 0
    for rep, members in groups30.items():
        # Past this guard no member carries the CheZOD prefix, so every member
        # of the cluster is a strict sequence.
        if any(m.startswith(CHEZOD_PREFIX) for m in members):
            n_chezod_touch += 1
            continue
        free_clusters[rep] = members
    print(
        f"30/80 clusters: {len(groups30)} "
        f"({n_chezod_touch} CheZOD-touching, {len(free_clusters)} CheZOD-free)"
    )

    # ---- Step 3: random 25% of CheZOD-free clusters (seeded) ----
    rng = random.Random(SEED)
    free_reps = sorted(free_clusters)
    n_sample = round(len(free_reps) * SAMPLE_FRACTION)
    # Keep rng.sample's deterministic (seeded) list order. Wrapping it in a set
    # would make iteration order depend on PYTHONHASHSEED, changing the FASTA
    # written below and hence the order-sensitive mmseqs 50/80 representative
    # pick — defeating the fixed-seed reproducibility this module promises.
    # sampling is already without replacement, so no dedup is needed.
    sampled = rng.sample(free_reps, n_sample)
    sampled_members = [m for rep in sampled for m in free_clusters[rep]]
    print(
        f"sampled {n_sample} / {len(free_reps)} CheZOD-free clusters "
        f"({len(sampled_members)} member sequences) [seed={SEED}]"
    )

    # ---- Step 4: recluster sampled members @50/80 -> test reps ----
    step4_in = tmp / "sampled_members.fasta"
    write_fasta({m: strict[m] for m in sampled_members}, step4_in)
    pref2 = tmp / "c50"
    _easy_cluster(
        step4_in, pref2, tmp / "w50", 0.5, out / "build_test_set.log", append=True
    )
    rep_fasta = Path(str(pref2) + "_rep_seq.fasta")
    clu_tsv = Path(str(pref2) + "_cluster.tsv")

    test_recs = read_fasta(rep_fasta)
    out_fasta = out / "TriZOD_test_set.fasta"
    write_fasta(test_recs, out_fasta)
    shutil.copy2(clu_tsv, out / "TriZOD_test_set_clu.tsv")

    with clu_tsv.open() as fh:
        n_members = sum(1 for ln in fh if ln.strip())
    summary = {
        "seed": SEED,
        "sample_fraction": SAMPLE_FRACTION,
        "strict_unique_seqs": len(strict),
        "chezod_seqs": len(chezod),
        "clusters_30_80": len(groups30),
        "chezod_touching_clusters": n_chezod_touch,
        "chezod_free_clusters": len(free_reps),
        "sampled_clusters": n_sample,
        "sampled_member_seqs": len(sampled_members),
        "test_set_size": len(test_recs),
        "test_set_clu_members": n_members,
    }
    (out / "build_test_set_summary.json").write_text(json.dumps(summary, indent=2))

    return test_recs


def _write_pin(pin_path, test_recs: dict[str, str]) -> None:
    """Overwrite the committed pin FASTA + provenance sidecar."""
    pin_path.parent.mkdir(parents=True, exist_ok=True)
    write_fasta(test_recs, pin_path)
    prov = {
        "count": len(test_recs),
        "seed": SEED,
        "sample_fraction": SAMPLE_FRACTION,
        "written": datetime.date.today().isoformat(),
        "mode": "redraw",
    }
    pin_path.with_suffix(".provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n"
    )


def retire_dropped(
    pinned: dict[str, str],
    dropped: list[str],
    tier_pools: dict[str, dict[str, str]],
    retire_tier: str = RETIRE_TIER,
) -> tuple[dict[str, str], dict]:
    """Resolve the pinned sequences the test pool dropped against the looser
    ``retire_tier`` pool. Those found are the RETIRED test chains (same
    ``(records, info)`` shape as :func:`resolve_pinned_testset`); those not
    found stay in ``info["dropped"]``. No retire pool -> nothing retired."""
    subset = {pid: pinned[pid] for pid in dropped}
    pool = tier_pools.get(retire_tier)
    if pool is None:
        return {}, {"dropped": list(dropped), "label_tiers": {}, "id_map": []}
    return resolve_pinned_testset(subset, pool, tier_pools=tier_pools)


def _write_fasta_with_attrs(path: Path, recs: dict[str, str], attrs: dict) -> None:
    with path.open("w") as fh:
        for rid, seq in recs.items():
            tokens = " ".join(f"{k}={v}" for k, v in attrs[rid].items())
            fh.write(f">{rid} {tokens}\n{seq}\n")


def _label_tier_counts(label_tiers) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tier in [*TIER_ORDER, UNKNOWN_TIER]:
        n = sum(1 for t in label_tiers if t == tier)
        if n:
            counts[tier] = n
    return counts


def retire_chains(
    test_recs: dict[str, str],
    info: dict,
    retired_recs: dict[str, str],
    retired_info: dict,
    ids,
    reason: str,
) -> None:
    """Move the test chains ``ids`` to the retired set, in place: their records
    from ``test_recs`` to ``retired_recs`` and their ID-map rows, tagged with
    ``reason``, from ``info["id_map"]`` to ``retired_info["id_map"]``. The
    test set only shrinks; a retired chain stays a leakage target."""
    ids = set(ids)
    for row in [r for r in info["id_map"] if r["test_id"] in ids]:
        info["id_map"].remove(row)
        retired_info["id_map"].append({**row, "reason": reason})
        retired_recs[row["test_id"]] = test_recs.pop(row["test_id"])
        info["label_tiers"].pop(row["test_id"], None)
    info["label_tier_counts"] = _label_tier_counts(info["label_tiers"].values())


# --------------------------------------------------------------------------- #
# reviewed retirements (re-audit 2026-11-17b, item 9)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReviewedRetirement:
    """One row of the reviewed retirement table: the pinned chain ``id`` with
    ``sequence`` is retired for ``reason``; ``partners`` are the training
    chains it nearly copies, ``evidence`` says how."""

    id: str
    sequence: str
    reason: str
    partners: tuple[str, ...]
    evidence: str


def _display_path(path: Path) -> str:
    """``path`` relative to the code checkout when inside it (the committed
    table), else as given: no absolute home path in a build record."""
    try:
        return Path(path).resolve().relative_to(_CODE_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_test_retirements(path=None) -> list[ReviewedRetirement]:
    """The rows of the reviewed retirement table (default
    :data:`TEST_RETIREMENTS_TSV`), in file order. ``#`` lines and blank lines
    are comments. Raises ``ValueError`` on a wrong header or field count, a
    malformed chain ID or sequence, an unknown reason, a row without partners
    or evidence, or a repeated ID."""
    path = Path(TEST_RETIREMENTS_TSV if path is None else path)
    rows: list[ReviewedRetirement] = []
    header = None
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip() or raw.startswith("#"):
            continue
        fields = [f.strip() for f in raw.split("\t")]
        where = f"{path}:{lineno}"
        if header is None:
            if tuple(fields) != RETIREMENT_COLUMNS:
                raise ValueError(
                    f"{where}: header must be {RETIREMENT_COLUMNS}, got {fields}"
                )
            header = fields
            continue
        if len(fields) != len(RETIREMENT_COLUMNS):
            raise ValueError(
                f"{where}: expected {len(RETIREMENT_COLUMNS)} fields, got {fields}"
            )
        rid, seq, reason, partners, evidence = fields
        if not _CHAIN_ID.match(rid):
            raise ValueError(f"{where}: {rid!r} is no chain ID")
        if not _SEQUENCE.match(seq):
            raise ValueError(f"{where}: {rid}: malformed sequence")
        if reason not in TABLE_REASONS:
            raise ValueError(f"{where}: {rid}: unknown reason {reason!r}")
        partner_ids = tuple(p.strip() for p in partners.split(",") if p.strip())
        if not partner_ids:
            raise ValueError(f"{where}: {rid}: no training partner given")
        if not evidence:
            raise ValueError(f"{where}: {rid}: no evidence given")
        if any(r.id == rid for r in rows):
            raise ValueError(f"{where}: duplicate id {rid}")
        rows.append(ReviewedRetirement(rid, seq, reason, partner_ids, evidence))
    if header is None:
        raise ValueError(f"{path}: no header {RETIREMENT_COLUMNS}")
    return rows


def match_test_retirements(
    pinned: dict[str, str], info: dict, retired_info: dict, rows
) -> tuple[dict[str, ReviewedRetirement], dict]:
    """Match the reviewed rows to the resolved pin (``pinned``, the
    test chains' ``info`` and the retired chains' ``retired_info``, as
    :func:`resolve_pinned_testset` and :func:`retire_chains` leave them), by
    sequence.

    Returns ``({test ID: row}`` of the test chains to retire now, ``report)``.
    A row applies to the pinned chain with its sequence, under its pinned ID or
    the ID it resolved to. The report lists the rows ``applied``, those whose
    chain is ``already_retired`` (an earlier reason wins and is kept), those
    whose chain was ``dropped`` (in no pool) and those ``not_in_pin`` (a row
    for another pin: neither its ID nor its sequence is pinned). A row whose ID
    is pinned with another sequence, or whose sequence is pinned under an ID
    the row does not name, raises ``ValueError``: table and pin disagree."""
    pid_of_seq = {seq: pid for pid, seq in pinned.items()}
    resolved = {row["pinned_id"]: row["test_id"] for row in info["id_map"]}
    retired = {
        row["pinned_id"]: (row["test_id"], row.get("reason"))
        for row in retired_info["id_map"]
    }
    apply: dict[str, ReviewedRetirement] = {}
    report: dict[str, list] = {
        "applied": [],
        "already_retired": [],
        "dropped": [],
        "not_in_pin": [],
    }
    for row in rows:
        if row.id in pinned and pinned[row.id] != row.sequence:
            raise ValueError(
                f"reviewed retirement {row.id}: the pin holds another sequence "
                "under this ID; the table and the pin disagree"
            )
        pid = pid_of_seq.get(row.sequence)
        if pid is None:
            report["not_in_pin"].append(row.id)
            continue
        names = {pid, resolved.get(pid), retired.get(pid, (None,))[0]}
        if row.id not in names:
            raise ValueError(
                f"reviewed retirement {row.id}: its sequence is pinned as {pid}, "
                "which the row does not name"
            )
        entry = {
            "pinned_id": pid,
            "reason": row.reason,
            "partners": list(row.partners),
            "evidence": row.evidence,
        }
        if pid in resolved:
            apply[resolved[pid]] = row
            report["applied"].append({"test_id": resolved[pid], **entry})
        elif pid in retired:
            test_id, reason = retired[pid]
            report["already_retired"].append(
                {"test_id": test_id, "pinned_id": pid, "reason": reason}
            )
        else:
            report["dropped"].append(pid)
    return apply, report


def find_chezod117_overlaps(
    test_recs: dict[str, str], chezod117: dict[str, str], work: Path
) -> list[mm.LeakageHit]:
    """Test chains that meet the leakage criterion (rule ``either``) against a
    CheZOD117 sequence, in either search direction. In each hit the TriZOD
    test chain is ``train_id`` and the CheZOD117 sequence ``test_id``."""
    work.mkdir(parents=True, exist_ok=True)
    test_fa, chezod_fa = work / "trizod_test.fasta", work / "chezod117.fasta"
    write_fasta(test_recs, test_fa)
    write_fasta(chezod117, chezod_fa)
    return mm.find_leaks(
        test_fa,
        chezod_fa,
        work / "chezod117_hits.tsv",
        work / "tmp_chezod117",
        work / "chezod117_search.log",
        rule="either",
    )


def find_within_test_pairs(
    test_recs: dict[str, str], work: Path
) -> list[mm.LeakageHit]:
    """Pairs of distinct test chains at >= 50 % identity over >= 80 % of
    either chain (and the bit floor), one hit per unordered pair. The search
    reports alignments of any identity (``--min-seq-id 0``, so alternatives
    behind a weaker best alignment are seen); the 50 % is applied to each."""
    work.mkdir(parents=True, exist_ok=True)
    fa = work / "trizod_test.fasta"
    write_fasta(test_recs, fa)
    hits = mm.search_both_directions(
        fa,
        fa,
        work / "within_test_hits.tsv",
        work / "tmp_within_test",
        work / "within_test_search.log",
    )
    pairs: dict[frozenset, mm.LeakageHit] = {}
    for h in hits:
        if h.train_id != h.test_id and mm.is_leak(
            h, rule="either", min_seq_id=mm.WITHIN_TEST_MIN_SEQ_ID
        ):
            pairs.setdefault(frozenset((h.train_id, h.test_id)), h)
    return list(pairs.values())


def select_within_test_duplicates(test_recs: dict[str, str], pairs) -> dict[str, str]:
    """``{retired ID: kept partner}`` for near-duplicate ``pairs`` (ID pairs,
    either order) of test chains: going from the longest chain down (lower
    entry ID first on a tie), a chain is kept unless it pairs with a chain
    already kept, in which case it is retired against the first such partner.
    So the shorter chain of a pair -- the contained one of an exact substring
    -- is retired, no two kept chains form a pair, and the result does not
    depend on the order of ``test_recs`` or ``pairs``."""
    partners: dict[str, set[str]] = {}
    for a, b in pairs:
        if a != b:
            partners.setdefault(a, set()).add(b)
            partners.setdefault(b, set()).add(a)

    def priority(cid):
        return (-len(test_recs[cid]), _entry_sort_key(cid))

    kept: list[str] = []
    retired: dict[str, str] = {}
    for cid in sorted(test_recs, key=priority):
        match = [k for k in kept if k in partners.get(cid, ())]
        if match:
            retired[cid] = match[0]
        else:
            kept.append(cid)
    return retired


def _write_testset(
    out: Path,
    test_recs: dict[str, str],
    info: dict,
    retired_recs: dict[str, str] | None = None,
    retired_info: dict | None = None,
    reason: str = f"not_{RESOLVE_TIER}",
) -> None:
    """Emit the resolved test set: annotated FASTA + the label/ID-map TSV, and
    the retired chains' FASTA (written even when empty, so a stale one from an
    earlier run cannot survive). A retired row's own ``reason`` wins over
    ``reason``, the default for rows without one (``not_<resolve tier>``).

    The FASTA header carries ``label_tier`` and the pinned (previous-release)
    ID so the sequence file is self-describing; every consumer in the chain
    keys on the first header token, as the per-tier pool FASTAs already do.
    """
    retired_recs = retired_recs or {}
    retired_rows = [
        {**row, "reason": row.get("reason") or reason}
        for row in (retired_info or {}).get("id_map", [])
    ]

    def attrs(rows, with_reason):
        return {
            row["test_id"]: {
                "label_tier": row["label_tier"],
                "pinned_id": row["pinned_id"],
                **({"reason": row["reason"]} if with_reason else {}),
            }
            for row in rows
        }

    _write_fasta_with_attrs(
        out / "TriZOD_test_set.fasta", test_recs, attrs(info["id_map"], False)
    )
    _write_fasta_with_attrs(
        out / RETIRED_FASTA, retired_recs, attrs(retired_rows, True)
    )
    with (out / "TriZOD_test_set_labels.tsv").open("w") as fh:
        fh.write(
            "test_id\tpinned_id\tsubstituted\tlabel_tier\tlength\tstatus\treason\n"
        )
        for rows, recs, status in (
            (info["id_map"], test_recs, "test"),
            (retired_rows, retired_recs, "retired"),
        ):
            for row in rows:
                fh.write(
                    f"{row['test_id']}\t{row['pinned_id']}\t{row['substituted']}\t"
                    f"{row['label_tier']}\t{len(recs[row['test_id']])}\t{status}\t"
                    f"{row.get('reason', '') if status == 'retired' else ''}\n"
                )


def main(argv=None) -> None:
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
        "--redraw",
        action="store_true",
        help="Redraw the seeded test set and OVERWRITE the committed pin "
        "(default: emit the committed pin). DESTRUCTIVE — also requires "
        "--confirm-redraw.",
    )
    ap.add_argument(
        "--confirm-redraw",
        action="store_true",
        help="Acknowledge that --redraw invalidates every published test "
        "number. Without it, --redraw refuses to run.",
    )
    ap.add_argument(
        "--resolve-tier",
        choices=TIER_ORDER,
        default=RESOLVE_TIER,
        help=f"pool the pinned sequences are resolved against "
        f"(default: {RESOLVE_TIER}; see the module docstring)",
    )
    ap.add_argument(
        "--retire-tier",
        choices=TIER_ORDER,
        default=RETIRE_TIER,
        help=f"looser pool a pinned sequence that left --resolve-tier is "
        f"retired against: kept as a leakage target in {RETIRED_FASTA} "
        f"(default: {RETIRE_TIER}; no retiring unless looser than --resolve-tier)",
    )
    ap.add_argument(
        "--retirements",
        type=Path,
        default=None,
        help="reviewed retirement table (default: the committed "
        "trizod/dataset/data/test_retirements.tsv)",
    )
    args = ap.parse_args(argv)
    paths = resolve_paths(args.work_dir, args.root)
    out = paths.testset
    out.mkdir(parents=True, exist_ok=True)

    if args.redraw:
        if not args.confirm_redraw:
            raise SystemExit(
                f"{REDRAW_WARNING}\n\nRefusing to redraw. Re-run with "
                f"--confirm-redraw if you really mean to break comparability."
            )
        print(REDRAW_WARNING)
        # A labels sidecar or retired list from an earlier pinned run
        # describes the OLD set.
        (out / "TriZOD_test_set_labels.tsv").unlink(missing_ok=True)
        (out / RETIRED_FASTA).unlink(missing_ok=True)
        test_recs = _redraw(paths, out)
        _write_pin(paths.pinned_testset, test_recs)
        print(f"Re-pinned {len(test_recs)} sequences -> {paths.pinned_testset}")
        return

    if not paths.pinned_testset.exists():
        raise SystemExit(
            f"Pinned test set not found at {paths.pinned_testset}. "
            f"Run `trizod dataset test-set --redraw` to establish it."
        )

    tier_pools = load_tier_pools(paths.final_dataset)
    pool = tier_pools.get(args.resolve_tier)
    if pool is None:
        raise SystemExit(
            f"{args.resolve_tier} pool not found at "
            f"{paths.final_dataset / args.resolve_tier / f'{args.resolve_tier}.fasta'}."
            f" Run `trizod dataset build` first."
        )
    pinned = read_fasta(paths.pinned_testset)
    test_recs, info = resolve_pinned_testset(pinned, pool, tier_pools=tier_pools)

    # Pinned sequences that left the test pool: retired (still leakage targets)
    # when the looser retire pool holds them, dropped only when it does not.
    retired_recs, retired_info = {}, {"dropped": info["dropped"], "id_map": []}
    if TIER_ORDER.index(args.retire_tier) > TIER_ORDER.index(args.resolve_tier):
        if args.retire_tier not in tier_pools and info["dropped"]:
            print(
                f"WARNING: no {args.retire_tier} pool -- the "
                f"{len(info['dropped'])} pinned sequences outside "
                f"{args.resolve_tier} cannot be retired and are dropped"
            )
        retired_recs, retired_info = retire_dropped(
            pinned, info["dropped"], tier_pools, args.retire_tier
        )
    not_resolved = f"not_{args.resolve_tier}"
    for row in retired_info["id_map"]:
        row["reason"] = not_resolved

    # CheZOD117 overlaps, then within-test near-duplicates (release audit
    # E23/E24): retired, never deleted -- they stay leakage targets.
    if not shutil.which("mmseqs"):
        raise SystemExit("mmseqs binary not found in PATH (test-set checks)")
    for src in (paths.chezod117, paths.chezod117_original):
        if not src.exists():
            raise SystemExit(
                f"CheZOD117 sequences not found at {src}: the test set is checked "
                "against every released and original CheZOD117 sequence"
            )
    chezod117 = {**read_fasta(paths.chezod117), **read_fasta(paths.chezod117_original)}
    work = out / "_checks"
    if work.exists():
        shutil.rmtree(work)
    overlaps = mm.strongest_per_pair(
        sorted(
            find_chezod117_overlaps(test_recs, chezod117, work),
            key=lambda h: (h.train_id, -h.bits, h.test_id),
        )
    )
    retire_chains(
        test_recs,
        info,
        retired_recs,
        retired_info,
        {h.train_id for h in overlaps},
        CHEZOD117_OVERLAP,
    )
    # Reviewed retirements (re-audit 2026-11-17b, item 9), before the
    # within-test check, which then runs on the chains that stay test chains.
    table = Path(args.retirements or TEST_RETIREMENTS_TSV)
    reviewed, reviewed_report = match_test_retirements(
        pinned, info, retired_info, load_test_retirements(table)
    )
    for reason in sorted({row.reason for row in reviewed.values()}):
        retire_chains(
            test_recs,
            info,
            retired_recs,
            retired_info,
            {tid for tid, row in reviewed.items() if row.reason == reason},
            reason,
        )
    pair_hits = find_within_test_pairs(test_recs, work)
    duplicates = select_within_test_duplicates(
        test_recs, [(h.train_id, h.test_id) for h in pair_hits]
    )
    retire_chains(
        test_recs, info, retired_recs, retired_info, duplicates, WITHIN_TEST_DUPLICATE
    )
    hit_of = {frozenset((h.train_id, h.test_id)): h for h in pair_hits}

    retired = [row["pinned_id"] for row in retired_info["id_map"]]
    info["dropped"] = retired_info["dropped"]
    info["retired"] = retired
    info["retired_id_map"] = retired_info["id_map"]
    by_reason: dict[str, list[str]] = {}
    for row in retired_info["id_map"]:
        by_reason.setdefault(row["reason"], []).append(row["test_id"])
    info["retired_by_reason"] = by_reason

    _write_testset(out, test_recs, info, retired_recs, retired_info, not_resolved)
    first_overlap: dict[str, mm.LeakageHit] = {}
    for h in overlaps:
        first_overlap.setdefault(h.train_id, h)
    summary = {
        "mode": "pinned",
        "resolve_tier": args.resolve_tier,
        "retire_tier": args.retire_tier,
        "test_set_size": len(test_recs),
        **info,
        "leakage_criterion": mm.describe_rule("either"),
        "chezod117_overlaps": [
            {
                "test_id": h.train_id,
                "chezod117_id": h.test_id,
                "pident": h.pident,
                "bits": h.bits,
                "test_cov": round(h.train_cov, 3),
                "chezod117_cov": round(h.test_cov, 3),
                "direction": h.direction,
            }
            for h in first_overlap.values()
        ],
        "reviewed_retirements": {
            "table": _display_path(table),
            "sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
            **reviewed_report,
        },
        "within_test_criterion": (
            f">= {mm.WITHIN_TEST_MIN_SEQ_ID * 100:g} % identity over >= "
            f"{mm.LEAKAGE_MIN_COV * 100:g} % of either chain, >= "
            f"{mm.LEAKAGE_MIN_BITS:g} bits"
        ),
        "within_test_duplicates": [
            [rid, kid, hit_of[frozenset((rid, kid))].pident]
            for rid, kid in duplicates.items()
        ],
    }
    (out / "build_test_set_summary.json").write_text(json.dumps(summary, indent=2))

    if retired:
        print(
            f"NOTE: {len(retired)} pinned sequences were retired: not test chains "
            f"any more, but written to {out / RETIRED_FASTA} as leakage targets, "
            f"so they and their homologues stay out of training. Do not trim "
            f"them from the pin. By reason: "
            + "; ".join(f"{r}: {len(ids)} {ids}" for r, ids in by_reason.items())
        )
    if reviewed_report["not_in_pin"]:
        print(
            f"NOTE: {len(reviewed_report['not_in_pin'])} rows of {table} name no "
            f"chain of this pin and were not applied: {reviewed_report['not_in_pin']}"
        )
    if info["dropped"]:
        print(
            f"WARNING: {len(info['dropped'])} pinned sequences are in neither "
            f"the {args.resolve_tier} nor the {args.retire_tier} pool and were "
            f"dropped: {info['dropped']}"
        )
    if info["substitutions"]:
        print(
            f"WARNING: {len(info['substitutions'])} pinned representatives were "
            f"substituted by the lowest-numbered entry with the same sequence. "
            f"The sequence is unchanged but the ID is NOT the released one — "
            f"ID-based joins against an older release must go through "
            f"TriZOD_test_set_labels.tsv: {info['substitutions']}"
        )
    non_strict = {t: n for t, n in info["label_tier_counts"].items() if t != "strict"}
    if non_strict:
        print(
            f"NOTE: {sum(non_strict.values())} test chains do not meet the "
            f"strict criteria (label_tier: {non_strict}); they are retained "
            f"because the pin was resolved against {args.resolve_tier}, not "
            f"strict."
        )
    print(
        f"TriZOD test set (pinned, resolved against {args.resolve_tier}): "
        f"{len(test_recs)} sequences -> {out / 'TriZOD_test_set.fasta'}"
    )


if __name__ == "__main__":
    main()
