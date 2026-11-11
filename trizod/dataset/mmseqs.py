"""Shared mmseqs2 helpers for the dataset-build chain.

The ``COMMON`` option block is used verbatim by every clustering step and by the
test-set draw (``--alignment-mode 3 --cov-mode 0 -s 7.5 --comp-bias-corr 0
--mask 0``); do not change it without rebuilding + re-verifying the whole funnel.

Test-set leakage is defined separately, by :func:`leakage_search_cmd` and
:func:`is_leak`, under one of two rules (``LEAKAGE_RULES``, decision D2):

* ``"test"``: a training chain leaks when an alignment to a
  held-out test chain reaches >= 30 % identity over >= 80 % of the TEST chain.
  A longer training construct that contains a test protein is a leak; a
  training fragment of a longer test chain is not.
* ``"either"`` (the default, decision D2): >= 30 % identity over >= 80 % of
  the test chain OR of the training chain, i.e. of the shorter of the two (the
  union of an mmseqs ``--cov-mode 1`` and ``--cov-mode 2`` pass). This also
  catches a test chain that contains a shorter training chain, which the
  ``"test"`` rule leaves in (and which the release's substring gate rejects).

Under both rules the alignment must also be reported by mmseqs, i.e. reach
E <= 1e-3: the ``easy-search`` default, which :func:`leakage_search_options`
deliberately leaves in place. Identity and coverage alone would count chance
alignments of short chains (a 20-30 aa pair at 30 % identity is expected by
chance). E-values depend on the search -- which side is the database, its size,
the options -- so a search set up differently can report a borderline pair this
one does not: the paper's Fig. 3c search of test chains against the strict
representatives finds 34481_1_1_1 ~ 15008_1_1_1 (31.8 % identity over 93 % of
the test chain, E = 6e-4), which the stage-2 search does not report. The exact
>= 20 aa substring check of the release gate needs no E-value.

Coverage is the mmseqs definition, the aligned span ``(end - start + 1) /
length``, gaps included. ``--cov-mode 0`` would demand 80 % of both chains, so
neither kind of containment would be found. The leakage search runs query =
training chains against target = test chains, with E-values computed against the
same (test) target database in both places that use it: the stage-2 removal in
:mod:`trizod.dataset.redundancy` and the release gate in
:mod:`trizod.dataset.package_release`. The search command is the same under both
rules; only the coverage test on the parsed hits differs. The redundancy step
records the rule it applied (:func:`write_leakage_rule`) so the gate checks the
same one.

The 80 % is NOT passed to mmseqs (``-c 0``): with any ``-c`` above 0, mmseqs
skips a pair by length ratio before aligning it (under ``--cov-mode 1``, when
the training chain is shorter than 80 % of the test chain). Such a chain can
still span >= 80 % of the test chain with a gap, e.g. a test protein minus an
internal loop, so the coverage threshold is applied afterwards, to the
unrounded spans parsed from the hit table (:func:`is_leak`). With ``-c 0`` the
``--cov-mode`` value has no effect on what mmseqs reports.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

COMMON = [
    "--alignment-mode",
    "3",
    "--cov-mode",
    "0",
    "-s",
    "7.5",
    "--comp-bias-corr",
    "0",
    "--mask",
    "0",
]

# Test-set leakage criterion: >= 30 % identity over >= 80 % of the test chain
# (rule "test") or of either chain (rule "either").
LEAKAGE_MIN_SEQ_ID = 0.3
LEAKAGE_MIN_TEST_COV = 0.8
LEAKAGE_MIN_COV = LEAKAGE_MIN_TEST_COV
LEAKAGE_RULES = ("test", "either")
# Decision D2: the build applies "either" unless --leakage-rule says otherwise.
DEFAULT_LEAKAGE_RULE = "either"
LEAKAGE_RULE_COVERAGE_OF = {
    "test": "the test chain",
    "either": "the test chain or the training chain (at least one of the two)",
}
# Exact-substring containment (either direction) the release gate rejects
# whatever the rule; three real containment pairs are only 21-22 aa long.
SUBSTRING_MIN_LEN = 20
# Written by the redundancy step into its output dir, read by the package step.
LEAKAGE_RULE_FILE = "leakage_rule.json"
# --cov-mode 1 = coverage of the target; the target is always the test set.
LEAKAGE_COV_MODE = 1
# Coverage threshold handed to mmseqs. 0 on purpose: any value above 0 makes
# mmseqs skip pairs by length ratio before alignment (see module docstring);
# LEAKAGE_MIN_TEST_COV is applied to the parsed alignment span instead.
LEAKAGE_MMSEQS_COV = 0
# Start/end columns rather than qcov/tcov, which mmseqs rounds to 3 decimals.
LEAKAGE_FORMAT = "query,target,pident,evalue,bits,qstart,qend,qlen,tstart,tend,tlen"
# No k-mer prefilter: at -s 7.5 it misses weak homologs (30-50 % identity) that
# meet the criterion. Aligning every training x test pair costs seconds here.
LEAKAGE_PREFILTER_MODE = 2


def with_cov_mode(opts: list[str], cov_mode: int) -> list[str]:
    """Copy of an option block with its ``--cov-mode`` value replaced."""
    out = list(opts)
    out[out.index("--cov-mode") + 1] = str(cov_mode)
    return out


def run(cmd: list[str], log: Path, append: bool = False) -> None:
    """Run an mmseqs command, teeing stdout+stderr to ``log``."""
    print("  $", " ".join(str(c) for c in cmd))
    mode = "ab" if append else "wb"
    with open(log, mode) as h:
        subprocess.run(cmd, check=True, stdout=h, stderr=subprocess.STDOUT)


def cluster_tsv_groups(tsv: Path) -> dict:
    """Parse an mmseqs ``*_cluster.tsv`` into {representative: [members]}."""
    groups: dict[str, list[str]] = {}
    with tsv.open() as f:
        for line in f:
            if line.strip():
                rep, mem = line.rstrip("\n").split("\t")[:2]
                groups.setdefault(rep, []).append(mem)
    return groups


def check_rule(rule: str) -> str:
    """Return ``rule`` if it is one of ``LEAKAGE_RULES``, else raise."""
    if rule not in LEAKAGE_RULES:
        raise ValueError(
            f"unknown leakage rule {rule!r}; expected one of {LEAKAGE_RULES}"
        )
    return rule


def describe_rule(rule: str) -> str:
    """The leakage criterion of ``rule`` in words, e.g. ``">= 30 % identity
    over >= 80 % of the test chain"`` -- the one phrasing used by the CLI help,
    the redundancy step's log and the release gate's messages."""
    check_rule(rule)
    return (
        f">= {LEAKAGE_MIN_SEQ_ID * 100:g} % identity over "
        f">= {LEAKAGE_MIN_TEST_COV * 100:g} % of {LEAKAGE_RULE_COVERAGE_OF[rule]}"
    )


#: Every rule and its criterion, for option help texts (argparse help needs its
#: ``%`` doubled).
LEAKAGE_RULES_HELP = "; ".join(f"'{r}': {describe_rule(r)}" for r in LEAKAGE_RULES)


@dataclass(frozen=True)
class LeakageHit:
    """One reported training/test alignment (:func:`parse_leakage_hits`
    returns every alignment; :func:`is_leak` applies the criterion)."""

    train_id: str
    test_id: str
    pident: float  # percent identity over the alignment
    evalue: float
    bits: float
    train_cov: float  # aligned span / length of the training chain (gaps count)
    test_cov: float  # aligned span / length of the test chain (gaps count)
    train_len: int
    test_len: int


def leakage_search_cmd(
    train_fa: Path, test_fa: Path, out_tsv: Path, tmp: Path
) -> list[str]:
    """``mmseqs easy-search`` for the test-set leakage criterion.

    Query = training chains, target = test chains, ``--min-seq-id 0.3`` and
    ``--cov-mode 1 -c 0``; every other option is the shared ``COMMON`` block.
    mmseqs gets no coverage threshold, so it aligns every prefilter hit
    whatever the length ratio; the >= 80 % coverage threshold of the chosen
    rule is applied to the parsed hits (:func:`is_leak`). The command is the
    same for every rule.
    """
    return [
        "mmseqs",
        "easy-search",
        str(train_fa),
        str(test_fa),
        str(out_tsv),
        str(tmp),
        *leakage_search_options(),
    ]


def leakage_search_options() -> list[str]:
    """The options of :func:`leakage_search_cmd` (everything after the paths)."""
    return [
        "--min-seq-id",
        str(LEAKAGE_MIN_SEQ_ID),
        "-c",
        str(LEAKAGE_MMSEQS_COV),
        *with_cov_mode(COMMON, LEAKAGE_COV_MODE),
        "--prefilter-mode",
        str(LEAKAGE_PREFILTER_MODE),
        "--format-output",
        LEAKAGE_FORMAT,
    ]


def parse_leakage_hits(tsv: Path) -> list[LeakageHit]:
    """Parse an easy-search TSV written with ``LEAKAGE_FORMAT`` (every reported
    alignment, whatever its coverage). Coverages are computed from the
    alignment start/end, unrounded."""
    cols = LEAKAGE_FORMAT.split(",")
    hits = []
    with tsv.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = dict(zip(cols, line.rstrip("\n").split("\t")))
            qlen, tlen = int(r["qlen"]), int(r["tlen"])
            hits.append(
                LeakageHit(
                    train_id=r["query"],
                    test_id=r["target"],
                    pident=float(r["pident"]),
                    evalue=float(r["evalue"]),
                    bits=float(r["bits"]),
                    train_cov=(int(r["qend"]) - int(r["qstart"]) + 1) / qlen,
                    test_cov=(int(r["tend"]) - int(r["tstart"]) + 1) / tlen,
                    train_len=qlen,
                    test_len=tlen,
                )
            )
    return hits


def is_leak(hit: LeakageHit, rule: str = DEFAULT_LEAKAGE_RULE) -> bool:
    """The leakage criterion on one reported alignment. Identity >= 30 % and
    E <= 1e-3 are enforced by mmseqs (``--min-seq-id`` and the default ``-e``:
    an alignment above either limit is never reported); coverage is checked
    here: of the test chain (rule ``"test"``), or of the test or the training
    chain (rule ``"either"``)."""
    check_rule(rule)
    if hit.test_cov >= LEAKAGE_MIN_COV:
        return True
    return rule == "either" and hit.train_cov >= LEAKAGE_MIN_COV


def find_leaks(
    train_fa: Path,
    test_fa: Path,
    out_tsv: Path,
    tmp: Path,
    log: Path,
    rule: str = DEFAULT_LEAKAGE_RULE,
) -> list[LeakageHit]:
    """Every training/test pair that meets the leakage criterion of ``rule``
    (one row per pair mmseqs reports). ``out_tsv`` keeps every alignment
    mmseqs reported, including those below the coverage threshold."""
    check_rule(rule)
    run(leakage_search_cmd(train_fa, test_fa, out_tsv, tmp), log)
    return [h for h in parse_leakage_hits(out_tsv) if is_leak(h, rule)]


def find_test_coverage_leaks(
    train_fa: Path, test_fa: Path, out_tsv: Path, tmp: Path, log: Path
) -> list[LeakageHit]:
    """:func:`find_leaks` under rule ``"test"`` (>= 80 % of the test chain)."""
    return find_leaks(train_fa, test_fa, out_tsv, tmp, log, rule="test")


def mmseqs_version() -> str | None:
    """``mmseqs version`` output, or None when it cannot be run."""
    try:
        out = subprocess.run(
            ["mmseqs", "version"], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def leakage_rule_record(rule: str) -> dict:
    """Provenance of the leakage definition a build used (recorded by the
    redundancy step, copied into the release MANIFEST)."""
    check_rule(rule)
    return {
        "rule": rule,
        "min_seq_id": LEAKAGE_MIN_SEQ_ID,
        "min_coverage": LEAKAGE_MIN_COV,
        "coverage_of": LEAKAGE_RULE_COVERAGE_OF[rule],
        "coverage_definition": "aligned span / chain length, gaps included",
        "evalue": "mmseqs easy-search default (1e-3), test chains as target DB",
        "stage2_search": "mmseqs easy-search training (query) vs test (target) "
        + " ".join(leakage_search_options()),
        "stage1_cluster": "mmseqs easy-cluster --min-seq-id 0.3 -c 0.8 "
        + " ".join(COMMON)
        + " (training + test together; unchanged by the rule)",
        "substring_gate_min_len": SUBSTRING_MIN_LEN,
        "mmseqs_version": mmseqs_version(),
    }


def write_leakage_rule(out_dir: Path, rule: str) -> Path:
    """Record the rule a redundancy run applied, as ``<out_dir>/leakage_rule.json``."""
    path = Path(out_dir) / LEAKAGE_RULE_FILE
    path.write_text(json.dumps(leakage_rule_record(rule), indent=2) + "\n")
    return path


def read_leakage_record(out_dir: Path) -> dict | None:
    """The recorded leakage provenance, or None when no run recorded one."""
    path = Path(out_dir) / LEAKAGE_RULE_FILE
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    check_rule(rec.get("rule"))
    return rec


def read_leakage_rule(out_dir: Path) -> str | None:
    """The rule recorded in ``out_dir``, or None when none was recorded."""
    rec = read_leakage_record(out_dir)
    return None if rec is None else rec["rule"]
