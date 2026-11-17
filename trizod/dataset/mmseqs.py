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

Under both rules the alignment must also score >= ``LEAKAGE_MIN_BITS`` (27)
bits. Identity and coverage alone would count chance alignments of short chains
(a 20-30 aa pair at 30 % identity is expected by chance), so a significance
floor is needed -- but not an E-value: an mmseqs E-value scales with the query
length and the size of the target database, so the same pair passed or failed
the former ``E <= 1e-3`` cut depending on which side was the query and how many
chains the target set held (release audit E22: 7350~52238, 30761~26610 and
34286~52584 sat at E 5e-4 to 1e-3 with the test chain as query, above 1e-3 the
other way round). The bit score is a property of the alignment alone. The floor
is the most permissive bit score the former cut admitted in the 2026-11 build
(27 bits, its shortest queries against the ~570-chain test database), so every
pair it removed is still removed; for two 100-residue chains a chance local
alignment of 27 bits has a pairwise E-value of about 1e-4, and the identity and
coverage thresholds carry the rest. mmseqs is run with a permissive
``-e LEAKAGE_SEARCH_EVALUE`` only so that it reports every alignment above the
floor; :func:`check_evalue_headroom` fails a search in which the cap could have
hidden one.

The search runs in BOTH directions (:func:`search_both_directions`): training
chains as queries against the test chains, and test chains as queries against
the training chains. The alignment score is symmetric, but the aligned span is
not always: 26357_1_1_1 ~ CheZOD117 19135 covers 76 % of the training chain with
the training chain as query and 80 % the other way round. A pair counts when
either search meets the criterion.

The criterion is applied to EVERY alignment mmseqs reports for a pair, not
only to the best-scoring one: the search runs with ``--alt-ali``
(``LEAKAGE_ALT_ALI``, up to 10 alternative alignments per pair besides the
best). A repeat or a tandem construct can hold a second alignment that meets
the rule while the best one does not (release re-audit 2026-11-17b, E20/E22:
51068_1_1_1 ~ test 51080_1_1_1, best alignment 61.5 % over 49 % of 51068,
second 43.1 % over 84 %; 4029_1_1_1 ~ 19298_1_1_1; 6280_1_1_1 ~ retired
34124_1_1_1). mmseqs finds an alternative alignment after masking the target
span of the previous one, so a pair can need either direction for its second
alignment too. The same search serves the stage-2 removal
in :mod:`trizod.dataset.redundancy`, the release gate in
:mod:`trizod.dataset.package_release`, the test-set checks in
:mod:`trizod.dataset.testset` (CheZOD117 overlap, within-test duplicates), the
CheZOD-overlap flags of the Parquet and the homology components of the
UdonPred validation split (:mod:`trizod.dataset.export_jsonl`). The exact
>= 20 aa substring check of the release gate needs no score at all.

The 30 % identity is NOT passed to mmseqs either (``--min-seq-id 0``,
``LEAKAGE_SEARCH_MIN_SEQ_ID``): mmseqs computes a pair's alternative
alignments only behind a best alignment that passes ``--min-seq-id``, so with
0.3 a pair whose best alignment is below 30 % identity was dropped with all
its alternatives (release validation 2026-11-17c, E20/E22, decision option a:
36490_1_1_1 ~ retired 53299_1_1_1, best 25.8 % at 38 bits, second 32.9 % over
81.9 % of 36490 at 32 bits; 50596_1_1_1 ~ test 30007_1_1_1, best 28.5 %,
second 35.0 % over 80.0 % of 30007). :func:`is_leak` applies the identity
floor to every reported alignment instead (``pident``, printed by mmseqs to
three decimals).

Coverage is the mmseqs definition, the aligned span ``(end - start + 1) /
length``, gaps included. ``--cov-mode 0`` would demand 80 % of both chains, so
neither kind of containment would be found. The search command is the same
under both rules; only the coverage test on the parsed hits differs. The
redundancy step records the rule it applied (:func:`write_leakage_rule`) so the
gate checks the same one.

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
import math
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
#: mmseqs --min-seq-id of every leakage search: 0, so that mmseqs computes the
#: alternative alignments of a pair whose best alignment is below the rule's
#: identity (validation 2026-11-17c, E20/E22). The rule's identity
#: (LEAKAGE_MIN_SEQ_ID, or WITHIN_TEST_MIN_SEQ_ID) is applied by is_leak.
LEAKAGE_SEARCH_MIN_SEQ_ID = 0
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
# Significance floor of a leak: the alignment's bit score, which does not depend
# on the search direction or the database size (see the module docstring).
LEAKAGE_MIN_BITS = 27.0
# mmseqs -e: permissive, so that every alignment above LEAKAGE_MIN_BITS is
# reported; check_evalue_headroom fails a search where it could bind.
LEAKAGE_SEARCH_EVALUE = 100.0
# The two orientations of a leakage search.
FORWARD, REVERSE = "forward", "reverse"
#: The within-test near-duplicate check (release audit E24): 50 % identity over
#: 80 % of either chain, same search, same bit floor.
WITHIN_TEST_MIN_SEQ_ID = 0.5
#: mmseqs --alt-ali: alternative alignments reported per pair besides the best
#: one; the criterion is applied to every reported alignment (re-audit
#: 2026-11-17b, decision 1a). 10 is what the audit searched with.
LEAKAGE_ALT_ALI = 10
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
        f", >= {LEAKAGE_MIN_BITS:g} bits, either search direction, "
        f"every reported alignment (--alt-ali {LEAKAGE_ALT_ALI}, "
        f"--min-seq-id {LEAKAGE_SEARCH_MIN_SEQ_ID:g})"
    )


#: Every rule and its criterion, for option help texts (argparse help needs its
#: ``%`` doubled).
LEAKAGE_RULES_HELP = "; ".join(f"'{r}': {describe_rule(r)}" for r in LEAKAGE_RULES)


@dataclass(frozen=True)
class LeakageHit:
    """One reported training/test alignment (:func:`parse_leakage_hits`
    returns every alignment; :func:`is_leak` applies the criterion).
    ``direction`` is the search that reported it: ``"forward"`` (training
    chain as query) or ``"reverse"`` (test chain as query)."""

    train_id: str
    test_id: str
    pident: float  # percent identity over the alignment
    evalue: float
    bits: float
    train_cov: float  # aligned span / length of the training chain (gaps count)
    test_cov: float  # aligned span / length of the test chain (gaps count)
    train_len: int
    test_len: int
    direction: str = FORWARD


def leakage_search_cmd(
    query_fa: Path,
    target_fa: Path,
    out_tsv: Path,
    tmp: Path,
    min_seq_id: float = LEAKAGE_SEARCH_MIN_SEQ_ID,
) -> list[str]:
    """``mmseqs easy-search`` for the test-set leakage criterion, one direction.

    The forward search has the training chains as ``query_fa`` and the test
    chains as ``target_fa``; :func:`search_both_directions` adds the reverse.
    ``--min-seq-id 0``, ``--cov-mode 1 -c 0``, a permissive ``-e`` and
    ``--alt-ali 10`` (every alternative alignment of a pair is reported, not
    only the best, also behind a best alignment below 30 % identity); every
    other option is the shared ``COMMON`` block. mmseqs gets no identity and no
    coverage threshold, so it aligns every pair whatever the length ratio; the
    identity, the >= 80 % coverage of the chosen rule and the bit floor are
    applied to every parsed alignment (:func:`is_leak`). The command is the
    same for every rule.
    """
    return [
        "mmseqs",
        "easy-search",
        str(query_fa),
        str(target_fa),
        str(out_tsv),
        str(tmp),
        *leakage_search_options(min_seq_id),
    ]


def leakage_search_options(
    min_seq_id: float = LEAKAGE_SEARCH_MIN_SEQ_ID, alt_ali: int = LEAKAGE_ALT_ALI
) -> list[str]:
    """The options of :func:`leakage_search_cmd` (everything after the paths).
    ``alt_ali=0`` drops ``--alt-ali`` (best alignment per pair only): for a
    control, never for the release."""
    return [
        "--min-seq-id",
        f"{min_seq_id:g}",
        "-c",
        str(LEAKAGE_MMSEQS_COV),
        *with_cov_mode(COMMON, LEAKAGE_COV_MODE),
        "--prefilter-mode",
        str(LEAKAGE_PREFILTER_MODE),
        "-e",
        f"{LEAKAGE_SEARCH_EVALUE:g}",
        *(["--alt-ali", str(alt_ali)] if alt_ali else []),
        # amino acids, never auto-detected: a FASTA of a few short chains that
        # happen to hold only A/C/G/T (GAGA... peptides) reads as nucleotides
        "--dbtype",
        "1",
        "--format-output",
        LEAKAGE_FORMAT,
    ]


def reverse_hits_path(out_tsv: Path) -> Path:
    """Where :func:`search_both_directions` writes the reverse search's hits:
    ``<name>_reverse.tsv`` beside the forward ``out_tsv``."""
    out_tsv = Path(out_tsv)
    return out_tsv.with_name(f"{out_tsv.stem}_reverse{out_tsv.suffix}")


def parse_leakage_hits(tsv: Path, reverse: bool = False) -> list[LeakageHit]:
    """Parse an easy-search TSV written with ``LEAKAGE_FORMAT`` (every reported
    alignment, whatever its coverage). Coverages are computed from the
    alignment start/end, unrounded. ``reverse``: the test chain was the query
    (the reverse search); the hit is still keyed (training, test)."""
    cols = LEAKAGE_FORMAT.split(",")
    hits = []
    with Path(tsv).open() as f:
        for line in f:
            if not line.strip():
                continue
            r = dict(zip(cols, line.rstrip("\n").split("\t")))
            qlen, tlen = int(r["qlen"]), int(r["tlen"])
            qcov = (int(r["qend"]) - int(r["qstart"]) + 1) / qlen
            tcov = (int(r["tend"]) - int(r["tstart"]) + 1) / tlen
            common = {
                "pident": float(r["pident"]),
                "evalue": float(r["evalue"]),
                "bits": float(r["bits"]),
            }
            if reverse:
                hit = LeakageHit(
                    train_id=r["target"],
                    test_id=r["query"],
                    train_cov=tcov,
                    test_cov=qcov,
                    train_len=tlen,
                    test_len=qlen,
                    direction=REVERSE,
                    **common,
                )
            else:
                hit = LeakageHit(
                    train_id=r["query"],
                    test_id=r["target"],
                    train_cov=qcov,
                    test_cov=tcov,
                    train_len=qlen,
                    test_len=tlen,
                    direction=FORWARD,
                    **common,
                )
            hits.append(hit)
    return hits


def is_leak(
    hit: LeakageHit,
    rule: str = DEFAULT_LEAKAGE_RULE,
    min_bits: float = LEAKAGE_MIN_BITS,
    min_seq_id: float = LEAKAGE_MIN_SEQ_ID,
) -> bool:
    """The leakage criterion on one reported alignment: >= ``min_seq_id``
    identity (30 %; the within-test check passes ``WITHIN_TEST_MIN_SEQ_ID``),
    >= ``min_bits`` bits, and >= 80 % coverage of the test chain (rule
    ``"test"``) or of the test or the training chain (rule ``"either"``).
    mmseqs reports alignments of any identity (``--min-seq-id 0``), so the
    identity is checked here, on each alignment."""
    check_rule(rule)
    if hit.pident < 100 * min_seq_id - 1e-9:
        return False
    if hit.bits < min_bits:
        return False
    if hit.test_cov >= LEAKAGE_MIN_COV:
        return True
    return rule == "either" and hit.train_cov >= LEAKAGE_MIN_COV


def _fasta_lengths(path: Path) -> list[int]:
    lengths, n = [], None
    with Path(path).open() as f:
        for line in f:
            if line.startswith(">"):
                if n is not None:
                    lengths.append(n)
                n = 0
            elif n is not None:
                n += len(line.strip())
    if n is not None:
        lengths.append(n)
    return lengths


def check_evalue_headroom(
    tsv: Path,
    query_fa: Path,
    min_bits: float = LEAKAGE_MIN_BITS,
    cap: float = LEAKAGE_SEARCH_EVALUE,
) -> float | None:
    """Fail when the search's ``-e`` cap could have hidden an alignment above
    the bit floor.

    An mmseqs E-value is ``k * query length * 2**-bits`` with ``k`` set by the
    target database. ``k`` is read off the reported alignments (its largest
    value), and the E-value a ``min_bits`` alignment of the longest query would
    get must stay below ``cap``; otherwise raise ``RuntimeError`` (raise
    ``LEAKAGE_SEARCH_EVALUE``). Returns that bound, None for a search without
    hits."""
    log2_k = None  # in log space: 2**bits overflows for a long identical pair
    with Path(tsv).open() as f:
        for line in f:
            if not line.strip():
                continue
            r = dict(zip(LEAKAGE_FORMAT.split(","), line.rstrip("\n").split("\t")))
            evalue = float(r["evalue"])
            if evalue <= 0:  # underflowed: says nothing about k
                continue
            est = math.log2(evalue) + float(r["bits"]) - math.log2(int(r["qlen"]))
            log2_k = est if log2_k is None else max(log2_k, est)
    if log2_k is None:
        return None
    longest = max(_fasta_lengths(query_fa), default=1)
    bound = 2.0 ** (log2_k + math.log2(longest) - min_bits)
    if bound > cap:
        raise RuntimeError(
            f"{tsv}: a {min_bits:g}-bit alignment of the longest query would get "
            f"E = {bound:.3g} > -e {cap:g}, so mmseqs may have dropped pairs above "
            "the bit floor; raise trizod.dataset.mmseqs.LEAKAGE_SEARCH_EVALUE"
        )
    return bound


def search_both_directions(
    train_fa: Path,
    test_fa: Path,
    out_tsv: Path,
    tmp: Path,
    log: Path,
    min_seq_id: float = LEAKAGE_SEARCH_MIN_SEQ_ID,
) -> list[LeakageHit]:
    """Every alignment of the forward search (training chains as queries,
    hits in ``out_tsv``) and of the reverse search (test chains as queries,
    hits in :func:`reverse_hits_path`), keyed (training, test) either way.
    ``min_seq_id`` is mmseqs' ``--min-seq-id`` (0 for every leakage search;
    a criterion's identity is applied afterwards, :func:`is_leak`)."""
    reverse_tsv = reverse_hits_path(out_tsv)
    tmp = Path(tmp)
    for direction in (FORWARD, REVERSE):  # mmseqs makes one level, not two
        (tmp / direction).mkdir(parents=True, exist_ok=True)
    run(leakage_search_cmd(train_fa, test_fa, out_tsv, tmp / FORWARD, min_seq_id), log)
    run(
        leakage_search_cmd(test_fa, train_fa, reverse_tsv, tmp / REVERSE, min_seq_id),
        log,
        append=True,
    )
    check_evalue_headroom(out_tsv, train_fa)
    check_evalue_headroom(reverse_tsv, test_fa)
    return parse_leakage_hits(out_tsv) + parse_leakage_hits(reverse_tsv, reverse=True)


def strongest_per_pair(hits) -> list[LeakageHit]:
    """One hit per (training, test) pair, the first seen: the forward one when
    both searches report it, and within a search the best-scoring of the
    pair's alignments (mmseqs lists them best first)."""
    best: dict[tuple[str, str], LeakageHit] = {}
    for h in hits:
        best.setdefault((h.train_id, h.test_id), h)
    return list(best.values())


def find_leaks(
    train_fa: Path,
    test_fa: Path,
    out_tsv: Path,
    tmp: Path,
    log: Path,
    rule: str = DEFAULT_LEAKAGE_RULE,
    min_seq_id: float = LEAKAGE_MIN_SEQ_ID,
) -> list[LeakageHit]:
    """Every training/test pair with at least one reported alignment -- the
    best or an alternative one, in either search direction -- that meets the
    leakage criterion of ``rule`` at identity ``min_seq_id``; one hit per pair,
    the first qualifying alignment (forward before reverse, best-scoring
    first). ``out_tsv`` (forward) and its ``_reverse`` sibling keep every
    alignment mmseqs reported (``--min-seq-id 0``), including those below the
    thresholds."""
    check_rule(rule)
    hits = search_both_directions(train_fa, test_fa, out_tsv, tmp, log)
    return strongest_per_pair(
        h for h in hits if is_leak(h, rule, min_seq_id=min_seq_id)
    )


def find_self_leaks(
    fa: Path,
    out_tsv: Path,
    tmp: Path,
    log: Path,
    rule: str = DEFAULT_LEAKAGE_RULE,
    min_seq_id: float = LEAKAGE_MIN_SEQ_ID,
) -> list[LeakageHit]:
    """All-vs-all: every pair of distinct chains of ``fa`` with a reported
    alignment -- the best or an alternative one -- that meets the leakage
    criterion of ``rule`` at identity ``min_seq_id``, one hit per unordered
    pair (the first qualifying alignment). One search covers both directions:
    every chain is a query and a target. In a hit the query is ``train_id``
    and the target ``test_id``. Used to keep the UdonPred validation split
    apart from training (:mod:`trizod.dataset.export_jsonl`) and for the
    homology components of the test set (:mod:`trizod.dataset.finalize`)."""
    check_rule(rule)
    Path(tmp).mkdir(parents=True, exist_ok=True)
    run(leakage_search_cmd(fa, fa, out_tsv, tmp), log)
    check_evalue_headroom(out_tsv, fa)
    pairs: dict[frozenset, LeakageHit] = {}
    for h in parse_leakage_hits(out_tsv):
        if h.train_id != h.test_id and is_leak(h, rule, min_seq_id=min_seq_id):
            pairs.setdefault(frozenset((h.train_id, h.test_id)), h)
    return list(pairs.values())


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
        "search_min_seq_id": LEAKAGE_SEARCH_MIN_SEQ_ID,
        "identity": f">= {LEAKAGE_MIN_SEQ_ID * 100:g} % identity, applied after the "
        "search to every alignment mmseqs reports; the search runs with "
        f"--min-seq-id {LEAKAGE_SEARCH_MIN_SEQ_ID:g}, so that mmseqs also computes "
        "the alternative alignments behind a best alignment below "
        f"{LEAKAGE_MIN_SEQ_ID * 100:g} % (with --min-seq-id "
        f"{LEAKAGE_MIN_SEQ_ID:g} such a pair was dropped with all its alternatives)",
        "min_coverage": LEAKAGE_MIN_COV,
        "coverage_of": LEAKAGE_RULE_COVERAGE_OF[rule],
        "coverage_definition": "aligned span / chain length, gaps included",
        "min_bits": LEAKAGE_MIN_BITS,
        "significance": f"bit score >= {LEAKAGE_MIN_BITS:g} (independent of the "
        "search direction and the database size); mmseqs -e "
        f"{LEAKAGE_SEARCH_EVALUE:g} only so that every such alignment is reported",
        "directions": "both: training chains as queries against the test chains, "
        "and test chains as queries against the training chains; a pair counts "
        "when either search meets the criterion",
        "alignments": "every reported alignment of a pair, the best one and up to "
        f"{LEAKAGE_ALT_ALI} alternative ones (mmseqs --alt-ali {LEAKAGE_ALT_ALI}); "
        "a pair counts when any of them meets the criterion",
        "alt_alignments": LEAKAGE_ALT_ALI,
        "stage2_search": "mmseqs easy-search training (query) vs test (target) and "
        "test (query) vs training (target) " + " ".join(leakage_search_options()),
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
