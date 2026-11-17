"""Stage 1 of the redundancy step does not depend on how its inputs are ordered
or grouped into files.

Stage 1 clusters the training superset together with every leakage target and
removes every training chain that shares a cluster with a target. mmseqs'
greedy set cover breaks ties by input order, so the removed set used to depend
on the order of the target files: the 2026-11-17b dry run moved four test
chains from ``TriZOD_test_set.fasta`` to ``TriZOD_test_retired.fasta``, and
6514_1_1_1 (no leak under the rule) came back into training while eight other
chains flipped the other way. Now the targets are merged and sorted by
sequence ID before they are written, and the stage-1 input lists the training
pool in ID order too, so the clustering input -- and the removed set -- is a
function of the records alone.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import pytest

from trizod.dataset import redundancy
from trizod.io.fasta import read_fasta, write_fasta

requires_mmseqs = pytest.mark.skipif(
    shutil.which("mmseqs") is None, reason="mmseqs binary not found in PATH"
)

AA = "ACDEFGHIKLMNPQRSTVWY"


def _rand(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(AA) for _ in range(n))


def _records(seed: int = 7):
    rng = random.Random(seed)
    targets = {
        "19482_1_1_1": _rand(rng, 91),  # CheZOD117 as released
        "hmbdi": _rand(rng, 82),  # an original CheZOD117 name
        "6078": _rand(rng, 70),
        "100_1_1_1": _rand(rng, 120),  # TriZOD test
        "52047_1_1_1": _rand(rng, 103),  # retired in one grouping, test in another
        "9_1_1_1": _rand(rng, 60),
    }
    training = {f"{n}_1_1_1": _rand(rng, 90) for n in (12, 3, 200, 45, 7, 1000)}
    return targets, training


def _write_groups(tmp: Path, targets: dict, groups: list[list[str]]) -> list[Path]:
    paths = []
    for i, ids in enumerate(groups):
        path = tmp / f"source_{i}.fasta"
        write_fasta({k: targets[k] for k in ids}, path)
        paths.append(path)
    return paths


GROUPINGS = [
    # released, original, test, retired: as the build lays them out
    [["19482_1_1_1"], ["hmbdi", "6078"], ["100_1_1_1", "52047_1_1_1", "9_1_1_1"], []],
    # one chain moved from the test file to the retired file
    [["19482_1_1_1"], ["hmbdi", "6078"], ["100_1_1_1", "9_1_1_1"], ["52047_1_1_1"]],
    # everything in one file, in another order
    [["9_1_1_1", "6078", "52047_1_1_1", "hmbdi", "100_1_1_1", "19482_1_1_1"]],
    # the files in another order
    [["100_1_1_1", "9_1_1_1"], ["52047_1_1_1"], ["6078", "hmbdi"], ["19482_1_1_1"]],
]


def test_the_combined_target_set_is_sorted_by_id(tmp_path):
    targets, _ = _records()
    sources = _write_groups(tmp_path, targets, GROUPINGS[0])
    out = tmp_path / "combined.fasta"
    redundancy.build_combined_testset(sources, out)
    assert list(read_fasta(out)) == [
        "9_1_1_1",
        "100_1_1_1",
        "6078",
        "19482_1_1_1",
        "52047_1_1_1",
        "hmbdi",
    ]
    assert read_fasta(out) == targets


@pytest.mark.parametrize("grouping", GROUPINGS[1:])
def test_the_combined_target_set_does_not_depend_on_its_files(tmp_path, grouping):
    targets, _ = _records()
    ref_dir, alt_dir = tmp_path / "ref", tmp_path / "alt"
    ref_dir.mkdir()
    alt_dir.mkdir()
    ref = ref_dir / "combined.fasta"
    alt = alt_dir / "combined.fasta"
    redundancy.build_combined_testset(
        _write_groups(ref_dir, targets, GROUPINGS[0]), ref
    )
    redundancy.build_combined_testset(_write_groups(alt_dir, targets, grouping), alt)
    assert alt.read_bytes() == ref.read_bytes()


def test_a_target_id_in_two_files_is_still_an_error(tmp_path):
    targets, _ = _records()
    sources = _write_groups(tmp_path, targets, [["9_1_1_1"], ["9_1_1_1", "6078"]])
    with pytest.raises(SystemExit, match="collision"):
        redundancy.build_combined_testset(sources, tmp_path / "combined.fasta")


def test_the_stage1_input_is_canonical(tmp_path):
    targets, training = _records()
    inputs = []
    for i, (train_order, grouping) in enumerate(
        [(list(training), GROUPINGS[0]), (sorted(training, reverse=True), GROUPINGS[3])]
    ):
        d = tmp_path / str(i)
        d.mkdir()
        train_fa = d / "unfiltered.fasta"
        # the build's training FASTAs carry attributes after the ID
        train_fa.write_text(
            "".join(f">{k} tier=strict quality=1\n{training[k]}\n" for k in train_order)
        )
        combined = d / "combined.fasta"
        redundancy.build_combined_testset(_write_groups(d, targets, grouping), combined)
        inputs.append(
            redundancy.write_stage1_input(train_fa, combined, d / "stage1_in.fasta")
        )
    assert inputs[0].read_bytes() == inputs[1].read_bytes()
    ids = list(read_fasta(inputs[0]))
    n = len(training)
    assert ids[:n] == [
        "3_1_1_1",
        "7_1_1_1",
        "12_1_1_1",
        "45_1_1_1",
        "200_1_1_1",
        "1000_1_1_1",
    ]
    assert all(i.startswith(redundancy.TEST_PREFIX) for i in ids[n:])
    assert [i.removeprefix(redundancy.TEST_PREFIX) for i in ids[n:]] == list(
        read_fasta(tmp_path / "0" / "combined.fasta")
    )


# --------------------------------------------------------------------------- #
# the clustering itself (mmseqs)
# --------------------------------------------------------------------------- #


def _mutate(seq: str, positions, shift: int) -> str:
    return "".join(
        AA[(AA.index(c) + shift) % len(AA)] if i in positions else c
        for i, c in enumerate(seq)
    )


def _path_fixture(seed: int = 1):
    """A path of homologues T - X - Y - Z (each ~55 % identical to the next,
    the ends unrelated): stage 1 removes X and Y when the set cover picks X as
    a representative, nothing when it picks Y -- a tie that input order
    breaks."""
    rng = random.Random(seed)
    t = _rand(rng, 100)
    idx = list(range(100))
    rng.shuffle(idx)
    a, b = set(idx[:45]), set(idx[45:90])
    x = _mutate(t, a, 7)
    y = _mutate(x, b, 7)
    z = _mutate(y, a, 3)
    return {"30_1_1_1": t}, {"40_1_1_1": x, "50_1_1_1": y, "60_1_1_1": z}


@requires_mmseqs
def test_stage1_removes_the_same_chains_however_the_inputs_are_ordered(tmp_path):
    targets, training = _path_fixture()
    # an unrelated target in a second file, so the grouping can change too
    targets2 = {"8_1_1_1": _rand(random.Random(11), 80)}
    removed = []
    for i, (train_order, files) in enumerate(
        [
            (list(training), [targets, targets2]),
            (list(reversed(training)), [targets2, targets]),
            (list(training)[1:] + list(training)[:1], [{**targets2, **targets}]),
        ]
    ):
        d = tmp_path / str(i)
        (d / "tmp").mkdir(parents=True)
        train_fa = d / "unfiltered.fasta"
        write_fasta({k: training[k] for k in train_order}, train_fa)
        sources = []
        for j, recs in enumerate(files):
            sources.append(d / f"targets_{j}.fasta")
            write_fasta(recs, sources[-1])
        combined = d / "combined.fasta"
        redundancy.build_combined_testset(sources, combined)
        removed.append(
            redundancy.stage1_cluster_member_removal(
                train_fa, combined, d / "tmp", d, d / "stage1.log"
            )
        )
    assert removed[0] == removed[1] == removed[2]


def _cluster_removed(records: list[tuple[str, str]], work: Path) -> set[str]:
    """Stage 1's clustering on ``records`` exactly in the given order (no
    canonical sorting): the training IDs sharing a cluster with a target."""
    work.mkdir(parents=True)
    fa = work / "in.fasta"
    fa.write_text("".join(f">{k}\n{s}\n" for k, s in records))
    redundancy.run(
        [
            "mmseqs",
            "easy-cluster",
            str(fa),
            str(work / "c"),
            str(work / "tmp"),
            "--min-seq-id",
            "0.3",
            "-c",
            "0.8",
            *redundancy.COMMON,
        ],
        work / "log",
    )
    removed = set()
    for rep, members in redundancy.cluster_tsv_groups(work / "c_cluster.tsv").items():
        group = set(members) | {rep}
        if any(m.startswith(redundancy.TEST_PREFIX) for m in group):
            removed |= {m for m in group if not m.startswith(redundancy.TEST_PREFIX)}
    return removed


@requires_mmseqs
def test_the_fixture_is_order_sensitive_without_the_canonical_order(tmp_path):
    """Control: the same four records, two input orders, two removed sets --
    the tie the canonical order settles."""
    targets, training = _path_fixture()
    (tid, tseq), x, y, z = (
        next(iter(targets.items())),
        "40_1_1_1",
        "50_1_1_1",
        "60_1_1_1",
    )
    t = (redundancy.TEST_PREFIX + tid, tseq)
    first = _cluster_removed(
        [t, (x, training[x]), (y, training[y]), (z, training[z])], tmp_path / "a"
    )
    second = _cluster_removed(
        [t, (y, training[y]), (x, training[x]), (z, training[z])], tmp_path / "b"
    )
    assert first != second
