"""Per-chain overlap with the CheZOD sets (release audit E23, 2026-11-17).

Every released chain carries three flags, ``chezod117_overlap``,
``chezod1325_overlap`` and ``chezod1174_overlap``: whether its sequence meets
the leakage criterion (>= 30 % identity over >= 80 % of either chain, >= 27
bits, either search direction) against any sequence of the set. They let a
benchmark against SETH, ODiNPred or ADOPT use the subset clean of the
predictor's own training set.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import pytest

from trizod.dataset import chezod_overlap as co
from trizod.io.fasta import write_fasta

requires_mmseqs = pytest.mark.skipif(
    shutil.which("mmseqs") is None, reason="mmseqs binary not found in PATH"
)

AA = "ACDEFGHIKLMNPQRSTVWY"


def _rand(rng, n):
    return "".join(rng.choice(AA) for _ in range(n))


def test_the_three_sets_and_their_columns():
    assert co.CHEZOD_SETS == ("chezod117", "chezod1325", "chezod1174")
    assert co.overlap_column("chezod1325") == "chezod1325_overlap"


def _write_sets(root: Path, chezod117, original, c1325, c1174):
    ext = root / "data" / "external"
    (ext / "chezod117").mkdir(parents=True)
    write_fasta(chezod117, ext / "chezod117" / "CheZOD117_test_set.fasta")
    seth = ext / "chezod" / "seth_github"
    seth.mkdir(parents=True)
    write_fasta(original, seth / "CheZOD117_test_set_sequences.fasta")
    write_fasta(c1174, seth / "CheZOD1174_training_set_sequences.fasta")
    nmr = ext / "chezod" / "protein_nmr_1325"
    nmr.mkdir(parents=True)
    (nmr / "allseqs1325.txt").write_text(
        "".join(f"{k} {v}\n" for k, v in c1325.items())
    )


def test_the_sets_are_read_from_the_external_files(tmp_path):
    _write_sets(
        tmp_path,
        {"19482_1_1_1": "MKV" * 20},
        {"hmbdi": "GHC" * 20},
        {"26672": "QSY" * 20},
        {"dss1": "PEN" * 20},
    )
    sets = co.load_chezod_sets(root=tmp_path)
    # CheZOD117 is the released chains plus the original sequences
    assert sets["chezod117"] == {"19482_1_1_1": "MKV" * 20, "hmbdi": "GHC" * 20}
    assert sets["chezod1325"] == {"26672": "QSY" * 20}
    assert sets["chezod1174"] == {"dss1": "PEN" * 20}


def test_a_missing_set_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="CheZOD"):
        co.load_chezod_sets(root=tmp_path)


@requires_mmseqs
def test_overlapping_chains_are_flagged_identical_sequences_alike(tmp_path):
    rng = random.Random(1325)
    chezod = _rand(rng, 90)
    chains = {
        "1_1_1_1": _rand(rng, 25) + chezod + _rand(rng, 30),  # contains it
        "2_1_1_1": chezod[5:70],  # a fragment of it
        "3_1_1_1": _rand(rng, 120),  # unrelated
    }
    chains["4_1_1_1"] = chains["2_1_1_1"]  # same sequence, another chain
    got = co.overlapping_chains(chains, {"c1": chezod}, tmp_path / "work")
    assert got == {"1_1_1_1", "2_1_1_1", "4_1_1_1"}


@requires_mmseqs
def test_overlaps_of_every_set(tmp_path):
    rng = random.Random(117)
    a, b = _rand(rng, 80), _rand(rng, 80)
    chains = {"1_1_1_1": a, "2_1_1_1": b, "3_1_1_1": _rand(rng, 80)}
    sets = {"chezod117": {"x": a}, "chezod1325": {"y": a, "z": b}, "chezod1174": {}}
    got = co.chezod_overlaps(chains, sets, tmp_path / "work")
    assert got == {
        "chezod117": {"1_1_1_1"},
        "chezod1325": {"1_1_1_1", "2_1_1_1"},
        "chezod1174": set(),
    }
