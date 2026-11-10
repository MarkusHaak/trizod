"""Every released label must be the selecting tier's label (audit issue I-01).

The four per-tier ``scores.json`` files are NOT interchangeable. Tolerant is the
only tier that combines a finite ``--max-offset`` (3 sigma) with
``--reject-shift-type-only``: an over-limit atom type is masked out of the score
there and its ``off_<a>_sigma`` / ``total_off_<a>_ppm`` are null, whereas the
unfiltered run applies the offset and scores the type. In release 2026-10 that
makes 1,073 of 15,073 tolerant records differ from the unfiltered run (478 of
them training representatives); moderate and strict reject such a chain outright
and match unfiltered exactly.

The Parquet used to read every row from ``scores/unfiltered/scores.json`` on the
premise that the tiers were "byte-identical for a given chain", so the release
contradicted its own per-tier ``scores.json`` and the UdonPred deploy FASTA.
The rule now is: a chain's labels AND offsets come from ONE record, that of the
tier that selected it -- ``train_tier`` for a training representative, otherwise
the strictest tier whose ``scores.json`` holds it.

:func:`label_mismatches` is the reusable check. It re-derives each chain's
selecting tier from the bundle files on its own (it does not call the builder's
rule), so it can gate any built artefact::

    TRIZOD_CHECK_PARQUET=<trizod_dataset.parquet> \\
    TRIZOD_CHECK_SHIFTS=<trizod_shifts.parquet> \\
    TRIZOD_RELEASE_BUNDLE=<release_bundle/trizod-dataset-YYYY-MM> \\
        uv run --with pyarrow pytest tests/test_release_label_consistency.py -rs
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from trizod import paths
from trizod.constants import BACKBONE_ATOMS
from trizod.dataset import deploy_fasta
from trizod.offsets import (
    lacs_off_ppm_col,
    off_sigma_col,
    total_off_ppm_col,
    total_offset_ppm,
)
from trizod.shifts import backbone_slot, chain_offsets

SCRIPT = paths.ROOT / "scripts" / "build_parquet_dataset.py"
TIERS = ("unfiltered", "tolerant", "moderate", "strict")  # loosest -> strictest


def _load_builder():
    """``scripts/build_parquet_dataset.py`` as a module (it is not installed)."""
    if "build_parquet_dataset" in sys.modules:
        return sys.modules["build_parquet_dataset"]
    spec = importlib.util.spec_from_file_location("build_parquet_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass resolves annotations through it
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


# --------------------------------------------------------------------------- #
# the reusable check -- deliberately independent of the builder's own rule
# --------------------------------------------------------------------------- #


def _jsonl(path: Path) -> dict[str, dict]:
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["ID"]] = rec
    return out


def _fasta_ids(path: Path) -> set[str]:
    with open(path) as fh:
        return {ln[1:].split()[0] for ln in fh if ln.startswith(">")}


def selecting_records(bundle: Path) -> dict[str, tuple[str, dict]]:
    """``chain ID -> (selecting tier, that tier's scores.json record)``.

    A training representative is selected by its ``train_tier`` (the strictest
    tier whose ``train_<tier>_best.fasta`` lists it); any other chain by the
    strictest tier whose ``scores.json`` holds it.
    """
    scores = {t: _jsonl(bundle / "scores" / t / "scores.json") for t in TIERS}
    reps = {
        t: _fasta_ids(bundle / "train" / t / f"train_{t}_best.fasta") for t in TIERS
    }
    out = {}
    for cid in scores["unfiltered"]:
        trained = [t for t in TIERS if cid in reps[t]]
        held = [t for t in TIERS if cid in scores[t]]
        tier = trained[-1] if trained else held[-1]
        out[cid] = (tier, scores[tier][cid])
    return out


def _null(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _same(a, b, tol=1e-6) -> bool:
    """Equal as published: null == NaN == None; numbers to float32 precision."""
    if _null(a) or _null(b):
        return _null(a) and _null(b)
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def _same_list(got, want) -> bool:
    got, want = list(got or []), list(want or [])
    return len(got) == len(want) and all(_same(a, b) for a, b in zip(got, want))


def _as_int(x):
    return None if _null(x) else int(x)


def row_diffs(table: dict[str, list], i: int, rec: dict) -> list[str]:
    """The label/offset fields in which row ``i`` of ``table`` is not ``rec``."""
    diffs = []
    g = rec.get("gscores") or []
    if not _same_list(table["gscores"][i], g):
        diffs.append("gscores")
    if not _same_list(table["zscores"][i], rec.get("zscores") or []):
        diffs.append("zscores")
    k = [0 if _null(x) else int(x) for x in rec.get("k") or []]
    if list(table["k"][i] or []) != k:
        diffs.append("k")
    if list(table["mask"][i] or []) != [x is not None for x in g]:
        diffs.append("mask")
    for col in ("bbshift_types_post", "bbshift_positions_post"):
        if _as_int(table[col][i]) != _as_int(rec.get(col)):
            diffs.append(col)
    offs = chain_offsets(rec)
    for atom in BACKBONE_ATOMS:
        for col, want in (
            (off_sigma_col(atom), offs.sigma[atom]),
            (lacs_off_ppm_col(atom), offs.lacs_ppm[atom]),
            (total_off_ppm_col(atom), offs.total_ppm[atom]),
        ):
            if not _same(table[col][i], want):
                diffs.append(col)
    return diffs


def label_mismatches(bundle: Path, table: dict[str, list]) -> dict[str, set[str]]:
    """Per field, the chains whose published value is not the selecting record's.

    ``table`` is the main Parquet as ``name -> list of values`` (a
    ``pyarrow.Table.to_pydict()`` or the builder's ``collect_columns`` dict).
    An empty result means every row is its selecting tier's record.
    """
    expected = selecting_records(bundle)
    bad: dict[str, set[str]] = defaultdict(set)
    ids = table["id"]
    for cid in set(ids) ^ set(expected):
        bad["rows"].add(cid)
    if "label_tier_source" not in table:
        bad["label_tier_source"].add("<column missing>")
    for i, cid in enumerate(ids):
        if cid not in expected:
            continue
        tier, rec = expected[cid]
        if "label_tier_source" in table and table["label_tier_source"][i] != tier:
            bad["label_tier_source"].add(cid)
        for field in row_diffs(table, i, rec):
            bad[field].add(cid)
    return dict(bad)


def deploy_mismatches(deploy_path: Path, table: dict[str, list]) -> set[str]:
    """Deploy-FASTA records whose TARGET/MASK/sequence disagree with the table."""
    row = {cid: i for i, cid in enumerate(table["id"])}
    bad = set()
    lines = deploy_path.read_text().splitlines()
    for header, seq in zip(lines[::2], lines[1::2]):
        tokens = dict(t.split("=", 1) for t in header[1:].split()[1:])
        cid = header[1:].split()[0]
        i = row.get(cid)
        if i is None or table["sequence"][i] != seq:
            bad.add(cid)
            continue
        target = [float(x) for x in tokens["TARGET"].split(";")]
        mask = [c == "1" for c in tokens["MASK"]]
        want = table["gscores"][i]
        if mask != list(table["mask"][i]) or not all(
            not m or _same(t, w, tol=1e-4) for t, w, m in zip(target, want, mask)
        ):
            bad.add(cid)
    return bad


def shift_offset_mismatches(bundle: Path, shifts_parquet: Path) -> set[str]:
    """Chains whose backbone ``offset_applied_ppm`` is not the selecting record's
    ``total_off_<slot>_ppm`` (null where that record has none)."""
    import numpy as np
    import pandas as pd

    pq = pytest.importorskip("pyarrow.parquet")
    expected = selecting_records(bundle)
    df = (
        pq.read_table(
            shifts_parquet,
            columns=["id", "comp_id", "atom_id", "is_backbone", "offset_applied_ppm"],
            filters=[("is_backbone", "=", True)],
            read_dictionary=["id", "comp_id", "atom_id"],
        )
        .to_pandas()
        .drop(columns="is_backbone")
        .astype({"id": str, "comp_id": str, "atom_id": str})
        .drop_duplicates()  # one offset per (chain, slot): ~10 rows per chain
    )
    pairs = df[["comp_id", "atom_id"]].drop_duplicates()
    pairs["slot"] = [backbone_slot(c, a) for c, a in zip(pairs.comp_id, pairs.atom_id)]
    df = df.merge(pairs, on=["comp_id", "atom_id"], how="left")
    want = pd.DataFrame(
        [
            (cid, atom, chain_offsets(rec).total_ppm[atom])
            for cid, (_, rec) in expected.items()
            for atom in BACKBONE_ATOMS
        ],
        columns=["id", "slot", "want"],
    )
    want["want"] = want["want"].astype(float)
    df = df.merge(want, on=["id", "slot"], how="left")
    got, exp = df["offset_applied_ppm"].astype(float), df["want"]
    ok = (got.isna() & exp.isna()) | np.isclose(got, exp, rtol=0, atol=1e-9)
    return set(df.loc[~ok, "id"])


def _read_table(path: Path) -> dict[str, list]:
    pq = pytest.importorskip("pyarrow.parquet")
    return pq.read_table(path).to_pydict()


# --------------------------------------------------------------------------- #
# synthetic bundle: a tolerant record that differs from the unfiltered one
# --------------------------------------------------------------------------- #

SEQ = "AAAA"
LACS = {a: 0.1 * (i + 1) for i, a in enumerate(BACKBONE_ATOMS)}
#: Unfiltered applies an over-limit C offset; tolerant masks C and nulls it.
SIGMA_UNF = {**dict.fromkeys(BACKBONE_ATOMS, 0.5), "C": 4.2}
SIGMA_TOL = {**SIGMA_UNF, "C": None}

TOL_TRAIN = "100_1_1_1"  # training representative at tolerant (not moderate)
UNF_TRAIN = "200_1_1_1"  # training representative at unfiltered only
UNF_TRAIN_IN_TOL = "300_1_1_1"  # unfiltered rep, ALSO held (unselected) at tolerant
EXCLUDED_TOL = "400_1_1_1"  # no rep/pool anywhere, strictest scores tier tolerant
MOD_TRAIN = "500_1_1_1"  # training representative at moderate
TEST = "600_1_1_1"  # TriZOD test chain, held up to tolerant
REDUNDANT = "700_1_1_1"  # tolerant pool member, not a representative
UNF_ONLY = "800_1_1_1"  # held by unfiltered only

EXPECTED_SOURCE = {
    TOL_TRAIN: "tolerant",
    UNF_TRAIN: "unfiltered",
    UNF_TRAIN_IN_TOL: "unfiltered",
    EXCLUDED_TOL: "tolerant",
    MOD_TRAIN: "moderate",
    TEST: "tolerant",
    REDUNDANT: "tolerant",
    UNF_ONLY: "unfiltered",
}


def _record(cid, gscores, sigma, types_post=7):
    rec = {
        "ID": cid,
        "entryID": cid.split("_", 1)[0],
        "stID": "1",
        "entity_assemID": "1",
        "entityID": "1",
        "seq": SEQ,
        "gscores": gscores,
        "zscores": [None if g is None else round(10 * g - 3, 4) for g in gscores],
        "k": [None if g is None else 3 for g in gscores],
        "bbshift_types_post": types_post,
        "bbshift_positions_post": sum(g is not None for g in gscores),
    }
    for atom in BACKBONE_ATOMS:
        total = total_offset_ppm(atom, LACS[atom], sigma[atom])
        rec[off_sigma_col(atom)] = sigma[atom]
        rec[lacs_off_ppm_col(atom)] = LACS[atom]
        rec[total_off_ppm_col(atom)] = None if math.isnan(total) else total
    return rec


def _unf(cid, g):
    return _record(cid, g, SIGMA_UNF)


def _tol(cid, g):
    return _record(cid, g, SIGMA_TOL, types_post=6)


SCORES = {
    "unfiltered": [
        _unf(TOL_TRAIN, [0.9, 0.8, 0.7, None]),
        _unf(UNF_TRAIN, [0.1, 0.2, 0.3, 0.4]),
        _unf(UNF_TRAIN_IN_TOL, [0.15, 0.25, 0.35, None]),
        _unf(EXCLUDED_TOL, [0.95, 0.85, 0.75, 0.65]),
        _unf(MOD_TRAIN, [0.5, 0.5, 0.5, 0.5]),
        _unf(TEST, [0.33, 0.44, 0.55, None]),
        _unf(REDUNDANT, [0.11, 0.22, 0.33, 0.44]),
        _unf(UNF_ONLY, [0.01, 0.02, None, None]),
    ],
    "tolerant": [
        _tol(TOL_TRAIN, [0.6, 0.5, 0.4, None]),
        _tol(UNF_TRAIN_IN_TOL, [0.12, 0.13, None, None]),
        _tol(EXCLUDED_TOL, [0.45, 0.35, None, 0.25]),
        _unf(MOD_TRAIN, [0.5, 0.5, 0.5, 0.5]),
        _tol(TEST, [0.3, 0.2, 0.1, None]),
        _tol(REDUNDANT, [0.61, 0.62, 0.63, None]),
    ],
    "moderate": [_unf(MOD_TRAIN, [0.5, 0.5, 0.5, 0.5])],
    "strict": [],
}
REPS = {
    "unfiltered": [TOL_TRAIN, UNF_TRAIN, UNF_TRAIN_IN_TOL, MOD_TRAIN],
    "tolerant": [TOL_TRAIN, MOD_TRAIN],
    "moderate": [MOD_TRAIN],
    "strict": [],
}
POOLS = {
    "unfiltered": [*REPS["unfiltered"], REDUNDANT],
    "tolerant": [*REPS["tolerant"], REDUNDANT],
    "moderate": REPS["moderate"],
    "strict": [],
}


def _fasta(ids):
    return "".join(f">{cid}\n{SEQ}\n" for cid in ids)


def write_bundle(root: Path, scores=SCORES, reps=REPS, pools=POOLS) -> Path:
    """A release bundle in the ``package_release`` layout."""
    for tier in TIERS:
        sdir = root / "scores" / tier
        sdir.mkdir(parents=True)
        (sdir / "scores.json").write_text(
            "".join(json.dumps(r) + "\n" for r in scores[tier])
        )
        tdir = root / "train" / tier
        tdir.mkdir(parents=True)
        (tdir / f"train_{tier}_best.fasta").write_text(_fasta(reps[tier]))
        rows = "".join(f"{m}\t{m}\t{m}\t1.0\n" for m in pools[tier])
        (tdir / "clusters_best.tsv").write_text(
            "cluster_repr\tmember\tbest_repr\tmember_quality\n" + rows
        )
    (root / "test").mkdir(parents=True)
    (root / "test" / "CheZOD117_test_set.fasta").write_text("")
    (root / "test" / "TriZOD_test_set.fasta").write_text(
        f">{TEST} label_tier=tolerant pinned_id={TEST}\n{SEQ}\n"
    )
    return root


@pytest.fixture
def bundle(tmp_path):
    return write_bundle(tmp_path / "bundle")


#: Chains whose selecting record is tolerant AND differs from the unfiltered one.
TOLERANT_DIFFERS = {TOL_TRAIN, EXCLUDED_TOL, TEST, REDUNDANT}


def _stale(tmp_path):
    """The bundle as the pre-fix builder effectively read it: every tier's
    record replaced by the unfiltered one, membership unchanged."""
    unf = {r["ID"]: r for r in SCORES["unfiltered"]}
    as_before = {t: [unf[r["ID"]] for r in SCORES[t]] for t in TIERS}
    return write_bundle(tmp_path / "stale", scores=as_before)


def _row(cols, cid):
    i = cols["id"].index(cid)
    return {name: values[i] for name, values in cols.items()}


def _by_id(records):
    return {r["ID"]: r for r in records}


# --------------------------------------------------------------------------- #
# the main table
# --------------------------------------------------------------------------- #


def test_each_row_names_the_tier_its_labels_came_from(bundle):
    cols = builder.collect_columns(bundle, None)
    got = dict(zip(cols["id"], cols["label_tier_source"]))
    assert got == EXPECTED_SOURCE


def test_a_tolerant_training_chain_ships_its_tolerant_labels_and_offsets(bundle):
    """THE I-01 case: train_tier tolerant, C masked there, applied at unfiltered."""
    row = _row(builder.collect_columns(bundle, None), TOL_TRAIN)
    want = _by_id(SCORES["tolerant"])[TOL_TRAIN]
    assert (row["split"], row["train_tier"]) == ("train", "tolerant")
    assert _same_list(row["gscores"], want["gscores"])
    assert _same_list(row["zscores"], want["zscores"])
    assert row["k"] == [3, 3, 3, 0]
    assert row["mask"] == [True, True, True, False]
    assert row["n_scored"] == 3
    assert row["bbshift_types_post"] == 6
    # the masked type's offset is null, never the unfiltered 4.2 sigma
    assert row[off_sigma_col("C")] is None
    assert row[total_off_ppm_col("C")] is None
    assert row[lacs_off_ppm_col("C")] == pytest.approx(LACS["C"])
    assert row[off_sigma_col("CA")] == pytest.approx(0.5)


def test_a_chain_only_the_unfiltered_tier_selects_keeps_unfiltered_values(bundle):
    cols = builder.collect_columns(bundle, None)
    for cid in (UNF_TRAIN, UNF_ONLY):
        row = _row(cols, cid)
        want = _by_id(SCORES["unfiltered"])[cid]
        assert _same_list(row["gscores"], want["gscores"]), cid
        assert row[off_sigma_col("C")] == pytest.approx(4.2), cid


def test_a_training_chain_takes_its_train_tier_even_when_a_stricter_tier_holds_it(
    bundle,
):
    """Pins the rule's precedence: a chain trained at unfiltered only, but also
    present (unselected) in tolerant's scores.json, ships its unfiltered record --
    the record of the training set it is actually part of."""
    row = _row(builder.collect_columns(bundle, None), UNF_TRAIN_IN_TOL)
    want = _by_id(SCORES["unfiltered"])[UNF_TRAIN_IN_TOL]
    assert row["train_tier"] == "unfiltered"
    assert _same_list(row["gscores"], want["gscores"])
    assert row[off_sigma_col("C")] == pytest.approx(4.2)


@pytest.mark.parametrize("cid", [EXCLUDED_TOL, TEST, REDUNDANT])
def test_a_non_training_chain_takes_the_strictest_tier_holding_it(bundle, cid):
    row = _row(builder.collect_columns(bundle, None), cid)
    want = _by_id(SCORES["tolerant"])[cid]
    assert row["label_tier_source"] == "tolerant"
    assert _same_list(row["gscores"], want["gscores"])
    assert row[total_off_ppm_col("C")] is None


def test_the_built_parquet_passes_the_check_and_declares_the_rule(bundle, tmp_path):
    pytest.importorskip("pyarrow")
    out = tmp_path / "trizod_dataset.parquet"
    builder.build(bundle, out, None)
    assert label_mismatches(bundle, _read_table(out)) == {}
    pq = pytest.importorskip("pyarrow.parquet")
    rule = pq.read_schema(out).metadata[b"label_source"].decode()
    assert "train_tier" in rule and "strictest tier" in rule


def test_the_check_catches_unfiltered_labels_on_tolerant_chains(bundle, tmp_path):
    """The pre-fix builder read every row from the unfiltered run. Simulate it
    with a bundle whose stricter tiers hold the unfiltered records, and the
    check must flag exactly the chains whose tolerant record differs."""
    bad = label_mismatches(bundle, builder.collect_columns(_stale(tmp_path), None))
    assert bad["gscores"] == TOLERANT_DIFFERS
    assert bad[off_sigma_col("C")] == TOLERANT_DIFFERS
    assert bad["bbshift_types_post"] == TOLERANT_DIFFERS
    assert "label_tier_source" not in bad  # membership was never the problem


def test_a_missing_tier_scores_file_is_an_error(bundle):
    """Falling back to the unfiltered record would silently restore I-01."""
    (bundle / "scores" / "tolerant" / "scores.json").unlink()
    with pytest.raises(SystemExit, match="tolerant"):
        builder.collect_columns(bundle, None)


def test_a_training_chain_absent_from_its_tier_scores_is_an_error(tmp_path):
    scores = {**SCORES, "moderate": []}
    broken = write_bundle(tmp_path / "broken", scores=scores)
    with pytest.raises(SystemExit, match=MOD_TRAIN):
        builder.collect_columns(broken, None)


def test_a_tier_record_outside_the_unfiltered_superset_is_an_error(tmp_path):
    stray = _tol("999_1_1_1", [0.5, 0.5, 0.5, 0.5])
    scores = {**SCORES, "tolerant": [*SCORES["tolerant"], stray]}
    broken = write_bundle(tmp_path / "broken", scores=scores)
    with pytest.raises(SystemExit, match="999_1_1_1"):
        builder.collect_columns(broken, None)


# --------------------------------------------------------------------------- #
# the shift table takes its offsets from the same record
# --------------------------------------------------------------------------- #

#: Two ALA residues with a backbone C and CA each, in ShiftTable.shifts layout.
TOY_SHIFTS = [
    ("1", "1", "1", "ALA", "C", "C", "176.00", "0.10", "1"),
    ("1", "1", "1", "ALA", "CA", "C", "52.30", "0.10", "1"),
    ("1", "1", "2", "ALA", "C", "C", "177.10", "0.10", "1"),
    ("1", "1", "2", "ALA", "CA", "C", "52.90", "0.10", "1"),
]


@pytest.fixture
def toy_shift_frames(monkeypatch):
    """Replace the pickle reader with a toy frame per requested chain."""
    pytest.importorskip("pyarrow")
    from trizod.bmrb.bmrb import get_deposited_shifts

    def fake(ids, pkl_dir, *args, **kwargs):
        for cid in sorted(ids):
            yield cid, get_deposited_shifts(TOY_SHIFTS, SEQ)

    monkeypatch.setattr(builder, "iter_shift_frames", fake)


def test_shift_parquet_takes_offsets_from_the_selecting_record(
    bundle, tmp_path, toy_shift_frames
):
    pq = pytest.importorskip("pyarrow.parquet")
    out = tmp_path / "trizod_shifts.parquet"
    builder.build_shifts(bundle, out, tmp_path)
    got = pq.read_table(out).to_pydict()
    rows = {
        (cid, sid, atom): (applied, corrected, src)
        for cid, sid, atom, applied, corrected, src in zip(
            got["id"],
            got["seq_id"],
            got["atom_id"],
            got["offset_applied_ppm"],
            got["val_corrected_ppm"],
            got["offset_source"],
        )
    }
    # tolerant masked C: no offset is applied and no corrected value published
    assert rows[(TOL_TRAIN, 1, "C")] == (None, None, "none")
    # ...where the unfiltered run would have moved every C by 4.2 sigma
    unf_c = total_offset_ppm("C", LACS["C"], 4.2)
    assert rows[(UNF_TRAIN, 1, "C")][0] == pytest.approx(unf_c)
    assert rows[(UNF_TRAIN_IN_TOL, 1, "C")][0] == pytest.approx(unf_c)
    # an unmasked type keeps its offset at either tier
    ca = total_offset_ppm("CA", LACS["CA"], 0.5)
    assert rows[(TOL_TRAIN, 2, "CA")][0] == pytest.approx(ca)
    assert shift_offset_mismatches(bundle, out) == set()


def test_the_shift_check_catches_an_unfiltered_offset_on_a_masked_type(
    bundle, tmp_path, toy_shift_frames
):
    out = tmp_path / "stale_shifts.parquet"
    builder.build_shifts(_stale(tmp_path), out, tmp_path)
    assert shift_offset_mismatches(bundle, out) == TOLERANT_DIFFERS


# --------------------------------------------------------------------------- #
# the deploy FASTA reads labels by the same rule
# --------------------------------------------------------------------------- #


def _deploy(bundle, tmp_path, tier, name):
    out = tmp_path / name
    deploy_fasta.main(
        [
            "--tier",
            tier,
            "--reps-dir",
            str(bundle / "train"),
            "--scored-dir",
            str(bundle / "scores"),
            "--test-fasta",
            str(bundle / "test" / "TriZOD_test_set.fasta"),
            "--root",
            str(tmp_path),
            "--out",
            str(out),
        ]
    )
    return out


def _targets(path):
    lines = path.read_text().splitlines()
    out = {}
    for header in lines[::2]:
        tokens = header[1:].split()
        attrs = dict(t.split("=", 1) for t in tokens[1:])
        out[tokens[0]] = (attrs["SET"], attrs["TARGET"])
    return out


@pytest.mark.parametrize("tier", ["tolerant", "unfiltered"])
def test_deploy_fasta_agrees_with_the_parquet(bundle, tmp_path, tier):
    cols = builder.collect_columns(bundle, None)
    out = _deploy(bundle, tmp_path, tier, f"deploy_{tier}.fasta")
    assert deploy_mismatches(out, cols) == set()
    got = _targets(out)
    # tolerant labels even in the unfiltered-tier training set: its train_tier
    assert got[TOL_TRAIN] == ("train", "0.6;0.5;0.4;999.0")
    assert got[TEST] == ("test", "0.3;0.2;0.1;999.0")
    summary = json.loads(out.with_suffix(".summary.json").read_text())
    assert summary["label_tier_source_counts"]["tolerant"] == 2


def test_deploy_fasta_default_work_dir_layout(tmp_path):
    """Defaults: <root>/data/interim/scored/<tier>/scores.json and the flat
    <work-dir>/mmseqs/train_<tier>_best.fasta files of the build chain."""
    bundle = write_bundle(tmp_path / "bundle")
    root = tmp_path / "repo"
    scored = root / "data" / "interim" / "scored"
    shutil.copytree(bundle / "scores", scored)
    work = tmp_path / "work"
    (work / "mmseqs").mkdir(parents=True)
    for tier in TIERS:
        shutil.copy(
            bundle / "train" / tier / f"train_{tier}_best.fasta", work / "mmseqs"
        )
    (work / "testset").mkdir()
    shutil.copy(bundle / "test" / "TriZOD_test_set.fasta", work / "testset")
    out = tmp_path / "deploy.fasta"
    deploy_fasta.main(["--work-dir", str(work), "--root", str(root), "--out", str(out)])
    assert deploy_mismatches(out, builder.collect_columns(bundle, None)) == set()


def test_label_source_tier_rule():
    reps = {t: set(REPS[t]) for t in TIERS}
    held = {t: {r["ID"] for r in SCORES[t]} for t in TIERS}
    for cid, tier in EXPECTED_SOURCE.items():
        assert deploy_fasta.label_source_tier(cid, reps, held) == tier, cid
    assert deploy_fasta.label_source_tier("nowhere", reps, held) is None


# --------------------------------------------------------------------------- #
# real data: the 2026-10 release bundle
# --------------------------------------------------------------------------- #

BUNDLE_NAME = "trizod-dataset-2026-10"
#: Built from that bundle on 2026-10-30, before I-01 was fixed.
PRE_FIX_DIR = Path("parquet-2026-10-simplify")


def _main_checkout_root() -> Path | None:
    """The main checkout's root when running from a git worktree (data/ is
    gitignored, so worktrees have none of their own)."""
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=paths.ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(common).parent if common else None


def _build_dirs() -> list[Path]:
    dirs = [paths.INTERIM_BUILD]
    main = _main_checkout_root()
    if main is not None:
        dirs.append(main / "data" / "interim" / "build")
    return dirs


def _real_bundle() -> Path | None:
    env = os.environ.get("TRIZOD_RELEASE_BUNDLE")
    if env:
        return Path(env)
    for build in _build_dirs():
        candidate = build / "release_bundle" / BUNDLE_NAME
        if (candidate / "scores" / "tolerant" / "scores.json").is_file():
            return candidate
    return None


def _pre_fix(name: str) -> Path | None:
    for build in _build_dirs():
        candidate = build / PRE_FIX_DIR / name
        if candidate.is_file():
            return candidate
    return None


REAL_BUNDLE = _real_bundle()
requires_release_bundle = pytest.mark.skipif(
    REAL_BUNDLE is None,
    reason=f"release bundle {BUNDLE_NAME} not found (set TRIZOD_RELEASE_BUNDLE)",
)


def _splits(table, ids):
    split = dict(zip(table["id"], table["split"]))
    out = defaultdict(int)
    for cid in ids:
        out[split[cid]] += 1
    return dict(out)


@requires_release_bundle
def test_release_parquet_built_before_the_fix_fails_the_check():
    """The check would have caught I-01: the 2026-10 Parquet ships unfiltered
    labels for the 478 tolerant-tier training representatives whose tolerant
    record masks an over-limit atom type (1,073 records in all)."""
    old = _pre_fix("trizod_dataset.parquet")
    if old is None:
        pytest.skip("pre-fix 2026-10 Parquet not on disk")
    table = _read_table(old)
    bad = label_mismatches(REAL_BUNDLE, table)
    assert bad["label_tier_source"] == {"<column missing>"}
    wrong = set().union(*(v for k, v in bad.items() if k != "label_tier_source"))
    assert len(wrong) == 1073
    assert _splits(table, wrong)["train"] == 478


@requires_release_bundle
def test_shift_parquet_built_before_the_fix_fails_the_check():
    """...and its shift table applies offsets the tolerant tier rejected, e.g.
    every C shift of 51029_1_1_1 moved by -1.01 ppm."""
    old = _pre_fix("trizod_shifts.parquet")
    if old is None:
        pytest.skip("pre-fix 2026-10 shift Parquet not on disk")
    bad = shift_offset_mismatches(REAL_BUNDLE, old)
    assert "51029_1_1_1" in bad
    assert len(bad) > 478


@requires_release_bundle
def test_rebuilt_release_labels_pass_the_check(tmp_path):
    """The fixed builder on the real bundle: every row is its selecting tier's
    record, 1,073 of them (478 training) now differ from the unfiltered run, and
    the deploy FASTA built from the same bundle agrees with the Parquet."""
    cols = builder.collect_columns(REAL_BUNDLE, None)
    assert label_mismatches(REAL_BUNDLE, cols) == {}
    unf = _jsonl(REAL_BUNDLE / "scores" / "unfiltered" / "scores.json")
    moved = {cid for i, cid in enumerate(cols["id"]) if row_diffs(cols, i, unf[cid])}
    assert len(moved) == 1073
    assert _splits(cols, moved)["train"] == 478
    out = _deploy(REAL_BUNDLE, tmp_path, "tolerant", "deploy.fasta")
    assert deploy_mismatches(out, cols) == set()


@pytest.mark.skipif(
    not os.environ.get("TRIZOD_CHECK_PARQUET")
    and not os.environ.get("TRIZOD_CHECK_SHIFTS"),
    reason="set TRIZOD_CHECK_PARQUET / TRIZOD_CHECK_SHIFTS to gate a built artefact",
)
@requires_release_bundle
def test_staged_artefacts_pass_the_check():
    main = os.environ.get("TRIZOD_CHECK_PARQUET")
    if main:
        assert label_mismatches(REAL_BUNDLE, _read_table(Path(main))) == {}
    shifts = os.environ.get("TRIZOD_CHECK_SHIFTS")
    if shifts:
        assert shift_offset_mismatches(REAL_BUNDLE, Path(shifts)) == set()
