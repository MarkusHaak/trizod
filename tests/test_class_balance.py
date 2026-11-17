"""Per-split class balance (release audit 2026-11-17, F33): zone shares at the
three cuts, the k < 10 share, the length distribution and, where the threshold
framework transported the calibration, pi and the posterior at each cut."""

import math

import pytest

from trizod.dataset import class_balance as cb

from ._release_fixtures import (
    FRAMEWORK_COMMIT,
    build_release_parquet,
    write_bundle,
    write_results,
)

NAN = math.nan


def _row(split, g, k, length=None, train_tier=None):
    mask = [not math.isnan(x) for x in g]
    return {
        "split": split,
        "train_tier": train_tier,
        "gscores": g,
        "mask": mask,
        "k": k,
        "length": length if length is not None else len(g),
    }


ROWS = [
    # strict training representative: G in every zone, two masked residues
    _row(
        "train", [NAN, 0.10, 0.20, 0.50, 0.80, NAN], [0, 3, 12, 12, 12, 0], 6, "strict"
    ),
    # tolerant representative: all ordered, all k >= 10
    _row("train", [0.0, 0.05, 0.149], [10, 11, 12], 10, "tolerant"),
    _row("redundant", [0.9, 0.9], [12, 12]),
    _row("test_trizod", [0.35, 0.70, 0.1499, NAN], [9, 10, 20, 0], 4),
    _row("test_chezod117", [0.9], [15], 1),
    _row("test_retired", [0.9], [15], 1),
]


def _by_split(table):
    return {r["split"]: r for r in table}


def test_rows_cover_the_training_tiers_and_both_test_sets():
    table = cb.compute(ROWS)
    assert [r["split"] for r in table] == [
        "train_unfiltered",
        "train_tolerant",
        "train_moderate",
        "train_strict",
        "test_trizod",
        "test_chezod117",
    ]


def test_training_tiers_nest():
    t = _by_split(cb.compute(ROWS))
    assert t["train_strict"]["chains"] == 1
    assert t["train_moderate"]["chains"] == 1
    assert t["train_tolerant"]["chains"] == 2
    assert t["train_unfiltered"]["chains"] == 2


def test_zone_shares_follow_g_ge_t_over_scored_residues():
    s = _by_split(cb.compute(ROWS))["train_strict"]
    assert s["scored_residues"] == 4
    assert s["share_ordered"] == pytest.approx(0.25)  # 0.10
    assert s["share_indeterminate"] == pytest.approx(0.25)  # 0.20
    assert s["share_disordered"] == pytest.approx(0.25)  # 0.50
    assert s["share_confidently_disordered"] == pytest.approx(0.25)  # 0.80
    t = _by_split(cb.compute(ROWS))["test_trizod"]
    # 0.35 is disordered and 0.70 confidently disordered (G >= t); 0.1499 ordered
    assert t["share_ordered"] == pytest.approx(1 / 3)
    assert t["share_disordered"] == pytest.approx(1 / 3)
    assert t["share_confidently_disordered"] == pytest.approx(1 / 3)
    assert t["share_G_ge_0.35"] == pytest.approx(2 / 3)


def test_k_below_10_share_is_over_scored_residues_only():
    t = _by_split(cb.compute(ROWS))
    assert t["train_strict"]["share_k_lt_10"] == pytest.approx(0.25)
    # tolerant = the strict representative (1 of 4 below 10) + the tolerant one
    assert t["train_tolerant"]["share_k_lt_10"] == pytest.approx(1 / 7)
    assert t["test_trizod"]["share_k_lt_10"] == pytest.approx(1 / 3)


def test_length_median_and_iqr():
    t = _by_split(cb.compute(ROWS))["train_tolerant"]
    assert t["length_median"] == 8.0
    assert (t["length_q1"], t["length_q3"]) == (7.0, 9.0)


def test_pi_and_posterior_come_from_the_transported_calibration(tmp_path):
    from trizod.dataset import release_info as ri

    cal = ri.read_calibration(write_results(tmp_path), FRAMEWORK_COMMIT)
    t = _by_split(cb.compute(ROWS, cal))
    assert t["test_trizod"]["pi"] == 0.15
    assert t["test_trizod"]["pi_ci95"] == pytest.approx([0.1, 0.2])
    assert t["test_trizod"]["posterior_at_0.35"] == 0.5
    # the framework's "train" set is every training representative = unfiltered
    assert t["train_unfiltered"]["pi"] == 0.27
    assert t["train_unfiltered"]["pi_source"].startswith("threshold framework")


def test_pi_of_an_untransported_split_comes_from_the_lr_curve(tmp_path):
    """The framework's prior-shift EM on its per-bin likelihood ratios: pi is
    the fixed point of pi = mean P(disordered | G, pi); the posterior at a cut
    follows from pi and the likelihood ratio at the cut."""
    from trizod.dataset import release_info as ri

    from ._release_fixtures import LOG_LR

    cal = ri.read_calibration(write_results(tmp_path), FRAMEWORK_COMMIT)
    assert cal["lr_curve"]["log_lr"] == LOG_LR
    s = _by_split(cb.compute(ROWS, cal))["train_strict"]
    g = [0.10, 0.20, 0.50, 0.80]
    lr = [math.exp(LOG_LR[int(x * 100)]) for x in g]
    pi = s["pi"]
    assert pi == pytest.approx(
        sum(x * pi / (x * pi + 1 - pi) for x in lr) / len(lr), abs=1e-4
    )
    assert s["pi_ci95"] is None
    assert s["pi_source"].startswith("EM on the calibration LR curve")
    assert s["posterior_at_0.35"] == pytest.approx(ri.posterior(pi, 5.2), abs=1e-4)
    assert "*" in cb.markdown(cb.compute(ROWS, cal))


def test_without_the_lr_curve_an_untransported_split_has_no_pi(tmp_path):
    from trizod.dataset import release_info as ri

    cal = ri.read_calibration(write_results(tmp_path, curve=False), FRAMEWORK_COMMIT)
    assert cal["lr_curve"] is None
    s = _by_split(cb.compute(ROWS, cal))["train_strict"]
    assert s["pi"] is None
    assert s["pi_source"] == "not calibrated"


def test_without_calibration_pi_is_empty():
    t = _by_split(cb.compute(ROWS))
    assert all(r["pi"] is None for r in t.values())


def test_markdown_and_tsv(tmp_path):
    table = cb.compute(ROWS)
    md = cb.markdown(table)
    assert md.splitlines()[0].startswith("| Split | Chains | Scored residues")
    assert "G ≥ 0.70" in md and "k < 10" in md
    assert "| train, strict | 1 | 4 | 25.0 % |" in md
    path = cb.write_tsv(table, tmp_path / "class_balance.tsv")
    lines = path.read_text().splitlines()
    assert len(lines) == 1 + len(table)
    assert lines[0].split("\t")[0] == "split"


def test_from_parquet_reads_the_calibration_metadata(tmp_path):
    pytest.importorskip("pyarrow")
    bundle = write_bundle(tmp_path / "bundle")
    out = build_release_parquet(
        bundle, tmp_path / "t.parquet", results_dir=write_results(tmp_path / "fw")
    )
    table, cal = cb.from_parquet(out)
    t = _by_split(table)
    assert cal["prevalence"]["value"] == 0.186
    assert t["test_trizod"]["chains"] == 2
    assert t["test_chezod117"]["chains"] == 1
    assert t["train_tolerant"]["chains"] == 3
    assert t["test_trizod"]["pi"] == 0.15
