"""Pipeline tests for Phase 2 (removal of the global ``bmrb_entries``).

Proves that ``create_peptide_dataframe`` and ``compute_scores_row`` receive the
BMRB entries by explicit argument (not a module global), and that the entries
payload is picklable — i.e. serializable to a pandarallel worker.

The scoring step is exercised with a *serial* ``apply`` here on purpose: running
the BLAS-heavy LACS solve inside a forked pandarallel worker from within the
pytest process (which has already loaded scipy/BLAS) segfaults on macOS
(fork + Accelerate/OpenBLAS is not fork-safe). The real multiprocessing path is
covered end-to-end by ``tests/test_smoke.py::test_emit_str_smoke`` (clean
subprocess). This test targets the Phase-2 contract: explicit passing + picklability.
"""

import pickle

import numpy as np
import pytest

import trizod.trizod as trizod_mod
from tests.conftest import BMRB_DIR, requires_bmrb_data
from trizod.pipeline import load_bmrb_entries, prefilter_dataframe

TWO_IDS = ["4493", "6968"]


def _prefilter_kwargs(tier="unfiltered"):
    d = trizod_mod.filter_defaults.loc[tier]
    plr = list(d["peptide-length-range"])
    if len(plr) == 1:
        plr = [plr[0], np.inf]
    return {
        "method_whitelist": d["exp-method-whitelist"],
        "method_blacklist": d["exp-method-blacklist"],
        "temperature_range": d["temperature-range"],
        "ionic_strength_range": d["ionic-strength-range"],
        "pH_range": d["pH-range"],
        "peptide_length_range": plr,
        "min_backbone_shift_types": d["min-backbone-shift-types"],
        "min_backbone_shift_positions": d["min-backbone-shift-positions"],
        "min_backbone_shift_fraction": d["min-backbone-shift-fraction"],
        "max_noncanonical_fraction": d["max-noncanonical-fraction"],
        "max_x_fraction": d["max-x-fraction"],
        "keywords": d["keywords-blacklist"],
        "perturbing_cosolvents": d["perturbing-cosolvents"],
        "exclude_paramagnetic": d["exclude-paramagnetic"],
    }


@requires_bmrb_data
def test_entries_passed_explicitly_and_scored(tmp_path):
    bmrb_files = {
        i: BMRB_DIR / f"bmr{i}" / f"bmr{i}_3.str"
        for i in TWO_IDS
        if (BMRB_DIR / f"bmr{i}" / f"bmr{i}_3.str").exists()
    }
    if len(bmrb_files) < 2:
        pytest.skip("need two BMRB entries on disk")

    from pandarallel import pandarallel

    pandarallel.initialize(verbose=0, nb_workers=1)

    entries, failed = load_bmrb_entries(bmrb_files, cache_dir=None)
    assert not failed

    # The entries payload that pandarallel would pickle to a worker must be
    # serializable (this is what makes explicit passing viable under fork/spawn).
    assert len(pickle.loads(pickle.dumps(entries))) == len(entries)

    tier = trizod_mod.filter_defaults.loc["unfiltered"]
    # create_peptide_dataframe -> fill_row_data reads `bmrb_entries` (the arg).
    df = trizod_mod.create_peptide_dataframe(
        entries,
        perturbing_cosolvents=tier["perturbing-cosolvents"],
        keywords=tier["keywords-blacklist"],
    )
    assert len(df) > 0
    assert df["seq"].notna().any(), "fill_row_data did not populate sequences"

    df, *_ = prefilter_dataframe(df, **_prefilter_kwargs("unfiltered"))
    assert df["pass_pre"].any(), "no peptide passed the unfiltered prefilter"

    cache_dir = tmp_path / "cache"
    for sub in ("wSCS", "potenci"):
        (cache_dir / sub).mkdir(parents=True)

    # Serial apply (no fork): proves compute_scores_row reads the explicit
    # `bmrb_entries` argument rather than a module global.
    df = df.apply(
        trizod_mod.compute_scores_row,
        axis=1,
        score_types=["zscores", "gscores"],
        cache_dir=cache_dir,
        bmrb_entries=entries,
    )
    scored = df[df["pass_pre"]]
    has_scores = scored["zscores"].apply(lambda x: isinstance(x, np.ndarray))
    assert has_scores.any(), "no scores computed via explicitly-passed entries"
