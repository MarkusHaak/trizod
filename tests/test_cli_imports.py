"""Import-smoke tests for the ``trizod.trizod`` backward-compatibility surface.

Phase 1 of the repo restructure split ``trizod/trizod.py`` into
``trizod/pipeline.py`` + ``trizod/cache.py``. Several ``scripts/`` still import
these names from ``trizod.trizod``; this test locks that the re-export shims keep
resolving to the moved implementations, and that the ``bmrb_entries`` module
global (set externally by ``scripts/filter_impact_report.py``) stays settable.
"""

import trizod.cache as cache
import trizod.pipeline as pipeline
import trizod.trizod as trizod_mod


def test_cache_reexports_are_identical():
    assert trizod_mod._potenci_cache_key is cache._potenci_cache_key
    assert trizod_mod.load_potenci_cache is cache.load_potenci_cache
    assert trizod_mod.save_potenci_cache is cache.save_potenci_cache


def test_pipeline_reexports_are_identical():
    assert trizod_mod.find_bmrb_files is pipeline.find_bmrb_files
    assert trizod_mod.load_bmrb_entries is pipeline.load_bmrb_entries
    assert trizod_mod.prefilter_dataframe is pipeline.prefilter_dataframe
    assert trizod_mod.postfilter_dataframe is pipeline.postfilter_dataframe
    assert trizod_mod.print_filter_losses is pipeline.print_filter_losses
    assert trizod_mod.compute_scores is pipeline.compute_scores
    assert trizod_mod.ZscoreComputationError is pipeline.ZscoreComputationError


def test_script_import_surface():
    # exactly the symbols scripts/ import from trizod.trizod
    from trizod.trizod import (  # noqa: F401
        _potenci_cache_key,
        create_peptide_dataframe,
        filter_defaults,
        find_bmrb_files,
        load_bmrb_entries,
        load_potenci_cache,
        prefilter_dataframe,
        save_potenci_cache,
    )


def test_bmrb_entries_global_is_settable():
    # filter_impact_report.py does `_trizod_mod.bmrb_entries = global_entries`
    # before the pandarallel workers (fill_row_data) read it.
    sentinel = {"sentinel": 1}
    trizod_mod.bmrb_entries = sentinel
    assert trizod_mod.bmrb_entries is sentinel
    del trizod_mod.bmrb_entries
