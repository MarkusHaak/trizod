"""A pickled ``BmrbEntry`` from older parsing code is re-parsed, not reused.

``tmp/bmrb_entries/<id>.pkl`` caches the parsed object, not the file. When the
parser learns a new field -- ``_Sample.Solvent_system`` in 2026-10 -- every
pickle written before that simply lacks the attribute, and a re-filter on the
warm cache would either crash on it or (with a ``getattr`` default) silently
filter as if no sample declared a solvent system. The entry therefore carries
the version of the parsing code that produced it, and the loader treats any
other version as a cache miss.
"""

import pickle

import pandas as pd

import trizod.pipeline as pipeline
from tests.conftest import requires_bmrb_data
from trizod import paths
from trizod.bmrb import bmrb
from trizod.provenance import bmrb_parser_version


class _Stale:
    """A pickled entry from before the version stamp existed."""

    id = "1"


class _Current:
    id = "1"

    def __init__(self, version):
        self.parser_version = version


def _pickle(path, obj):
    with path.open("wb") as fh:
        pickle.dump(obj, fh)
    return path


def test_parser_version_is_a_stable_short_digest():
    version = bmrb_parser_version()
    assert isinstance(version, str) and len(version) == 12
    assert bmrb_parser_version() == version


def test_current_pickle_is_reused(tmp_path):
    fp = _pickle(tmp_path / "1.pkl", _Current(bmrb_parser_version()))
    entry = pipeline.load_cached_entry(fp)
    assert isinstance(entry, _Current)


def test_unstamped_pickle_is_a_miss(tmp_path):
    fp = _pickle(tmp_path / "1.pkl", _Stale())
    assert pipeline.load_cached_entry(fp) is None


def test_pickle_from_other_parsing_code_is_a_miss(tmp_path):
    fp = _pickle(tmp_path / "1.pkl", _Current("0123456789ab"))
    assert pipeline.load_cached_entry(fp) is None


def test_missing_or_corrupt_pickle_is_a_miss(tmp_path):
    assert pipeline.load_cached_entry(tmp_path / "absent.pkl") is None
    corrupt = tmp_path / "corrupt.pkl"
    corrupt.write_bytes(b"not a pickle")
    assert pipeline.load_cached_entry(corrupt) is None


def test_load_bmrb_entries_reparses_a_stale_pickle(tmp_path, monkeypatch):
    """Stale and current pickles side by side: only the stale one is re-parsed."""
    cache = tmp_path / "cache"
    (cache / "bmrb_entries").mkdir(parents=True)
    _pickle(cache / "bmrb_entries" / "1.pkl", _Stale())
    _pickle(cache / "bmrb_entries" / "2.pkl", _Current(bmrb_parser_version()))
    bmrb_files = {"1": tmp_path / "bmr1_3.str", "2": tmp_path / "bmr2_3.str"}

    parsed = []

    def _fake_parse(row, cache_dir=None):
        parsed.append(row.name)
        return _Current(bmrb_parser_version())

    def _serial_apply(self, func, axis=0, **kwargs):
        return self.apply(func, axis=axis, **kwargs)

    monkeypatch.setattr(pipeline, "parse_bmrb_file", _fake_parse)
    monkeypatch.setattr(pd.DataFrame, "parallel_apply", _serial_apply, raising=False)

    entries, failed = pipeline.load_bmrb_entries(bmrb_files, cache_dir=cache)
    assert parsed == ["1"]
    assert failed == []
    assert sorted(entries.index) == ["1", "2"]


@requires_bmrb_data
def test_parsed_entry_carries_the_parser_version():
    entry = bmrb.BmrbEntry("34568", paths.RAW_BMRB / "bmr34568")
    assert entry.parser_version == bmrb_parser_version()
    assert entry.samples["1"].solvent_system == "trifluoroethanol/water"


@requires_bmrb_data
def test_parse_bmrb_file_rewrites_the_pickle_with_the_version(tmp_path):
    cache_fp = tmp_path / "34568.pkl"
    _pickle(cache_fp, _Stale())
    row = pd.Series(
        {"dir": paths.RAW_BMRB / "bmr34568", "cache_fp": cache_fp}, name="34568"
    )
    pipeline.parse_bmrb_file(row, cache_dir=tmp_path)
    reloaded = pipeline.load_cached_entry(cache_fp)
    assert reloaded is not None
    assert reloaded.samples["1"].solvent_system == "trifluoroethanol/water"


def test_version_tracks_the_parser_source_only():
    # The digest is over the parser module only: editing the parser must
    # invalidate the parse cache, editing scoring or filter code must not.
    import trizod.provenance as provenance

    assert provenance._PARSER_FILE == provenance._PKG_ROOT / "bmrb" / "bmrb.py"
    assert bmrb_parser_version() != provenance.scoring_cache_version()
