"""Regression: the scoring CLI reproduces a named build's labels on a seeded,
rule-aware subset of its BMRB snapshot (release audit 2026-11-17b, F30).

The baseline is the per-tier ``scores.json`` of a build that is on disk anyway
(``data/interim/<build>/root/data/interim/scored/<tier>/scores.json``), so no
separate baselines have to be regenerated, and the provenance of the baseline is
the build's own package bundle (``pipeline_version``). A seeded subset of the
build's snapshot is re-scored at every tier with the current checkout:

* the subset draws from every place a rule acts -- random scored entries, each
  tier band, the 273 K placeholder, high-pressure, membrane-mimetic and
  high-ionic-strength entries, entries no tier scores -- plus the CheZOD
  reference subset and named chains (the CheZOD117 placeholder chains, the
  alphabet-fix chains);
* every field of every baseline record is compared (lists element by element,
  floats to 1e-6), and the record sets must be equal at every tier.

Rule-aware: a difference in a rule flag (``membrane_mimetic_sample``,
``temperature_placeholder``, ``pressure_bar``, ...) is accepted only where the
code that sets the flag changed since the baseline's scoring commit (``git
diff``), and then only on the chains whose flag changed, with the tier moves
and re-scored labels that flag explains; the test prints them. Any other
difference fails.

Run (``-m slow``; the test is deselected by default)::

    TRIZOD_DATA_ROOT=<checkout holding data/> \\
    TRIZOD_REGRESSION_BUILD=rebuild-2026-11-17b \\
        uv run --with pyarrow python -m pytest tests/test_full_dataset_regression.py \\
        -m slow -v

``TRIZOD_REGRESSION_BUILD`` is a directory under ``data/interim`` (or a path)
holding ``root/data/interim/scored/``; its snapshot is
``root/data/raw/bmrb_entries`` unless ``TRIZOD_BMRB_DIR`` names another. The
scoring cache goes to ``tmp/regression/<build>`` of this checkout
(``TRIZOD_REGRESSION_CACHE``), never a cache of another snapshot: cached
entries are keyed by entry ID and code version, not by file content.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
from functools import cache, lru_cache
from pathlib import Path

import pytest

from tests.conftest import subprocess_env
from trizod import paths

TIERS = ("unfiltered", "tolerant", "moderate", "strict")
SEED = int(os.environ.get("TRIZOD_REGRESSION_SEED", "20261117"))
#: Entries drawn per group (the verifier's design of the 2026-11-17b audit).
GROUP_SIZES = {
    "random_scored": 50,
    "strict": 15,
    "moderate_not_strict": 10,
    "tolerant_not_moderate": 10,
    "temperature_placeholder": 12,
    "high_pressure": 5,
    "membrane_mimetic_sample": 10,
    "ionic_strength_above_0.5": 6,
    "not_scored": 12,
}
#: Entries always included when the snapshot holds them: the CheZOD117
#: placeholder chains and the two alphabet-fix chains of v0.4.0.
NAMED_ENTRIES = ("18177", "19357", "19796", "50015", "52376")
CHEZOD_REFERENCE = (
    Path(__file__).resolve().parent / "reference" / ("chezod_zscores_subset.json")
)
#: Rule flags and the code that sets them; a flag difference is accepted only
#: where one of its sources changed since the baseline's scoring commit.
RULE_SOURCES = {
    "membrane_mimetic_sample": (
        "trizod/bmrb/membrane.py",
        "trizod/dataset/data/membrane_overrides.tsv",
    ),
    "membrane_mimetic": (
        "trizod/bmrb/membrane.py",
        "trizod/dataset/data/membrane_overrides.tsv",
    ),
    "temperature_placeholder": ("trizod/bmrb/bmrb.py",),
    "temperature": ("trizod/bmrb/bmrb.py",),
    "pH": ("trizod/bmrb/bmrb.py",),
    "ionic_strength": ("trizod/bmrb/bmrb.py",),
    "pressure_bar": ("trizod/bmrb/bmrb.py",),
    "cosolvent_evidence": ("trizod/bmrb/sample_state.py", "trizod/bmrb/bmrb.py"),
    "sample_state_evidence": ("trizod/bmrb/sample_state.py",),
    "physical_state": ("trizod/bmrb/bmrb.py",),
}
#: Rule flags whose change re-scores a chain (its labels may then differ).
RESCORING_FLAGS = ("temperature_placeholder", "temperature", "pH", "ionic_strength")
FLOAT_TOL = 1e-6


# --------------------------------------------------------------------------- #
# the named build
# --------------------------------------------------------------------------- #


def _build_root() -> Path | None:
    name = os.environ.get("TRIZOD_REGRESSION_BUILD", "").strip()
    if not name:
        return None
    p = Path(name).expanduser()
    if not p.is_absolute():
        p = paths.INTERIM / name
    return p / "root" if (p / "root").is_dir() else p


BUILD_ROOT = _build_root()
SKIP_REASON = (
    "set TRIZOD_REGRESSION_BUILD to a build under data/interim (e.g. "
    "rebuild-2026-11-17b) and TRIZOD_DATA_ROOT to the checkout holding data/"
)


def _scored_dir() -> Path:
    return BUILD_ROOT / "data" / "interim" / "scored"


def _snapshot() -> Path:
    override = os.environ.get(paths.BMRB_DIR_ENV, "").strip()
    return Path(override) if override else BUILD_ROOT / "data" / "raw" / "bmrb_entries"


def _jsonl(path: Path) -> dict[str, dict]:
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[rec["ID"]] = rec
    return out


@lru_cache(maxsize=1)
def baseline() -> dict[str, dict[str, dict]]:
    return {t: _jsonl(_scored_dir() / t / "scores.json") for t in TIERS}


@lru_cache(maxsize=1)
def scoring_commit() -> str | None:
    """Short SHA of the commit that scored the build (its bundle's
    ``pipeline_version``), or None."""
    for manifest in sorted(
        (BUILD_ROOT / "data" / "interim" / "build" / "release_bundle").glob(
            "*/MANIFEST.json"
        )
    ):
        version = json.loads(manifest.read_text()).get("pipeline_version", "")
        if version.startswith("trizod-"):
            return version.removeprefix("trizod-").removesuffix("-dirty")
    return None


@cache
def changed_since_baseline(source: str) -> bool:
    """Did ``source`` change between the scoring commit and this checkout
    (working tree included)? False when that cannot be told."""
    commit = scoring_commit()
    if commit is None:
        return False
    r = subprocess.run(
        ["git", "diff", "--quiet", commit, "--", source],
        cwd=paths.ROOT,
        capture_output=True,
    )
    return r.returncode == 1


# --------------------------------------------------------------------------- #
# the seeded, rule-aware subset
# --------------------------------------------------------------------------- #


def _entry(cid: str) -> str:
    return cid.split("_", 1)[0]


@lru_cache(maxsize=1)
def subset() -> tuple[dict[str, list[str]], list[str]]:
    """``(groups, entries)``: the seeded draw per group and every entry."""
    base = baseline()
    ent = {t: {_entry(c) for c in base[t]} for t in TIERS}
    unf = base["unfiltered"].values()
    snapshot = sorted(
        d.name[3:]
        for d in _snapshot().iterdir()
        if d.name.startswith("bmr") and d.name[3:].isdigit()
    )

    def flagged(pred):
        return sorted({_entry(r["ID"]) for r in unf if pred(r)})

    pools = {
        "random_scored": sorted(ent["unfiltered"]),
        "strict": sorted(ent["strict"]),
        "moderate_not_strict": sorted(ent["moderate"] - ent["strict"]),
        "tolerant_not_moderate": sorted(ent["tolerant"] - ent["moderate"]),
        "temperature_placeholder": flagged(lambda r: r.get("temperature_placeholder")),
        "high_pressure": flagged(lambda r: (r.get("pressure_bar") or 0) >= 400),
        "membrane_mimetic_sample": flagged(lambda r: r.get("membrane_mimetic_sample")),
        "ionic_strength_above_0.5": flagged(
            lambda r: (r.get("ionic_strength") or 0) > 0.5
        ),
        "not_scored": sorted(set(snapshot) - ent["unfiltered"]),
    }
    rng = random.Random(SEED)
    groups = {
        g: sorted(rng.sample(pool, min(GROUP_SIZES[g], len(pool))))
        for g, pool in pools.items()
    }
    chosen = set().union(*groups.values())
    in_snapshot = set(snapshot)
    chezod = (
        set(json.loads(CHEZOD_REFERENCE.read_text()))
        if CHEZOD_REFERENCE.exists()
        else set()
    )
    named = (set(NAMED_ENTRIES) | chezod) & in_snapshot
    groups["named_and_chezod_reference"] = sorted(named)
    return groups, sorted(chosen | named, key=int)


# --------------------------------------------------------------------------- #
# re-scoring
# --------------------------------------------------------------------------- #


def _cache_dir() -> Path:
    env = os.environ.get("TRIZOD_REGRESSION_CACHE", "").strip()
    if env:
        return Path(env)
    return paths.TMP / "regression" / BUILD_ROOT.parent.name


@pytest.fixture(scope="module")
def rescored(tmp_path_factory) -> dict[str, dict[str, dict]]:
    _, entries = subset()
    work = tmp_path_factory.mktemp("regression")
    inp = work / "bmrb"
    inp.mkdir()
    for eid in entries:
        (inp / f"bmr{eid}").symlink_to((_snapshot() / f"bmr{eid}").resolve())
    out = {}
    for tier in TIERS:
        prefix = work / tier
        cmd = [
            sys.executable,
            "-m",
            "trizod.trizod",
            "--input-dir",
            str(inp),
            "--filter-defaults",
            tier,
            "--output-prefix",
            str(prefix),
            "--output-format",
            "json",
            "--no-progress",
            "--cache-dir",
            str(_cache_dir()),
        ]
        # single-threaded BLAS: the forked pandarallel workers deadlock on macOS
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200, env=subprocess_env()
        )
        assert r.returncode == 0, f"{tier}: pipeline failed\n{r.stderr[-4000:]}"
        out[tier] = _jsonl(Path(f"{prefix}.json"))
    return out


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #


def _null(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def same(a, b) -> bool:
    """Equal as written: null == NaN; numbers to 1e-6; lists element-wise."""
    if isinstance(a, list) or isinstance(b, list):
        return (
            isinstance(a, list)
            and isinstance(b, list)
            and len(a) == len(b)
            and all(same(x, y) for x, y in zip(a, b))
        )
    if _null(a) or _null(b):
        return _null(a) and _null(b)
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= FLOAT_TOL * max(1.0, abs(a), abs(b))
    return a == b


def rule_changes(base_unf: dict, new_unf: dict) -> dict[str, set[str]]:
    """Chain -> the rule flags that differ between the baseline and the
    re-scored unfiltered records."""
    out = {}
    for cid in set(base_unf) & set(new_unf):
        flags = {
            f
            for f in RULE_SOURCES
            if f in base_unf[cid]
            and not same(base_unf[cid].get(f), new_unf[cid].get(f))
        }
        if flags:
            out[cid] = flags
    return out


def compare_tier(base: dict, new: dict, flags: dict[str, set[str]]) -> dict:
    """Differences of one tier: ``unexplained`` fails the test, ``explained``
    are rule changes whose code changed since the baseline."""
    unexplained, explained = [], []

    def accepted(cid) -> bool:
        changed = flags.get(cid, set())
        return bool(changed) and all(
            any(changed_since_baseline(s) for s in RULE_SOURCES[f]) for f in changed
        )

    for cid in sorted(set(base) ^ set(new)):
        side = "only in the baseline" if cid in base else "only re-scored"
        (explained if accepted(cid) else unexplained).append(
            f"{cid}: {side} (flags changed: {sorted(flags.get(cid, ())) or 'none'})"
        )
    for cid in sorted(set(base) & set(new)):
        changed = flags.get(cid, set())
        rescoring = bool(changed & set(RESCORING_FLAGS))
        for field, value in base[cid].items():
            if same(value, new[cid].get(field)):
                continue
            if accepted(cid) and (field in changed or rescoring):
                explained.append(f"{cid}.{field} (flags changed: {sorted(changed)})")
            else:
                unexplained.append(f"{cid}.{field}")
    return {"unexplained": unexplained, "explained": explained}


@pytest.mark.slow
@pytest.mark.skipif(
    BUILD_ROOT is None or not (BUILD_ROOT / "data" / "interim" / "scored").is_dir(),
    reason=SKIP_REASON,
)
@pytest.mark.parametrize("tier", TIERS)
def test_rescored_subset_matches_the_named_build(rescored, tier):
    groups, entries = subset()
    keep = set(entries)
    base = {c: r for c, r in baseline()[tier].items() if _entry(c) in keep}
    new = rescored[tier]
    flags = rule_changes(
        {c: r for c, r in baseline()["unfiltered"].items() if _entry(c) in keep},
        rescored["unfiltered"],
    )
    diff = compare_tier(base, new, flags)
    print(
        f"\n{tier}: {len(entries)} entries ({ {g: len(v) for g, v in groups.items()} }), "
        f"{len(base)} baseline records, {len(new)} re-scored; baseline scored by "
        f"{scoring_commit() or 'unknown'}"
    )
    for line in diff["explained"]:
        print(f"  rule change since {scoring_commit()}: {line}")
    assert len(base) > 0, "the subset holds no record of this tier"
    assert not diff["unexplained"], (
        f"{tier}: {len(diff['unexplained'])} differences against "
        f"{_scored_dir()} (scored by {scoring_commit()}), e.g.\n"
        + "\n".join(diff["unexplained"][:20])
    )


# --------------------------------------------------------------------------- #
# the comparison itself, without data
# --------------------------------------------------------------------------- #


def test_same_treats_null_and_nan_alike_and_compares_lists():
    assert same([None, 1.0, float("nan")], [float("nan"), 1.0 + 1e-9, None])
    assert not same([1.0], [1.0, 2.0])
    assert not same(True, 1) and same("a", "a") and not same(0.5, 0.51)


def test_flag_changes_are_accepted_only_where_their_code_changed(monkeypatch):
    base = {
        "1_1_1_1": {"ID": "1_1_1_1", "gscores": [0.1], "membrane_mimetic_sample": False}
    }
    new = {
        "1_1_1_1": {"ID": "1_1_1_1", "gscores": [0.1], "membrane_mimetic_sample": True}
    }
    flags = rule_changes(base, new)
    assert flags == {"1_1_1_1": {"membrane_mimetic_sample"}}
    monkeypatch.setattr(
        sys.modules[__name__], "changed_since_baseline", lambda source: True
    )
    got = compare_tier({}, new, flags)  # left the tier: explained by the rule
    assert got["unexplained"] == [] and len(got["explained"]) == 1
    monkeypatch.setattr(
        sys.modules[__name__], "changed_since_baseline", lambda source: False
    )
    got = compare_tier(base, new, flags)
    assert got["unexplained"] == ["1_1_1_1.membrane_mimetic_sample"]
    # a label change without a re-scoring flag is never explained
    moved = {"1_1_1_1": {**new["1_1_1_1"], "gscores": [0.9]}}
    monkeypatch.setattr(
        sys.modules[__name__], "changed_since_baseline", lambda source: True
    )
    got = compare_tier(base, moved, flags)
    assert got["unexplained"] == ["1_1_1_1.gscores"]
