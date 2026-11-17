"""Build selection, G-score cuts, split values and input-keyed caches, shared by the paper
figures and ``make_numbers``.

One place decides which build the paper describes and where the G-score cuts lie, so the
artwork (``style_paper``) and the prose (``scripts/manuscript/make_numbers.py``,
``threshold_numbers.py``) can never read two builds or two cuts. This module imports neither
matplotlib nor pyarrow.

**G-score cuts** (:data:`G_ORD`, :data:`G_DIS`, :data:`G_CONF`). The three calibrated cuts
are defined here and nowhere else; ``style_paper`` re-exports them and both numbers scripts
import them. Convention: a residue is disordered at G >= t, never G > t.

**Split values** (:data:`TEST_SPLITS`, :data:`RETIRED_SPLIT`, :data:`HELD_OUT_SPLITS`): the
release Parquet's ``split`` column. A ``test_retired`` chain is neither test nor training.

**Build selection** (:func:`resolve`). The default is the build of record, the 2026-11-11
snapshot rebuild at ``trizod-a481315`` in ``data/interim/rebuild-2026-11-11/`` of the
checkout that holds ``data/`` (a git worktree uses the main checkout's). Environment
variables override each path (``README.md``, "Selecting the build"):

=====================  ===================================================  ==============================
variable               what                                                 default
=====================  ===================================================  ==============================
``TRIZOD_DATA_ROOT``   root with ``data/raw``, ``data/external``, ``tmp/``  ``<record>/root``
``TRIZOD_BUILD_DIR``   ``final_dataset/``, ``testset/``, ``mmseqs/``        ``<root>/data/interim/build``
``TRIZOD_RELEASE_DIR`` ``trizod_dataset.parquet``, ``trizod_shifts.parquet``  ``<record>/release``
``TRIZOD_BUNDLE``      release bundle (``MANIFEST.json``, ``scores/`` ...)  ``<build>/release_bundle/trizod-dataset-2026-11``
``TRIZOD_REBUILD_LOG`` the build log (``print_filter_losses()`` reports)    ``<record>/rebuild_2026-11-11.log``
``TRIZOD_PKL_DIR``     the build's BmrbEntry pickles                        ``<root>/tmp/bmrb_entries``
``TRIZOD_DRAW_DIR``    the 2026-08-11 test-set draw (``_tmp/``)             ``<build>/testset`` if it holds ``_tmp/``, else ``<checkout>/data/interim/build/testset``
``TRIZOD_CACHE_DIR``   the generators' caches                               ``tmp/`` of the code checkout
=====================  ===================================================  ==============================

The four build variables (build, release, bundle, log) go together: setting some of them
but not all raises, so a partial override never mixes two builds.

**Input-keyed caches** (:func:`keyed_json`). A cache is valid only for the inputs it was
computed from. Its key is a digest of everything that decides the result, typically
:func:`snapshot_digest` (the raw BMRB snapshot), :func:`build_digest` (the build's
``MANIFEST.json``), :func:`code_digest` (the code that computes it) and the command or
arguments. A cache written for other inputs is recomputed, never silently reused.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trizod import paths
from trizod.io.atomic import atomic_write
from trizod.provenance import file_sha256

REPO = paths.ROOT  # the code checkout: scripts/, docs/, trizod/

# ----------------------------------------------------------------------------------
# G-score cuts: the calibrated three-cut scheme, set HERE ONLY
# ----------------------------------------------------------------------------------
# The cuts of the threshold framework (publication notes, manuscript-revision-2026-11,
# "Threshold framework soundness check", section 1; scripts/validation/threshold_calibration/
# framework): the smallest G at which the posterior probability of disorder reaches 0.10 /
# 0.50 / 0.90 at the reference-consensus prevalence of the strict tier's unbound chains,
# reported as multiples of 0.05 under the dated rule "keep each value while it lies inside the
# 95 % interval of its strict-tier estimate, else take the nearest multiple of 0.05 inside the
# interval". The recalibration on the rebuilt strict tier may still move a value: change it
# here, and the figures (style_paper), numbers.tex (make_numbers) and numbers_threshold.tex
# (threshold_numbers, which checks the framework's paper values against these) follow.
# Convention (checklist F32): disordered at G >= t; ordered at G < t.
G_ORD = 0.15  # ordered edge: a residue with G < G_ORD is (confidently) ordered
G_DIS = 0.35  # the binary cut: a residue with G >= G_DIS is disordered
G_CONF = 0.70  # disordered edge: a residue with G >= G_CONF is confidently disordered
# Zones of the paper: ordered G < G_ORD; ambiguous G_ORD <= G < G_DIS; disordered G >= G_DIS
# (the binary call), of which confidently disordered G >= G_CONF. Supplementary Fig. S20 and
# threshold_numbers call G_ORD <= G < G_CONF the indeterminate zone.

# ----------------------------------------------------------------------------------
# split values of the release Parquet (scripts/build_parquet_dataset.py)
# ----------------------------------------------------------------------------------
# train / redundant / excluded / test_trizod / test_chezod117 and, from the 2026-11-17 build
# on, test_retired: a pinned TriZOD test chain that no longer meets the strict tier. It is not
# in the test set, and it stays a leakage target, so it is never training either; every
# analysis keeps it out of both (trizod/dataset/testset.py on feat/tier-rules-2026-11).
TEST_SPLITS = ("test_trizod", "test_chezod117")  # the two test sets
RETIRED_SPLIT = "test_retired"
HELD_OUT_SPLITS = (*TEST_SPLITS, RETIRED_SPLIT)  # never training
RETIRED_FASTA = "TriZOD_test_retired.fasta"  # bundle test/ file of the retired chains

# ----------------------------------------------------------------------------------
# build of record
# ----------------------------------------------------------------------------------
BUILD_OF_RECORD = "trizod-a481315"  # pipeline_version of the build the paper reports
FROZEN_BUILD = (
    "trizod-5b8147a"  # the pre-fix 2026-10 build the first draft was written on
)
RECORD_NAME = "rebuild-2026-11-11"  # <checkout>/data/interim/<RECORD_NAME>
RECORD_BUNDLE = "trizod-dataset-2026-11"
RECORD_LOG = "rebuild_2026-11-11.log"
BUILD_VARS = (
    "TRIZOD_BUILD_DIR",
    "TRIZOD_RELEASE_DIR",
    "TRIZOD_BUNDLE",
    "TRIZOD_REBUILD_LOG",
)


def find_checkout(start: Path = REPO) -> Path:
    """The first of ``start`` and its parents that holds ``data/interim/build`` (the main
    checkout; a git worktree has no ``data/`` of its own), else ``start``."""
    for p in [start, *start.parents]:
        if (p / "data" / "interim" / "build").is_dir():
            return p
    return start


CHECKOUT = find_checkout()
RECORD_DIR = CHECKOUT / "data" / "interim" / RECORD_NAME


def env_path(name: str, default: Path) -> Path:
    """``$name`` as an absolute path when set and non-empty, else ``default``."""
    v = os.environ.get(name)
    return Path(v).expanduser().resolve() if v else default


def cache_root() -> Path:
    """``$TRIZOD_CACHE_DIR``, else ``tmp/`` of the code checkout."""
    return env_path("TRIZOD_CACHE_DIR", paths.TMP)


@dataclass(frozen=True)
class Build:
    """Input locations of one build (see the module docstring)."""

    root: Path  # data root: data/raw, data/external, data/interim/chezod_verification, tmp/
    build: Path  # final_dataset/, testset/, mmseqs/
    release: Path  # trizod_dataset.parquet, trizod_shifts.parquet (+ post-build files)
    bundle: Path  # MANIFEST.json, scores/, train/, test/
    log: Path  # build log
    pkl: Path  # BmrbEntry pickle cache of the build
    draw: Path  # the 2026-08-11 test-set draw (_tmp/, TriZOD_test_set_clu.tsv)

    @property
    def pq_main(self) -> Path:
        return self.release / "trizod_dataset.parquet"

    @property
    def pq_shifts(self) -> Path:
        return self.release / "trizod_shifts.parquet"

    @property
    def manifest(self) -> Path:
        return self.bundle / "MANIFEST.json"

    @property
    def layout(self):
        """:func:`trizod.paths.layout` of the data root (raw, external, interim paths)."""
        return paths.layout(self.root)


def resolve(root: Path | None = None) -> Build:
    """The build to read: the build of record unless environment variables say otherwise.

    ``root`` (``make_numbers --root``) wins over ``$TRIZOD_DATA_ROOT``."""
    set_vars = [v for v in BUILD_VARS if os.environ.get(v)]
    if set_vars and len(set_vars) != len(BUILD_VARS):
        missing = ", ".join(v for v in BUILD_VARS if v not in set_vars)
        raise SystemExit(
            f"{', '.join(set_vars)} set but not {missing}: set all four build variables "
            "(scripts/figures/paper/README.md, 'Selecting the build') or none for the "
            f"build of record ({RECORD_DIR})"
        )
    if root is None:
        record_root = RECORD_DIR / "root"
        root = env_path(
            "TRIZOD_DATA_ROOT", record_root if record_root.is_dir() else CHECKOUT
        )
    root = Path(root).expanduser().resolve()
    L = paths.layout(root)
    build = env_path("TRIZOD_BUILD_DIR", L.interim_build)
    testset = build / "testset"
    return Build(
        root=root,
        build=build,
        release=env_path("TRIZOD_RELEASE_DIR", RECORD_DIR / "release"),
        bundle=env_path("TRIZOD_BUNDLE", build / "release_bundle" / RECORD_BUNDLE),
        log=env_path("TRIZOD_REBUILD_LOG", RECORD_DIR / RECORD_LOG),
        pkl=env_path("TRIZOD_PKL_DIR", L.pkl_dir),
        draw=env_path(
            "TRIZOD_DRAW_DIR", _draw_dir(testset, L.interim_build / "testset")
        ),
    )


def _draw_dir(*candidates: Path) -> Path:
    """The first test-set directory that holds the 2026-08-11 draw (``_tmp/``): the build's,
    then the data root's, else the checkout's (where the draw was made)."""
    for d in candidates:
        if (d / "_tmp").is_dir():
            return d
    return CHECKOUT / "data" / "interim" / "build" / "testset"


def read_manifest(b: Build) -> dict:
    """The bundle's ``MANIFEST.json`` (``{}`` when it cannot be read)."""
    try:
        return json.loads(b.manifest.read_text())
    except (OSError, ValueError):
        return {}


# ----------------------------------------------------------------------------------
# input digests and keyed caches
# ----------------------------------------------------------------------------------
def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_DIGESTS: dict = {}


def snapshot_digest(raw_dir: Path) -> str:
    """Digest of the raw BMRB snapshot: its ``BMRB_SNAPSHOT.json`` (retrieval record and
    entry list), else the sorted entry-directory listing."""
    raw_dir = Path(raw_dir)
    k = ("snapshot", str(raw_dir))
    if k not in _DIGESTS:
        rec = raw_dir / "BMRB_SNAPSHOT.json"
        if rec.is_file():
            _DIGESTS[k] = file_sha256(rec)
        else:
            names = sorted(os.listdir(raw_dir)) if raw_dir.is_dir() else []
            _DIGESTS[k] = _sha("\n".join(names).encode())
    return _DIGESTS[k]


def build_digest(b: Build) -> str:
    """Digest of the build: its ``pipeline_version`` plus the bytes of ``MANIFEST.json``
    (which lists the checksums of the release files)."""
    k = ("build", str(b.manifest))
    if k not in _DIGESTS:
        sha = file_sha256(b.manifest) if b.manifest.is_file() else _sha(b"")
        pv = read_manifest(b).get("pipeline_version")
        _DIGESTS[k] = f"{pv}:{sha}"
    return _DIGESTS[k]


def code_digest(*objs) -> str:
    """Digest of the source of functions/classes/modules (``repr`` for other objects), so a
    change in the code that computes a cache invalidates it (no hand-bumped versions)."""
    h = hashlib.sha256()
    for o in objs:
        try:
            s = inspect.getsource(o)
        except (TypeError, OSError):
            s = repr(o)
        h.update(s.encode())
        h.update(b"\0")
    return h.hexdigest()


def input_key(inputs: dict) -> str:
    """Short digest of ``inputs`` (any JSON-serialisable mapping; paths become strings)."""
    return _sha(json.dumps(inputs, sort_keys=True, default=str).encode())[:16]


def read_keyed(path: Path, inputs: dict):
    """The data cached at ``path`` when it was written for exactly ``inputs``, else None."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except ValueError:  # a cache another process is still writing
        return None
    if not isinstance(d, dict) or d.get("_key") != input_key(inputs):
        return None
    return d.get("data")


def write_keyed(path: Path, inputs: dict, data) -> None:
    """Write ``data`` to ``path`` with the key of ``inputs`` (atomic: a concurrent reader
    never sees half a file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(path, "w") as fh:
        json.dump(
            {"_key": input_key(inputs), "_inputs": inputs, "data": data},
            fh,
            default=str,
        )


def keyed_json(
    path: Path, inputs: dict, compute: Callable[[], Any], refresh: bool = False
):
    """``compute()``, cached as JSON at ``path`` and reused only for the same ``inputs``.

    ``inputs`` names everything the result depends on (snapshot / build digests, a code
    digest, the command or arguments); the cache file stores them next to the data, so a
    stale cache can be told apart by eye too."""
    if not refresh:
        hit = read_keyed(path, inputs)
        if hit is not None:
            return hit
    t = time.time()
    out = compute()
    write_keyed(path, inputs, out)
    print(f"  computed {Path(path).name} in {time.time() - t:.0f} s", file=sys.stderr)
    return out
