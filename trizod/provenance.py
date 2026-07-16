"""Provenance helpers: scoring-code version + git revision.

Two concerns, both aimed at making scored artifacts reproducible and
traceable:

* ``scoring_cache_version()`` — a short digest of the *source* of the modules
  that produce cached scoring artifacts (LACS, scoring, shared constants).
  It is folded into the ``tmp/wSCS/`` cache key so that any change to the
  scoring math invalidates stale cache entries by construction, rather than
  relying on a human to remember to bump a version. Over-invalidation
  (e.g. a comment-only edit busting the cache) is deliberately preferred over
  under-invalidation: the cache is a speed optimisation, but silently reusing
  offsets computed by different code is a correctness bug (see issue #20).

* ``git_revision()`` — the current commit, stamped into release/manifest
  metadata so a ``scores.json`` can always be traced back to the code that
  produced it.
"""

import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path

# Subpackages + modules whose contents determine the numeric content of cached
# scoring artifacts. Whole directories are hashed (not a filename allowlist) so
# that adding or splitting a module inside them still moves the version — under-
# invalidation would silently reuse offsets from different code, the very bug
# this guards against, whereas over-invalidation only costs a recompute.
_MATH_DIRS = ("lacs", "scoring", "potenci")
_MATH_FILES = ("constants.py",)

_PKG_ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def scoring_cache_version() -> str:
    """Return a short hash of the scoring-math source.

    Changes whenever any ``.py`` under the scoring subpackages (or the shared
    constants) changes, so it can be embedded in a cache key to force
    recomputation after a math change. Falls back to ``"nover"`` if the sources
    cannot be read (e.g. a zipped install); a stable-but-uninformative tag is
    safer than crashing the pipeline over a cache-key optimisation.
    """
    paths = [p for d in _MATH_DIRS for p in (_PKG_ROOT / d).rglob("*.py")]
    paths += [_PKG_ROOT / f for f in _MATH_FILES]
    h = hashlib.sha256()
    try:
        for p in sorted(paths):
            h.update(str(p.relative_to(_PKG_ROOT)).encode())
            h.update(p.read_bytes())
    except OSError:
        return "nover"
    return h.hexdigest()[:12]


@lru_cache(maxsize=1)
def git_revision() -> str:
    """Return the short git commit hash of the working tree, or ``"unknown"``.

    Appends ``"-dirty"`` when the working tree has uncommitted changes, so a
    build from modified sources is never silently labelled as a clean commit.
    """
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=_PKG_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    try:
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"],
            cwd=_PKG_ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError):
        dirty = 0
    return f"{sha}-dirty" if dirty else sha


def pipeline_version() -> str:
    """Full ``pipeline_version`` label stamped into scored/release metadata.

    Owns the ``trizod-<revision>`` format in one place so the two stamp sites
    (scoring output and the release manifest) cannot drift.
    """
    return f"trizod-{git_revision()}"
