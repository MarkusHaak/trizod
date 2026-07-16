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

# Source modules whose contents determine the numeric content of cached
# scoring artifacts. If any of these changes, cached wSCS arrays are stale.
_MATH_SOURCES = (
    "lacs/lacs.py",
    "scoring/scoring.py",
    "potenci/potenci.py",
    "constants.py",
)

_PKG_ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def scoring_cache_version() -> str:
    """Return a short hash of the scoring-math source.

    Changes whenever any module in ``_MATH_SOURCES`` changes, so it can be
    embedded in a cache key to force recomputation after a math change.
    Falls back to ``"nover"`` if the sources cannot be read (e.g. a zipped
    install); a stable-but-uninformative tag is safer than crashing the
    pipeline over a cache-key optimisation.
    """
    h = hashlib.sha256()
    try:
        for rel in _MATH_SOURCES:
            h.update(rel.encode())
            h.update((_PKG_ROOT / rel).read_bytes())
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
