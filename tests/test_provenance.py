"""Provenance + determinism regression (issue #20).

Covers the two defensive guarantees added after issue #20:

* the scoring-code version is stable within a build and embeddable in the
  wSCS cache key, so a math change cannot be silently masked by a stale cache;
* the LACS offset computation is bit-reproducible for identical input, i.e.
  the reported "non-determinism" is not intrinsic to the algorithm.
"""

import re

import numpy as np

from trizod.lacs import compute_lacs_offsets
from trizod.provenance import git_revision, scoring_cache_version

from .test_lacs import _SEQ, _SEQ_NUMS, _generate_shifts


def test_scoring_cache_version_is_stable_hex():
    v = scoring_cache_version()
    assert isinstance(v, str)
    # 12-char hex digest, or the documented "nover" fallback.
    assert v == "nover" or re.fullmatch(r"[0-9a-f]{12}", v)
    # Deterministic within a process (also exercises the lru_cache).
    assert scoring_cache_version() == v


def test_git_revision_is_nonempty_string():
    rev = git_revision()
    assert isinstance(rev, str) and rev


def test_lacs_offsets_bit_reproducible():
    """Same input -> byte-identical offsets across many repeats.

    Issue #20 hypothesised BLAS-threading non-determinism inside the LACS
    robust fit. This asserts the offsets are exactly reproducible so any real
    future regression (an unseeded RNG, an order-dependent reduction) is caught.
    """
    seq, seq_nums = _SEQ, _SEQ_NUMS
    obs = _generate_shifts(seq, secondary_noise_std=1.0, rng=np.random.default_rng(0))
    first = compute_lacs_offsets(seq, seq_nums, obs)
    for _ in range(25):
        again = compute_lacs_offsets(seq, seq_nums, obs)
        assert again.keys() == first.keys()
        for atom, val in first.items():
            assert again[atom] == val, f"{atom} offset not reproducible"
