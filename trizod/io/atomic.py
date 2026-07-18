"""Atomic file writes.

Write to a unique temp sibling and ``os.replace`` it into place on clean exit,
so an interrupted or concurrent write can never leave a half-written file that a
later run would read as valid (issue #20). On error the temp file is removed
rather than left behind.
"""

import os
from contextlib import contextmanager


@contextmanager
def atomic_write(path, mode="wb"):
    """Yield a handle to a temp file that is atomically renamed to ``path``.

    Usage mirrors ``path.open(mode)``; both text and binary consumers work
    (``json.dump(obj, fh)`` and ``np.savez(fh, ...)`` alike):

        with atomic_write(path, "w") as fh:
            json.dump(obj, fh)
    """
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open(mode) as fh:
            yield fh
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
