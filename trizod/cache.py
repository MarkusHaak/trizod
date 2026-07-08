"""POTENCI prediction cache.

Content-addressed JSON cache for POTENCI random-coil predictions, keyed by
sequence + experimental conditions (temperature, pH, ionic strength). Extracted
from ``trizod.trizod`` so the cache can be used without importing the CLI module;
``trizod.trizod`` re-exports these names for backward compatibility.
"""

import hashlib
import json
import logging


def _potenci_cache_key(seq, temperature, pH, ion):
    """Content-based cache key for POTENCI predictions."""
    raw = f"{seq}|{temperature}|{pH}|{ion}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_potenci_cache(cache_dir, seq, temperature, pH, ion):
    """Load cached POTENCI predictions if available."""
    if not cache_dir:
        return None
    cache_path = (
        cache_dir / "potenci" / f"{_potenci_cache_key(seq, temperature, pH, ion)}.json"
    )
    if cache_path.exists():
        try:
            with cache_path.open() as f:
                raw = json.load(f)
            # JSON keys are strings — convert back to (int, str) tuples
            return {(int(k.split(",")[0]), k.split(",")[1]): v for k, v in raw.items()}
        except Exception:
            logging.getLogger("trizod").debug(
                f"POTENCI cache file {cache_path} corrupt, ignoring"
            )
    return None


def save_potenci_cache(cache_dir, seq, temperature, pH, ion, predshiftdct):
    """Save POTENCI predictions to cache."""
    if not cache_dir:
        return
    cache_path = (
        cache_dir / "potenci" / f"{_potenci_cache_key(seq, temperature, pH, ion)}.json"
    )
    # Convert (int, str) tuple keys to strings for JSON
    raw = {f"{k[0]},{k[1]}": v for k, v in predshiftdct.items()}
    with cache_path.open("w") as f:
        json.dump(raw, f)
