"""Light-weight scan of the raw BMRB NMR-STAR files for the paper's Supplementary figures.

The release Parquet does not (yet) carry three per-entry facts the Supplementary
figures need, so they are read straight from ``data/raw/bmrb_entries/bmr*/bmr*_3.str``:

* ``_Entry.Submission_date`` -- the deposition date (Supplementary Fig. S5);
* the sample-condition lists, resolved per shift table the way pH and temperature are:
  ``_Assigned_chem_shift_list.Sample_condition_list_ID`` points at a
  ``_Sample_condition_list`` whose ``_Sample_condition_variable`` loop holds the
  pressure / pH / temperature / ionic-strength rows (Supplementary Fig. S4; fix (e) of
  the writing plan will add ``pressure_bar`` to the Parquet);
* ``_Entity_natural_src.Organism_name_scientific`` per entity (a heuristic parse;
  approximate -- Supplementary Fig. S5).

The scan is a text parse (no pynmrstar) over ~3 GB, run in parallel and cached as JSON in
``<repo>/tmp/paper_figures/raw_scan.json`` (``tmp*/`` is git-ignored). Delete the cache or
pass ``refresh=True`` to recompute. It never imports pyarrow.

"Deposited" -- the paper's ONE definition (12 Nov 2026). :func:`chain_deposited` says
whether a chain's own condition list deposits pH, temperature and ionic strength, read
with the PIPELINE's parser: the condition rows go through
``trizod.bmrb.bmrb.SampleConditions`` and its getters with ``return_default=False``, so a
value counts as deposited exactly when the release holds the deposited value and not a
default (pH 7.0, 298 K, 0.1 M). The scan's own numbers (``cond[...]["ph"/"temp"/"ionic"]``)
read a value only when ``float()`` accepts it and miss values such as '298K', '293,15' or
'298-303' (11 temperatures and one ionic strength of the 2026-11 release); they stay for
pressure, but every "deposited" count of the paper (Fig. 2a/b, Supplementary Fig. S4, the
condition comparison of Fig. 4a and the ``\\n*NotDeposited*`` macros) uses
:func:`chain_deposited`.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style_paper as sp  # noqa: E402

CACHE = sp.TMP / "paper_figures" / "raw_scan.json"
SCAN_VERSION = 3  # 3: pH*/pD rows count as a deposited pH

_FRAME_SPLIT = re.compile(r"(?m)^[ \t]*save_\S*[ \t]*$")
_SUB = re.compile(r"_Entry\.Submission_date\s+(\S+)")
_COND_ID = re.compile(r"_Sample_condition_list\.ID\s+(\S+)")
_ST_ID = re.compile(r"_Assigned_chem_shift_list\.ID\s+(\S+)")
_ST_COND = re.compile(r"_Assigned_chem_shift_list\.Sample_condition_list_ID\s+(\S+)")

# pressure units -> factor to bar
_TO_BAR = {
    "bar": 1.0,
    "bars": 1.0,
    "atm": 1.01325,
    "mpa": 10.0,
    "kpa": 0.01,
    "pa": 1e-5,
    "mbar": 1e-3,
    "mmhg": 0.00133322,
    "torr": 0.00133322,
    "psi": 0.0689476,
}


def to_bar(val, unit):
    """Numeric pressure in bar, or None when the value or unit cannot be read."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    f = _TO_BAR.get(str(unit).lower().strip("'\" "))
    return None if f is None else v * f


def _loop_rows(chunk: str, category: str):
    """Rows of the first ``loop_`` of ``category`` in a saveframe, as dicts (tag -> value)."""
    i = chunk.find(f"_{category}.")
    while i >= 0:
        lp = chunk.rfind("loop_", 0, i)
        if lp < 0:
            return []
        end = chunk.find("stop_", i)
        block = chunk[lp + 5 : end if end >= 0 else len(chunk)]
        hdr = re.findall(rf"^\s*_{category}\.(\S+)", block, re.M)
        if not hdr:
            return []
        # the body starts after the last header line
        last = re.search(rf"^\s*_{category}\.{re.escape(hdr[-1])}\s*$", block, re.M)
        body = block[last.end() :] if last else ""
        try:
            toks = shlex.split(body, posix=True)
        except ValueError:
            return []
        n = len(hdr)
        if not toks or len(toks) % n:
            return []
        return [
            dict(zip(hdr, toks[r * n : (r + 1) * n])) for r in range(len(toks) // n)
        ]
    return []


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def scan_file(path: str) -> tuple[str, dict]:
    txt = Path(path).read_text(errors="ignore")
    eid = Path(path).parent.name.replace("bmr", "")
    rec: dict = {"sub": None, "cond": {}, "st": {}, "org": {}}
    m = _SUB.search(txt)
    rec["sub"] = m.group(1) if m else None
    for chunk in _FRAME_SPLIT.split(txt):
        if "_Sample_condition_list.ID" in chunk:
            m = _COND_ID.search(chunk)
            if not m:
                continue
            c = {
                "pressure_raw": None,
                "pressure_bar": None,
                "ph": None,
                "temp": None,
                "ionic": None,
            }
            for row in _loop_rows(chunk, "Sample_condition_variable"):
                t = row.get("Type", "").lower()
                val, unit = row.get("Val"), row.get("Val_units", "")
                if t == "pressure":
                    c["pressure_raw"] = [val, unit]
                    c["pressure_bar"] = to_bar(val, unit)
                elif t in ("ph", "ph*", "pd", "pd*"):
                    # pH* (meter reading in D2O) and pD count as a deposited pH, as in
                    # the parser; a plain pH row wins when an entry has both
                    if t == "ph" or c["ph"] is None:
                        c["ph"] = _num(val)
                elif "ionic strength" in t:
                    c["ionic"] = _num(val)
                elif "temp" in t:
                    c["temp"] = _num(val)
            rec["cond"][m.group(1)] = c
        if "_Assigned_chem_shift_list.ID" in chunk:
            m, mc = _ST_ID.search(chunk), _ST_COND.search(chunk)
            if m:
                rec["st"][m.group(1)] = mc.group(1) if mc else None
        if "_Entity_natural_src.Organism_name_scientific" in chunk:
            for row in _loop_rows(chunk, "Entity_natural_src"):
                rec["org"][row.get("Entity_ID", "1")] = row.get(
                    "Organism_name_scientific"
                )
    return eid, rec


def scan_raw(refresh: bool = False, workers: int | None = None) -> dict:
    """``{entry_id: {sub, cond: {list_id: {...}}, st: {st_id: list_id}, org: {entity: name}}}``."""
    if CACHE.exists() and not refresh:
        d = json.loads(CACHE.read_text())
        if d.get("_version") == SCAN_VERSION:
            return d["entries"]
    files = []
    for d in sorted(os.listdir(sp.RAW_BMRB)):
        fs = sorted((sp.RAW_BMRB / d).glob("*.str"))
        if fs:
            files.append(str(fs[0]))
    t = time.time()
    with ProcessPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 2)) as ex:
        out = dict(ex.map(scan_file, files, chunksize=64))
    print(f"  raw scan: {len(out)} files in {time.time() - t:.0f} s", file=sys.stderr)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({"_version": SCAN_VERSION, "entries": out}))
    return out


def chain_conditions(raw: dict, entry_id: str, st_id: str) -> dict | None:
    """The sample-condition record behind one chain's shift table (None if unresolved)."""
    rec = raw.get(str(entry_id))
    if rec is None:
        return None
    lid = rec["st"].get(str(st_id))
    if lid is None:
        return None
    return rec["cond"].get(lid)


# ---------------------------------------------------------------- deposited (pipeline)
DEPOSITED = ("ph", "temperature", "ionic_strength")
DEPOSITED_CACHE = sp.TMP / "paper_figures" / "deposited_pipeline.json"
DEPOSITED_VERSION = 1


class _Saveframe:
    """The one method of a pynmrstar saveframe that ``bmrb.get_tag_vals`` calls."""

    def __init__(self, tags: dict):
        self._tags = tags

    def get_tag(self, tag):
        return self._tags[tag]


def _deposited_lists(task) -> tuple[str, dict]:
    """``(entry id, {condition list id: {condition: deposited}})`` for one raw file, read
    with the pipeline's ``SampleConditions`` (its row choice, unit handling and ranges)."""
    import logging

    import numpy as np

    from trizod.bmrb import bmrb as bm

    logging.getLogger("trizod.bmrb").setLevel(logging.CRITICAL)
    eid, path = task
    out = {}
    for chunk in _FRAME_SPLIT.split(Path(path).read_text(errors="ignore")):
        if "_Sample_condition_list.ID" not in chunk:
            continue
        m = _COND_ID.search(chunk)
        if not m:
            continue
        rows = _loop_rows(chunk, "Sample_condition_variable")
        sc = bm.SampleConditions(
            _Saveframe(
                {
                    "_Sample_condition_list.ID": [m.group(1)],
                    **{
                        f"_Sample_condition_variable.{k}": [r.get(k, ".") for r in rows]
                        for k in ("Type", "Val", "Val_units")
                    },
                }
            )
        )
        vals = (
            sc.get_pH(return_default=False),
            sc.get_temperature(return_default=False),
            sc.get_ionic_strength(return_default=False),
        )
        out[m.group(1)] = {
            c: v is not None and bool(np.isfinite(v)) for c, v in zip(DEPOSITED, vals)
        }
    return eid, out


def _parser_version() -> str:
    try:
        from trizod.provenance import bmrb_parser_version

        return bmrb_parser_version()
    except Exception:  # pragma: no cover - provenance is part of the package
        return "unknown"


def deposited_lists(
    raw: dict | None = None, refresh: bool = False, workers: int | None = None
) -> dict:
    """``{entry id: {condition list id: {"ph", "temperature", "ionic_strength": bool}}}``
    for every condition list that a shift table of the raw files points at, read with the
    pipeline's parser (module docstring). Cached as JSON, keyed by the parser version."""
    raw = scan_raw() if raw is None else raw
    pv = _parser_version()
    if DEPOSITED_CACHE.exists() and not refresh:
        try:
            d = json.loads(DEPOSITED_CACHE.read_text())
        except ValueError:  # a cache another process is still writing
            d = {}
        if d.get("_version") == DEPOSITED_VERSION and d.get("_parser") == pv:
            return d["entries"]
    tasks = []
    for eid, rec in raw.items():
        fs = sorted((sp.RAW_BMRB / f"bmr{eid}").glob("*.str"))
        if fs and any(v is not None for v in rec["st"].values()):
            tasks.append((eid, str(fs[0])))
    t = time.time()
    with ProcessPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 2)) as ex:
        out = dict(ex.map(_deposited_lists, tasks, chunksize=64))
    print(f"  deposited conditions: {len(out)} files in {time.time() - t:.0f} s", file=sys.stderr)  # fmt: skip
    DEPOSITED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEPOSITED_CACHE.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps({"_version": DEPOSITED_VERSION, "_parser": pv, "entries": out})
    )
    # atomic: a concurrent reader never sees half a file
    os.replace(tmp, DEPOSITED_CACHE)
    return out


def chain_deposited(entry_ids, st_ids, refresh: bool = False) -> list[dict]:
    """Per chain (``entry_ids[i]``, ``st_ids[i]``): ``{"ph", "temperature",
    "ionic_strength": bool}``, True when the chain's own condition list deposits a value the
    pipeline reads (the release then holds it, not a default). A chain whose condition list
    cannot be resolved counts as depositing nothing (none in the 2026-11 release)."""
    raw = scan_raw()
    dep = deposited_lists(raw, refresh=refresh)
    none = dict.fromkeys(DEPOSITED, False)
    out = []
    for e, s in zip(entry_ids, st_ids):
        lid = raw.get(str(e), {}).get("st", {}).get(str(s))
        out.append(dep.get(str(e), {}).get(lid, none) if lid is not None else none)
    return out


# ---------------------------------------------------------------- entry authors
AUTHORS_CACHE = sp.TMP / "paper_figures" / "entry_authors.json"


def _entry_author_rows(task) -> tuple[str, list]:
    """``(entry id, ["family|initial", ...])`` from the ``_Entry_author`` loop of one file:
    family name case-folded, plus the first letter of the given name (so that two
    depositors who only share a common family name are not one person)."""
    eid, path = task
    txt = Path(path).read_text(errors="ignore")
    out = []
    for row in _loop_rows(txt, "Entry_author"):
        fam = " ".join(str(row.get("Family_name", "")).split()).casefold()
        given = str(row.get("Given_name", "")).strip(". ")
        if fam and fam not in (".", "?"):
            out.append(f"{fam}|{given[:1].casefold() if given else ''}")
    return eid, sorted(set(out))


def entry_authors(entry_ids, workers: int | None = None) -> dict:
    """``{entry id: frozenset of "family|initial"}`` of the entries' depositors
    (``_Entry_author``), read from the raw files on demand and cached as JSON (entries are
    added to the cache; a deposited author list does not change within a snapshot)."""
    want = {str(e) for e in entry_ids}
    cache = {}
    if AUTHORS_CACHE.exists():
        try:
            cache = json.loads(AUTHORS_CACHE.read_text())
        except ValueError:
            cache = {}
    todo = []
    for e in sorted(want - set(cache)):
        fs = sorted((sp.RAW_BMRB / f"bmr{e}").glob("*.str"))
        if fs:
            todo.append((e, str(fs[0])))
        else:
            cache[e] = []
    if todo:
        with ProcessPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 2)) as ex:  # fmt: skip
            cache.update(dict(ex.map(_entry_author_rows, todo, chunksize=32)))
        AUTHORS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = AUTHORS_CACHE.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(cache))
        os.replace(tmp, AUTHORS_CACHE)
    return {e: frozenset(cache.get(e, [])) for e in want}


if __name__ == "__main__":
    r = scan_raw(refresh="--refresh" in sys.argv)
    print(len(r), "entries scanned")
