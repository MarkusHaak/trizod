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


if __name__ == "__main__":
    r = scan_raw(refresh="--refresh" in sys.argv)
    print(len(r), "entries scanned")
