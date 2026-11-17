"""Step 1: extract {1H}-15N heteronuclear NOE and order-parameter (S2) data from
the local BMRB NMR-STAR v3 snapshot (no downloads).

Writes out/hetnoe_raw.parquet, out/s2_raw.parquet, out/lists_meta.parquet,
out/counts_extract.json.
"""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path

# Data root: the checkout that holds data/ (set TRIZOD_ROOT when running from a worktree).
ROOT = _Path(
    _os.environ.get("TRIZOD_ROOT")
    or next(
        p
        for p in _Path(__file__).resolve().parents
        if (p / "data" / "interim").is_dir()
    )
)

import json
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pynmrstar

BMRB_DIR = Path(f"{ROOT}/data/raw/bmrb_entries_2026-11-11")
OUT = Path(f"{ROOT}/data/interim/threshold-calibration-hetnoe-2026-11-16/out")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def cond_map(entry):
    """sample_conditions list ID -> {pH, T(K)}"""
    out = {}
    for sf in entry.get_saveframes_by_category("sample_conditions"):
        sid = sf.get_tag("ID")[0] if sf.get_tag("ID") else None
        d = {"pH": float("nan"), "T": float("nan")}
        for lp in sf.loops:
            if lp.category != "_Sample_condition_variable":
                continue
            for row in lp.get_tag(["Type", "Val", "Val_units"]):
                t, v, u = (str(row[0]).lower(), row[1], str(row[2]).lower())
                if t == "ph" and d["pH"] != d["pH"]:
                    d["pH"] = fnum(v)
                elif t == "temperature" and d["T"] != d["T"]:
                    val = fnum(v)
                    if u in ("c", "°c", "degc", "celsius") or (
                        val == val and val < 150
                    ):
                        val = val + 273.15
                    d["T"] = val
        out[sid] = d
    return out


def first(sf, tag):
    v = sf.get_tag(tag)
    return v[0] if v else None


def parse(entry_dir):
    eid = entry_dir.name[3:]
    f = entry_dir / f"bmr{eid}_3.str"
    try:
        txt = f.read_text(errors="replace")
    except OSError:
        return None
    has_noe = "heteronucl_NOEs" in txt
    has_s2 = "order_parameters" in txt
    if not (has_noe or has_s2):
        return None
    try:
        entry = pynmrstar.Entry.from_string(txt)
    except Exception as exc:  # noqa: BLE001
        return {"entry_id": eid, "error": str(exc)[:200]}
    conds = cond_map(entry)
    lists, noe, s2 = [], [], []
    for cat, kind in (("heteronucl_NOEs", "noe"), ("order_parameters", "s2")):
        for sf in entry.get_saveframes_by_category(cat):
            lid = first(sf, "ID")
            scid = first(sf, "Sample_condition_list_ID")
            c = conds.get(scid, {"pH": float("nan"), "T": float("nan")})
            meta = {
                "entry_id": eid,
                "kind": kind,
                "list_id": lid,
                "sf": sf.name,
                "cond_id": scid,
                "pH": c["pH"],
                "T": c["T"],
                "field_1H_MHz": fnum(first(sf, "Spectrometer_frequency_1H")),
                "val_type": first(sf, "Heteronuclear_NOE_val_type")
                if kind == "noe"
                else None,
                "ref_val": first(sf, "NOE_ref_val") if kind == "noe" else None,
                "ref_desc": first(sf, "NOE_ref_description") if kind == "noe" else None,
                "details": first(sf, "Details"),
            }
            n = 0
            for lp in sf.loops:
                if kind == "noe" and lp.category == "_Heteronucl_NOE":
                    tags = [
                        "Entity_assembly_ID_1",
                        "Entity_ID_1",
                        "Comp_index_ID_1",
                        "Seq_ID_1",
                        "Comp_ID_1",
                        "Atom_ID_1",
                        "Atom_ID_2",
                        "Val",
                        "Val_err",
                    ]
                    have = [t for t in tags if t in lp.tags]
                    for row in lp.get_tag(have):
                        r = dict(zip(have, row))
                        noe.append(
                            {
                                "entry_id": eid,
                                "list_id": lid,
                                "ea": r.get("Entity_assembly_ID_1"),
                                "ent": r.get("Entity_ID_1"),
                                "comp_index": r.get("Comp_index_ID_1"),
                                "seq_id": r.get("Seq_ID_1"),
                                "comp": r.get("Comp_ID_1"),
                                "atom1": r.get("Atom_ID_1"),
                                "atom2": r.get("Atom_ID_2"),
                                "val": fnum(r.get("Val")),
                                "err": fnum(r.get("Val_err")),
                            }
                        )
                        n += 1
                if kind == "s2" and lp.category == "_Order_param":
                    tags = [
                        "Entity_assembly_ID",
                        "Entity_ID",
                        "Comp_index_ID",
                        "Seq_ID",
                        "Comp_ID",
                        "Atom_ID",
                        "Order_param_val",
                        "Order_param_val_fit_err",
                    ]
                    have = [t for t in tags if t in lp.tags]
                    for row in lp.get_tag(have):
                        r = dict(zip(have, row))
                        s2.append(
                            {
                                "entry_id": eid,
                                "list_id": lid,
                                "ea": r.get("Entity_assembly_ID"),
                                "ent": r.get("Entity_ID"),
                                "comp_index": r.get("Comp_index_ID"),
                                "seq_id": r.get("Seq_ID"),
                                "comp": r.get("Comp_ID"),
                                "atom1": r.get("Atom_ID"),
                                "val": fnum(r.get("Order_param_val")),
                                "err": fnum(r.get("Order_param_val_fit_err")),
                            }
                        )
                        n += 1
            meta["n_rows"] = n
            lists.append(meta)
    return {"entry_id": eid, "lists": lists, "noe": noe, "s2": s2}


def main():
    dirs = sorted(
        p for p in BMRB_DIR.iterdir() if p.is_dir() and p.name.startswith("bmr")
    )
    with Pool(12) as p:
        res = [r for r in p.imap_unordered(parse, dirs, chunksize=50) if r]
    errs = [r for r in res if "error" in r]
    lists = pd.DataFrame([m for r in res if "lists" in r for m in r["lists"]])
    noe = pd.DataFrame([x for r in res if "noe" in r for x in r["noe"]])
    s2 = pd.DataFrame([x for r in res if "s2" in r for x in r["s2"]])
    for d in (lists, noe, s2):
        for c in d.columns:
            if d[c].dtype == object:
                d[c] = d[c].astype(str)
    lists.to_parquet(OUT / "lists_meta.parquet")
    noe.to_parquet(OUT / "hetnoe_raw.parquet")
    s2.to_parquet(OUT / "s2_raw.parquet")
    counts = {
        "entries_scanned": len(dirs),
        "parse_errors": len(errs),
        "error_entries": [e["entry_id"] for e in errs],
        "hetnoe_entries": int(lists[lists.kind == "noe"].entry_id.nunique()),
        "hetnoe_lists": int((lists.kind == "noe").sum()),
        "hetnoe_rows": len(noe),
        "s2_entries": int(lists[lists.kind == "s2"].entry_id.nunique()),
        "s2_lists": int((lists.kind == "s2").sum()),
        "s2_rows": len(s2),
    }
    (OUT / "counts_extract.json").write_text(json.dumps(counts, indent=1))
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
