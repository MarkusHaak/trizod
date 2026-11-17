"""Step 2: QC the extracted hetNOE / S2 values and join them to the release
Parquet per residue.

Same-entry join: hetNOE row (entry, Entity_assembly_ID_1, Entity_ID_1, Seq_ID_1)
-> chain (entry_id, entity_assem_id, entity_id); Seq_ID is 1-based on the entity
sequence; residue names must match. Among several shift lists (st_id) of that
entity, the one with the closest pH/T is used; a list is accepted if
|dpH| <= 0.5 and |dT| <= 5 K (or if a value is missing on either side).
Cross-entry join (robustness): hetNOE of an identical entity sequence from a
different entry, with both pH and T known and |dpH| <= 0.5, |dT| <= 5 K.

Chain filter = the one of the DisProt/X-ray calibration (chains_filtered.parquet
of threshold-calibration-2026-11-16, 10,887 chains), applied to all splits.
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
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(f"{ROOT}/data/interim/threshold-calibration-hetnoe-2026-11-16")
OUT = BASE / "out"
PARQUET = Path(
    _os.environ.get("TRIZOD_RELEASE_PARQUET")
    or f"{ROOT}/data/interim/rebuild-2026-11-11/release/trizod_dataset.parquet"
)
FILTERED = Path(
    f"{ROOT}/data/interim/threshold-calibration-2026-11-16/out/chains_filtered.parquet"
)
DPH, DT = 0.5, 5.0
TERM = 3  # residues from either sequence end counted as terminal

AA3 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def to_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def qc(raw, lists, kind, counts):
    c = {}
    c["rows_raw"] = len(raw)
    c["entries_raw"] = raw.entry_id.nunique()
    if kind == "noe":
        a1, a2 = raw.atom1.str.upper(), raw.atom2.str.upper()
        bb = ((a1 == "N") & a2.isin(["H", "HN"])) | ((a2 == "N") & a1.isin(["H", "HN"]))
        c["rows_atoms_unspecified"] = int(((a1 == ".") & (a2 == ".")).sum())
        c["rows_non_backbone"] = int((~bb & ~((a1 == ".") & (a2 == "."))).sum())
        raw = raw[bb].copy()
        c["rows_backbone_NH"] = len(raw)
        lo, hi, med_lo, med_hi = -2.0, 1.05, -1.0, 0.95
    else:
        raw = raw[raw.atom1.str.upper() == "N"].copy()
        c["rows_backbone_N"] = len(raw)
        lo, hi, med_lo, med_hi = 0.0, 1.05, 0.0, 1.0
    raw = raw[np.isfinite(raw.val)]
    g = raw.groupby(["entry_id", "list_id"]).val
    st = pd.DataFrame(
        {
            "med": g.median(),
            "inrange": g.apply(lambda v: ((v >= lo) & (v <= hi)).mean()),
        }
    )
    bad = st[(st.med < med_lo) | (st.med > med_hi) | (st.inrange < 0.9)]
    c["lists_before_value_qc"] = len(st)
    c["lists_dropped_value_qc"] = len(bad)
    c["lists_dropped_value_qc_ids"] = [f"{e}:{lid}" for e, lid in bad.index]
    raw = (
        raw.set_index(["entry_id", "list_id"])
        .drop(bad.index, errors="ignore")
        .reset_index()
    )
    n0 = len(raw)
    raw = raw[(raw.val >= lo) & (raw.val <= hi)]
    c["rows_dropped_out_of_range"] = n0 - len(raw)
    c["rows_after_qc"] = len(raw)
    c["lists_after_qc"] = int(raw.groupby(["entry_id", "list_id"]).ngroups)
    c["entries_after_qc"] = raw.entry_id.nunique()
    counts[kind]["qc"] = c
    return raw.merge(
        lists[lists.kind == kind][["entry_id", "list_id", "pH", "T", "field_1H_MHz"]],
        on=["entry_id", "list_id"],
        how="left",
    )


def map_list(sub, ch_rows):
    """Map one hetNOE list's rows (one entity) onto a chain sequence.
    Returns DataFrame(idx0, val, err) and the residue-name match rate."""
    seq = ch_rows["sequence"]
    best = None
    for col in ("seq_id", "comp_index"):
        pos = sub[col].map(to_int)
        ok = pos.notna()
        if ok.sum() == 0:
            continue
        p = pos[ok].astype(int) - 1
        inb = (p >= 0) & (p < len(seq))
        aa = sub.loc[ok, "comp"].str.upper().map(AA3)
        match = inb & np.array(
            [inb_ and seq[pp] == a for pp, a, inb_ in zip(p, aa, inb)]
        )
        rate = match.sum() / max(len(sub), 1)
        if best is None or rate > best[0]:
            best = (
                rate,
                pd.DataFrame(
                    {
                        "idx0": p[match].to_numpy(),
                        "val": sub.loc[ok].val[match].to_numpy(),
                        "err": sub.loc[ok].err[match].to_numpy(),
                    }
                ),
            )
    return best if best else (0.0, pd.DataFrame(columns=["idx0", "val", "err"]))


def cond_ok(ph1, t1, ph2, t2, require_both=False):
    dph = abs(ph1 - ph2) if np.isfinite(ph1) and np.isfinite(ph2) else np.nan
    dt = abs(t1 - t2) if np.isfinite(t1) and np.isfinite(t2) else np.nan
    if require_both and (not np.isfinite(dph) or not np.isfinite(dt)):
        return False, dph, dt
    ok = (not np.isfinite(dph) or dph <= DPH) and (not np.isfinite(dt) or dt <= DT)
    return ok, dph, dt


def join(qcd, rel, filt_ids, kind, counts):
    c = {}
    rel_e = rel.groupby("entry_id")
    # assign entity for rows lacking ea/ent when the entry has a single protein chain entity
    per_list = []
    for (eid, lid), sub in qcd.groupby(["entry_id", "list_id"]):
        if eid not in rel_e.groups:
            per_list.append(
                {"entry_id": eid, "list_id": lid, "status": "entry not in release"}
            )
            continue
        chains = rel_e.get_group(eid)
        sub = sub.copy()
        if (sub.ea == ".").all() or (sub.ent == ".").all():
            ents = chains[["entity_assem_id", "entity_id"]].drop_duplicates()
            if len(ents) == 1:
                sub["ea"], sub["ent"] = (
                    ents.iloc[0].entity_assem_id,
                    ents.iloc[0].entity_id,
                )
        for (ea, ent), s2 in sub.groupby(["ea", "ent"]):
            cand = chains[(chains.entity_assem_id == ea) & (chains.entity_id == ent)]
            rec = {
                "entry_id": eid,
                "list_id": lid,
                "ea": ea,
                "ent": ent,
                "n_rows": len(s2),
                "pH": s2.pH.iloc[0],
                "T": s2["T"].iloc[0],
                "field": s2.field_1H_MHz.iloc[0],
            }
            if cand.empty:
                rec["status"] = "no release chain for entity"
                per_list.append(rec)
                continue
            cand = cand.assign(
                dph=(cand.ph - rec["pH"]).abs(), dt=(cand.temperature - rec["T"]).abs()
            )
            cand = cand.assign(
                score=cand.dph.fillna(0) / DPH + cand.dt.fillna(0) / DT
            ).sort_values(["score", "st_id"])
            row = cand.iloc[0]
            ok, dph, dt = cond_ok(rec["pH"], rec["T"], row.ph, row.temperature)
            rate, m = map_list(s2, row)
            rec.update(
                {
                    "chain": row.id,
                    "dpH": dph,
                    "dT": dt,
                    "cond_ok": ok,
                    "name_match": rate,
                    "nmapped_dfped": len(m),
                    "in_filter": row.id in filt_ids,
                    "sequence": row.sequence,
                }
            )
            if rate < 0.9:
                rec["status"] = "residue names mismatch (<90%)"
            elif not ok:
                rec["status"] = "conditions differ"
            elif not rec["in_filter"]:
                rec["status"] = "chain removed by chain filter"
            else:
                rec["status"] = "ok"
            rec["mapped_df"] = m
            per_list.append(rec)
    pl = pd.DataFrame(per_list)
    c["list_entity_status"] = pl.status.value_counts().to_dict()
    # residue table: all lists that mapped with good names (any filter/condition flag kept for audits)
    rows = []
    for r in pl.itertuples():
        if isinstance(getattr(r, "mapped_df", None), pd.DataFrame) and r.status in (
            "ok",
            "chain removed by chain filter",
            "conditions differ",
        ):
            m = r.mapped_df.assign(
                chain=r.chain,
                entry_id=r.entry_id,
                list_id=r.list_id,
                field=r.field,
                status=r.status,
                pH=r.pH,
                T=r.T,
            )
            rows.append(m)
    res = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    pl = pl.drop(columns=["mapped_df"])
    return pl, res, c


def main():
    counts = {"noe": {}, "s2": {}}
    lists = pd.read_parquet(OUT / "lists_meta.parquet")
    rel = pd.read_parquet(
        PARQUET,
        columns=[
            "id",
            "entry_id",
            "entity_id",
            "entity_assem_id",
            "st_id",
            "sequence",
            "gscores",
            "mask",
            "k",
            "split",
            "cluster_repr",
            "ph",
            "temperature",
        ],
    )
    filt = pd.read_parquet(FILTERED)
    filt_ids = set(filt.id)
    # cluster key: cluster_repr; else that of a train/redundant chain with identical sequence; else the sequence
    seq2cl = (
        rel.dropna(subset=["cluster_repr"])
        .drop_duplicates("sequence")
        .set_index("sequence")
        .cluster_repr
    )
    rel["cluster"] = rel.cluster_repr.fillna(rel.sequence.map(seq2cl)).fillna(
        "seq:" + rel.sequence
    )
    gmap = rel.set_index("id")
    for kind, fn in (("noe", "hetnoe_raw.parquet"), ("s2", "s2_raw.parquet")):
        raw = pd.read_parquet(OUT / fn)
        qcd = qc(raw, lists, kind, counts)
        pl, res, c = join(qcd, rel, filt_ids, kind, counts)
        counts[kind]["join"] = c
        pl.drop(columns=["sequence"], errors="ignore").to_csv(
            OUT / f"{kind}_list_join_status.csv", index=False
        )

        def attach(df):
            G, M, K, L = [], [], [], []
            for ch, i in zip(df.chain, df.idx0):
                r = gmap.loc[ch]
                G.append(r.gscores[i])
                M.append(bool(r["mask"][i]))
                K.append(int(r.k[i]))
                L.append(len(r.sequence))
            return df.assign(
                G=G,
                mask=M,
                k=K,
                length=L,
                terminal=[(i < TERM) or (i >= n - TERM) for i, n in zip(df.idx0, L)],
                split=df.chain.map(gmap.split),
                cluster=df.chain.map(gmap.cluster),
                sequence=df.chain.map(gmap.sequence),
            )

        res = attach(res)
        res["source"] = "same_entry"
        # cross-entry: identical entity sequence, other entry, both conditions known and within limits
        ok_lists = pl[
            pl.status.isin(["ok", "chain removed by chain filter", "conditions differ"])
        ]
        src = res.drop_duplicates(["entry_id", "list_id", "chain"])[
            ["entry_id", "list_id", "chain", "pH", "T", "field", "sequence"]
        ]
        fch = rel[rel.id.isin(filt_ids)]
        cross = []
        bys = fch.groupby("sequence")
        for s in src.itertuples():
            if s.sequence not in bys.groups:
                continue
            for t in bys.get_group(s.sequence).itertuples():
                if t.entry_id == s.entry_id:
                    continue
                ok, dph, dt = cond_ok(s.pH, s.T, t.ph, t.temperature, require_both=True)
                if not ok:
                    continue
                vals = res[
                    (res.entry_id == s.entry_id)
                    & (res.list_id == s.list_id)
                    & (res.chain == s.chain)
                ][["idx0", "val", "err"]]
                cross.append(
                    vals.assign(
                        chain=t.id,
                        entry_id=s.entry_id,
                        list_id=s.list_id,
                        field=s.field,
                        status="ok",
                        pH=s.pH,
                        T=s.T,
                        dpH=dph,
                        dT=dt,
                    )
                )
        cr = attach(pd.concat(cross, ignore_index=True)) if cross else pd.DataFrame()
        if len(cr):
            cr["source"] = "cross_entry"
        allres = pd.concat([res, cr], ignore_index=True)
        allres["G"] = allres.G.astype(float)
        allres.to_parquet(OUT / f"{kind}_residues_all.parquet")
        c2 = {}
        se = res[res.status == "ok"]
        c2["same_entry_ok_lists"] = int(
            se.groupby(["entry_id", "list_id", "chain"]).ngroups
        )
        c2["same_entry_ok_chains"] = int(se.chain.nunique())
        c2["same_entry_ok_residues_mapped"] = len(se)
        sc = se[se["mask"] & np.isfinite(se.G)]
        c2["same_entry_ok_residues_scored"] = len(sc)
        c2["same_entry_ok_chains_scored"] = int(sc.chain.nunique())
        c2["same_entry_ok_clusters_scored"] = int(sc.cluster.nunique())
        c2["same_entry_chains_by_split"] = (
            sc.drop_duplicates("chain").split.value_counts().to_dict()
        )
        if len(cr):
            cs = cr[cr["mask"] & np.isfinite(cr.G.astype(float))]
            c2["cross_entry_pairs"] = int(
                cs.groupby(["chain", "entry_id", "list_id"]).ngroups
            )
            c2["cross_entry_chains_scored"] = int(cs.chain.nunique())
            c2["cross_entry_chains_without_same_entry"] = int(
                len(set(cs.chain) - set(sc.chain))
            )
            c2["cross_entry_residues_scored"] = len(cs)
        counts[kind]["joined"] = c2
        print(kind, json.dumps(counts[kind], indent=1, default=str)[:3000], flush=True)
    (OUT / "counts_join.json").write_text(json.dumps(counts, indent=1, default=str))


if __name__ == "__main__":
    main()
