"""Walk BMRB entries and yield every deposited shift, per chain.

:func:`trizod.bmrb.bmrb.get_deposited_shifts` reads one chain; the helpers here
apply it across a set of chain IDs (the ``id`` column of the released dataset
Parquet, ``entryID_stID_entity_assemID_entityID``) and stream the result in
``id`` order so the released table can be written without holding ~12 M rows in
memory.
"""

from __future__ import annotations

import logging
import pickle
from itertools import groupby
from pathlib import Path

from trizod.bmrb.bmrb import get_deposited_shifts

SHIFTS_PARQUET_NAME = "trizod_shifts.parquet"

log = logging.getLogger("trizod.shifts")


def chain_id(entry_id, stID, entity_assemID, entityID) -> str:
    """The dataset's chain key, identical to ``output_dataset()``'s ``ID``."""
    return f"{entry_id}_{stID}_{entity_assemID}_{entityID}"


def entry_chains(entry, keep_ids=None):
    """Yield ``(chain_id, seq, shifts)`` per polypeptide chain of ``entry``."""
    for (stID, ea_id, e_id), (shifts, _, _, _) in entry.get_peptide_shifts().items():
        cid = chain_id(entry.id, stID, ea_id, e_id)
        if keep_ids is not None and cid not in keep_ids:
            continue
        seq = entry.entities[e_id].seq
        if not seq:
            continue
        yield cid, seq, shifts


def entry_shift_frames(entry, keep_ids=None):
    """Yield ``(chain_id, DataFrame)`` for every polypeptide chain of ``entry``.

    Chains whose shift table fails the sequence-consistency guards are skipped —
    the same chains the backbone read rejects.
    """
    for cid, seq, shifts in entry_chains(entry, keep_ids):
        df = get_deposited_shifts(shifts, seq)
        if df is None or df.empty:
            continue
        yield cid, df


def load_entry(entry_id: str, pkl_dir: Path, bmrb_dir: Path | None = None):
    """Load one ``BmrbEntry`` from the pickle cache, re-parsing if necessary."""
    pkl = Path(pkl_dir) / f"{entry_id}.pkl"
    if pkl.exists():
        with pkl.open("rb") as fh:
            return pickle.load(fh)
    if bmrb_dir is None:
        raise FileNotFoundError(f"no cached entry {pkl} and no bmrb_dir to parse from")
    from trizod.bmrb.bmrb import BmrbEntry

    return BmrbEntry(entry_id, Path(bmrb_dir) / f"bmr{entry_id}")


def iter_chains(ids, pkl_dir, bmrb_dir=None, progress_every=2000):
    """Yield ``(chain_id, seq, shifts)`` for ``ids``, in sorted ``id`` order.

    Every chain of an entry is contiguous under a lexicographic sort of the IDs
    (they share the ``<entry>_`` prefix), so each pickle is loaded exactly once.
    """
    ids = sorted(ids)
    id_set = set(ids)
    n_entries = 0
    for entry_id, group in groupby(ids, key=lambda i: i.split("_", 1)[0]):
        group = list(group)
        n_entries += 1
        if progress_every and n_entries % progress_every == 0:
            log.info(f"  {n_entries} entries processed")
        try:
            entry = load_entry(entry_id, pkl_dir, bmrb_dir)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            log.error(f"entry {entry_id}: {type(exc).__name__}: {exc}")
            continue
        chains = {
            cid: (seq, shifts) for cid, seq, shifts in entry_chains(entry, id_set)
        }
        for cid in group:
            if cid in chains:
                yield (cid, *chains[cid])


def iter_shift_frames(ids, pkl_dir, bmrb_dir=None, progress_every=2000):
    """Yield ``(chain_id, DataFrame)`` for ``ids``, in sorted ``id`` order.

    Chains whose table fails a sequence-consistency guard are skipped, so the
    released table has no empty groups (it still joins into the main table).
    """
    for cid, seq, shifts in iter_chains(ids, pkl_dir, bmrb_dir, progress_every):
        df = get_deposited_shifts(shifts, seq)
        if df is None or df.empty:
            continue
        yield cid, df
