"""Shared test stand-ins.

* The smallest BMRB entry ``trizod.trizod.fill_row_data`` reads: one entity, one
  assembly, one condition list and one shift table, with the samples given.
* The synthetic ``trizod dataset build`` harness of the withdrawn-entry and
  placeholder gates: scored records, a pickled entry stub and a run of the real
  ``build.main`` on a temporary root.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import trizod.bmrb.bmrb as bmrb
from trizod import paths
from trizod.offsets import off_sigma_col, total_off_ppm_col
from trizod.trizod import fill_row_data

TIERS = ("unfiltered", "tolerant", "moderate", "strict")
ATOMS = ("C", "CA", "CB", "H", "HA", "HB", "N")

# --------------------------------------------------------------------------- #
# fill_row_data
# --------------------------------------------------------------------------- #


class Conditions:
    """A condition list: pH 7, 298 K, 0.1 M, no pressure recorded."""

    def get_ionic_strength(self, **kwargs):
        return 0.1

    def get_pH(self, **kwargs):
        return 7.0

    def get_temperature(self, **kwargs):
        return 298.0

    def get_pressure_bar(self):
        return np.nan


class Sample:
    """Stand-in for ``bmrb.Sample``.

    A component is ``(ID, Assembly_ID, Entity_ID, Mol_common_name, conc, units)``.
    """

    def __init__(self, components=(), solvent_system=None, details=None, name=None):
        self.name = name
        self.details = details
        self.solvent_system = solvent_system
        self.framecode = "sample_1"
        self.type = "solution"
        self.components = list(components)


class Named:
    """Stand-in for an entity or an assembly."""

    def __init__(self, name=None):
        self.name = name
        self.details = None
        self.seq = None
        self.paramagnetic = None
        self.entities = [("1", "1", "$e", None)]


class ExperimentList:
    experiments = [("1", "2D 1H-15N HSQC", None, "1", None, "isotropic")]


class Entry:
    """Stand-in for ``bmrb.BmrbEntry``.

    ``samples`` maps sample IDs to ``Sample``s. By default the entry has one
    shift table, ``("1", "1", "1")``, recorded under condition list ``"1"`` and
    referencing the samples in ``referenced``; ``conditions`` and
    ``peptide_shifts`` replace those.
    """

    def __init__(
        self,
        samples,
        referenced=("1",),
        *,
        title=None,
        citation_title=None,
        conditions=None,
        peptide_shifts=None,
    ):
        self.id = "1"
        self.title = title
        self.details = None
        self.citation_title = citation_title
        self.citation_DOI = None
        self.exp_method = "NMR"
        self.exp_method_subtype = "solution"
        self.citation_keywords = []
        self.struct_keywords = []
        self.entities = {"1": Named(name="entity")}
        self.assemblies = {"1": Named(name="assembly")}
        self.samples = samples
        self.conditions = {"1": Conditions()} if conditions is None else conditions
        self.experiment_list = ExperimentList()
        self._peptide_shifts = (
            {("1", "1", "1"): (None, "1", "1", list(referenced))}
            if peptide_shifts is None
            else peptide_shifts
        )

    def get_peptide_shifts(self):
        return self._peptide_shifts


def fill(entry, perturbing_cosolvents=(), keywords=(), st_id="1", **kwargs):
    """``fill_row_data`` for shift table ``st_id`` of ``entry`` (entity 1)."""
    row = pd.Series(
        {"entryID": "1", "stID": st_id, "entity_assemID": "1", "entityID": "1"}
    )
    bmrb_entries = pd.DataFrame({"entry": [entry]}, index=["1"])
    return fill_row_data(
        row,
        perturbing_cosolvents=list(perturbing_cosolvents),
        keywords=list(keywords),
        bmrb_entries=bmrb_entries,
        **kwargs,
    )


def fill_real(entry_id, st_id=None, perturbing_cosolvents=(), keywords=(), **kwargs):
    """``fill_row_data`` for a deposited entry under ``paths.RAW_BMRB``: shift
    table ``st_id``'s first chain (the entry's first chain when ``st_id`` is
    None)."""
    entry = bmrb.BmrbEntry(entry_id, paths.RAW_BMRB / f"bmr{entry_id}")
    shift_tables = entry.get_peptide_shifts()
    if st_id is None:
        key = next(iter(shift_tables))
    else:
        key = next(k for k in shift_tables if k[0] == st_id)
    row = pd.Series(
        {
            "entryID": entry_id,
            "stID": key[0],
            "entity_assemID": key[1],
            "entityID": key[2],
        }
    )
    bmrb_entries = pd.DataFrame({"entry": [entry]}, index=[entry_id])
    return fill_row_data(
        row,
        perturbing_cosolvents=list(perturbing_cosolvents),
        keywords=list(keywords),
        bmrb_entries=bmrb_entries,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# synthetic `trizod dataset build`
# --------------------------------------------------------------------------- #


def scored_record(rid: str, seq: str, n_bb_pos: int) -> dict:
    """A ``scores.json`` record for chain ``rid`` with every field build reads."""
    entry_id, st_id, assem_id, entity_id = rid.split("_")
    rec = {
        "ID": rid,
        "entryID": entry_id,
        "stID": st_id,
        "entity_assemID": assem_id,
        "entityID": entity_id,
        "seq": seq,
        "bbshift_positions_post": n_bb_pos,
        "bbshift_types_post": 5,
        "total_bbshifts": n_bb_pos * 5,
        "entity_name": "stub",
        "ionic_strength": 0.1,
        "pH": 7.0,
        "temperature": 298.0,
    }
    for atom in ATOMS:
        rec[off_sigma_col(atom)] = 0.1
        rec[total_off_ppm_col(atom)] = 0.05
    return rec


def stub_entry(bound: bool = False):
    """Smallest object ``composition.detect_bound`` classifies; unbound unless
    ``bound`` adds a nucleic-acid entity, which makes it a bound complex."""
    entities = {
        "1": SimpleNamespace(
            id="1",
            name="stub",
            type="polymer",
            polymer_type="polypeptide(L)",
            nonpolymer_comp_id="",
            nonpolymer_comp_label="",
        )
    }
    if bound:
        entities["2"] = SimpleNamespace(
            id="2",
            name="dna",
            type="polymer",
            polymer_type="polydeoxyribonucleotide",
            nonpolymer_comp_id="",
            nonpolymer_comp_label="",
        )
    return SimpleNamespace(entities=entities, assemblies={}, chem_comps={})


def run_build(
    tmp_path: Path,
    by_tier: dict[str, list[dict]],
    *extra: str,
    bound_entries: frozenset[str] = frozenset(),
) -> Path:
    """Write a synthetic root, run ``build.main`` on it, return final_dataset/."""
    from trizod.dataset import build

    root = tmp_path / "root"
    pkl_dir = root / "tmp" / "bmrb_entries"
    pkl_dir.mkdir(parents=True)
    entry_ids = set()
    for tier in TIERS:
        tier_dir = root / "data" / "interim" / "scored" / tier
        tier_dir.mkdir(parents=True)
        with (tier_dir / "scores.json").open("w") as fh:
            for rec in by_tier.get(tier, []):
                fh.write(json.dumps(rec) + "\n")
                entry_ids.add(rec["entryID"])
    for eid in entry_ids:
        with (pkl_dir / f"{eid}.pkl").open("wb") as fh:
            pickle.dump(stub_entry(bound=eid in bound_entries), fh)
    work = tmp_path / "work"
    build.main(["--work-dir", str(work), "--root", str(root), *extra])
    return work / "final_dataset"
