"""A small synthetic release: package bundle, framework results, BMRB snapshot.

Shared by the release-metadata, class-balance and finalize tests. Ten chains
cover every split: training representatives at each tier, a redundant member,
two TriZOD test chains, a retired one, a CheZOD117 chain and an excluded one,
and one chain per 2026-11 condition rule.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

from trizod import paths

SCRIPT = paths.ROOT / "scripts" / "build_parquet_dataset.py"
TIERS = ("unfiltered", "tolerant", "moderate", "strict")
FRAMEWORK_COMMIT = {"commit": "f" * 40, "dirty": False}
BUILD_COMMIT = {"commit": "c" * 40, "dirty": False}


def load_builder():
    """``scripts/build_parquet_dataset.py`` as a module (once per process)."""
    name = "build_parquet_dataset"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _seq(i: int, n: int = 30) -> str:
    rng = random.Random(i)
    return "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(n))


#: chain -> (scored at these tiers, extra scores.json fields)
CHAINS = {
    "1001_1_1_1": (TIERS, {}),
    "1002_1_1_1": (TIERS[:3], {}),
    "1003_1_1_1": (TIERS[:2], {"membrane_mimetic_sample": True}),
    "1004_1_1_1": (TIERS[:1], {"pressure_bar": 2000.0}),
    "1005_1_1_1": (TIERS, {}),
    "2001_1_1_1": (TIERS, {}),
    "2002_1_1_1": (TIERS, {}),
    "2003_1_1_1": (TIERS[:2], {"temperature_placeholder": True}),
    "3001_1_1_1": (TIERS, {}),
    "4001_1_1_1": (TIERS[:1], {"ionic_strength": 1.0}),
}
#: training representatives per tier (nested)
REPS = {
    "unfiltered": ["1001_1_1_1", "1002_1_1_1", "1003_1_1_1", "1004_1_1_1"],
    "tolerant": ["1001_1_1_1", "1002_1_1_1", "1003_1_1_1"],
    "moderate": ["1001_1_1_1", "1002_1_1_1"],
    "strict": ["1001_1_1_1"],
}
TEST = ["2001_1_1_1", "2002_1_1_1"]
RETIRED = {"2003_1_1_1": "not_strict"}
CHEZOD = ["3001_1_1_1"]


def gscores(i: int, n: int = 30) -> list:
    """Deterministic G with the two termini unscored and every zone present."""
    rng = random.Random(1000 + i)
    g = [round(rng.random(), 4) for _ in range(n)]
    g[0] = g[-1] = None
    return g


def record(cid: str, extra: dict) -> dict:
    i = int(cid.split("_", 1)[0])
    g = gscores(i)
    rec = {
        "ID": cid,
        "entryID": cid.split("_", 1)[0],
        "stID": "1",
        "entity_assemID": "1",
        "entityID": "1",
        "entity_name": f"protein {i}",
        "seq": _seq(i),
        "k": [0] + [5 if j % 3 else 12 for j in range(1, 29)] + [0],
        "gscores": g,
        "zscores": [None if x is None else 10 * x for x in g],
        "pH": 6.5,
        "temperature": 298.0,
        "ionic_strength": 0.1,
        "membrane_mimetic_sample": False,
        "temperature_placeholder": False,
    }
    rec.update(extra)
    return rec


def _fasta(path: Path, ids, headers=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = headers or {}
    path.write_text(
        "".join(
            f">{cid}{(' ' + headers[cid]) if cid in headers else ''}\n"
            f"{_seq(int(cid.split('_', 1)[0]))}\n"
            for cid in ids
        )
    )


def write_bundle(root: Path) -> Path:
    """The package bundle ``trizod dataset package`` would stage."""
    bundle = Path(root)
    counts = {}
    for tier in TIERS:
        held = [c for c, (tiers, _) in CHAINS.items() if tier in tiers]
        counts[tier] = len(held)
        d = bundle / "scores" / tier
        d.mkdir(parents=True)
        (d / "scores.json").write_text(
            "".join(json.dumps(record(c, CHAINS[c][1])) + "\n" for c in held)
        )
        t = bundle / "train" / tier
        _fasta(t / f"train_{tier}_best.fasta", REPS[tier])
        _fasta(t / f"train_{tier}.fasta", REPS[tier])
        members = [(r, r) for r in REPS[tier]] + [("1001_1_1_1", "1005_1_1_1")]
        (t / "clusters_best.tsv").write_text(
            "cluster_repr\tmember\tbest_repr\tmember_quality\n"
            + "".join(f"{r}\t{m}\t{r}\t1.0\n" for r, m in members)
        )
        (t / "clusters.tsv").write_text("".join(f"{r}\t{m}\n" for r, m in members))
    test = bundle / "test"
    _fasta(
        test / "TriZOD_test_set.fasta",
        TEST,
        {c: f"label_tier=strict pinned_id={c}" for c in TEST},
    )
    _fasta(
        test / "TriZOD_test_retired.fasta",
        RETIRED,
        {
            c: f"label_tier=tolerant pinned_id={c} reason={r}"
            for c, r in RETIRED.items()
        },
    )
    _fasta(test / "CheZOD117_test_set.fasta", CHEZOD)
    (test / "CheZOD117_original_sequences.fasta").write_text(">hmbdi\nMKVLAAGG\n")
    (test / "TriZOD_test_set_labels.tsv").write_text(
        "test_id\tpinned_id\tsubstituted\tlabel_tier\tlength\n"
        + "".join(f"{c}\t{c}\tFalse\tstrict\t30\n" for c in TEST)
    )
    manifest = {
        "version": "0.4.0",
        "pipeline_version": "trizod-cccccc",
        "leakage": {"rule": "either", "gate": "passed", "min_bits": 27.0},
        "tiers": {
            t: {"scored_records": counts[t], "training_reps": len(REPS[t])}
            for t in TIERS
        },
    }
    (bundle / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return bundle


def _triple(v, lo, hi):
    return [v, lo, hi]


def results_json(cuts=(0.15, 0.35, 0.7), release=None) -> dict:
    """A threshold-framework results.json with every key read_calibration uses."""
    at = {
        "0.15": {
            "LR": _triple(0.56, 0.32, 0.83),
            "posterior": _triple(0.11, 0.05, 0.2),
        },
        "0.35": {"LR": _triple(5.2, 3.4, 7.95), "posterior": _triple(0.54, 0.38, 0.69)},
        "0.70": {
            "LR": _triple(36.7, 23.2, 157.6),
            "posterior": _triple(0.89, 0.81, 0.98),
        },
    }

    def transport(name, pi):
        row = {
            "set": name,
            "chains": 3,
            "residues": 84,
            "pi_em": pi,
            "pi_em_lo": pi - 0.05,
            "pi_em_hi": pi + 0.05,
        }
        for c in ("0.15", "0.35", "0.70"):
            row[f"posterior_at_{c}"] = 0.5
            row[f"posterior_at_{c}_lo"] = 0.4
            row[f"posterior_at_{c}_hi"] = 0.6
        return row

    return {
        "protocol": "synthetic",
        "settings": {
            "paper_values": list(cuts),
            "levels": [0.1, 0.5, 0.9],
            "primary_estimator": "interp",
            "rule": "disordered iff G >= t",
            "pools": {"consensus": {"A": 0.25, "B": 0.25, "C": 0.5}},
            "tier": "strict",
            "nboot": 1000,
            "seed": 1,
        },
        "inputs": {"release": str(release) if release else None},
        "input_sha256": {"release": "e" * 64},
        "analysis_set": {"chains": 30, "fit_chains": 25, "test_chains": 5},
        "calibration": {
            "estimators": {
                "interp": {
                    "consensus": {
                        "pi": [0.186, 0.141, 0.244],
                        "at_paper_values": at,
                    }
                }
            },
            "paper_check": {
                "interp": {
                    "ordered_edge": {
                        "level": 0.1,
                        "paper_value": 0.15,
                        "estimate": 0.138,
                        "CI": [0.068, 0.192],
                        "inside_CI": True,
                    }
                }
            },
        },
        "transport": [
            transport("analysis_set", 0.186),
            transport("train", 0.27),
            transport("test_trizod", 0.15),
            transport("test_chezod117", 0.89),
            {"set": "fixed prevalence 0.10", "pi_em": 0.1},
        ],
        "k_check": {
            "replicates": [
                {
                    "k": "k1-9",
                    "kappa_binary": [0.80, 0.69, 0.88],
                    "disordered_flip": [0.138, 0.04, 0.29],
                    "n_residue_pairs": 7075,
                },
                {
                    "k": "k10-21",
                    "kappa_binary": [0.94, 0.92, 0.96],
                    "disordered_flip": [0.022, 0.008, 0.046],
                    "n_residue_pairs": 25452,
                },
            ]
        },
    }


#: The synthetic per-bin consensus log-LR: rising linearly from -3 to 4.
LOG_LR = [round(-3 + 7 * b / 99, 6) for b in range(100)]


def write_results(directory: Path, curve: bool = True, **kw) -> Path:
    """results.json, and (``curve``) the per-bin ``curves_bins.csv`` beside it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(json.dumps(results_json(**kw)))
    if curve:
        (directory / "curves_bins.csv").write_text(
            "G_lo,G_mid,logLR_interp\n"
            + "".join(
                f"{b / 100:g},{b / 100 + 0.005:g},{v}\n" for b, v in enumerate(LOG_LR)
            )
        )
    return directory


def write_snapshot(directory: Path, entry_ids=("1001", "2001", "3001")) -> Path:
    """A BMRB snapshot directory: BMRB_SNAPSHOT.json + bmr<ID>/bmr<ID>_3.str."""
    directory = Path(directory)
    for eid in entry_ids:
        d = directory / f"bmr{eid}"
        d.mkdir(parents=True)
        (d / f"bmr{eid}_3.str").write_text(f"data_{eid}\n")
    (directory / "BMRB_SNAPSHOT.json").write_text(
        json.dumps(
            {
                "retrieved": "2026-11-11T12:46:31Z",
                "retrieved_end": "2026-11-11T13:23:49Z",
                "n_entries": len(entry_ids),
                "entry_list_sha256": "a" * 64,
                "entry_ids": list(entry_ids),
            }
        )
    )
    return directory


def write_shifts(path: Path) -> Path:
    """A minimal shift table (pyarrow required)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    ids = [c for c in CHAINS for _ in range(2)]
    table = pa.table(
        {
            "id": ids,
            "seq_id": [1, 2] * len(CHAINS),
            "atom_id": ["CA", "N"] * len(CHAINS),
            "val_ppm": [55.0, 120.0] * len(CHAINS),
        }
    )
    pq.write_table(table, path)
    return path


def build_release_parquet(
    bundle: Path, out: Path, *, snapshot_dir=None, results_dir=None, overlaps=None
) -> Path:
    """The release Parquet of the synthetic bundle, with release metadata."""
    from trizod.dataset import release_info as ri

    builder = load_builder()
    builder.build(
        bundle,
        out,
        None,
        chezod_overlaps=overlaps,
        snapshot=ri.snapshot_block(snapshot_dir) if snapshot_dir else None,
        calibration=(
            ri.read_calibration(results_dir, FRAMEWORK_COMMIT) if results_dir else None
        ),
        commit=BUILD_COMMIT,
    )
    return out
