"""Path contract for the dataset-build chain.

Replaces the ``ROOT = Path(__file__).resolve().parents[3]`` hack that every
former docs/260520/scripts module hard-coded. Two roots:

* ``root``      -- the repository root, holding the committed/gitignored *inputs*
                   (``data/release``, ``data/2024-05-09``, ``data/chezod``,
                   ``tmp/bmrb_entries``) and a couple of doc files.
* ``work_dir``  -- where the build's intermediate + output artifacts live.
                   Defaults to ``<root>/docs/260520/data`` so a re-run reproduces
                   the existing artifacts byte-for-byte (verification); override
                   with ``--work-dir`` to build elsewhere.
"""

from pathlib import Path
from types import SimpleNamespace


def repo_root() -> Path:
    """Repository root (this file is trizod/dataset/paths.py -> parents[2])."""
    return Path(__file__).resolve().parents[2]


def resolve_paths(work_dir=None, root=None) -> SimpleNamespace:
    """Resolve every named path the dataset-build chain reads or writes.

    Args:
        work_dir: output/intermediate root (default ``<root>/docs/260520/data``).
        root:     repository root (default: auto-detected).
    """
    root = Path(root).resolve() if root is not None else repo_root()
    work_dir = (
        Path(work_dir).resolve()
        if work_dir is not None
        else root / "docs" / "260520" / "data"
    )
    return SimpleNamespace(
        root=root,
        work_dir=work_dir,
        # repo-root inputs
        release=root / "data" / "release",
        pkl_dir=root / "tmp" / "bmrb_entries",
        chezod117=root / "data" / "2024-05-09" / "CheZOD117_test_set.fasta",
        chezod1325_txt=root
        / "data"
        / "chezod"
        / "protein_nmr_1325"
        / "allseqs1325.txt",
        bundle_readme=root / "docs" / "260520" / "bundle-README.md",
        # work-dir artifacts
        final_dataset=work_dir / "final_dataset",
        mmseqs=work_dir / "mmseqs",
        testset=work_dir / "testset",
        release_bundle=work_dir / "release_bundle",
        # deploy-fasta output (260623)
        deploy_out=root
        / "docs"
        / "260623"
        / "data"
        / "deploy"
        / "disorder_trizod.fasta",
    )
