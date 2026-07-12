"""Path contract for the dataset-build chain.

Thin wrapper over :mod:`trizod.paths` (the repo-wide source of truth). Keeps the
``resolve_paths(work_dir=..., root=...)`` contract the CLI modules rely on, so a
build can be redirected with ``--work-dir`` / ``--root`` while defaulting to the
canonical ``data/interim/build`` work area and the ``data/{raw,external,interim,
processed}`` tiers.
"""

from pathlib import Path
from types import SimpleNamespace

from trizod import paths


def repo_root() -> Path:
    """Repository root (delegates to :func:`trizod.paths.repo_root`)."""
    return paths.repo_root()


def resolve_paths(work_dir=None, root=None) -> SimpleNamespace:
    """Resolve every named path the dataset-build chain reads or writes.

    Args:
        work_dir: output/intermediate root (default ``data/interim/build``).
        root:     repository root (default: auto-detected).
    """
    layout = paths.layout(root)
    work_dir = (
        Path(work_dir).resolve() if work_dir is not None else layout.interim_build
    )
    return SimpleNamespace(
        root=layout.root,
        work_dir=work_dir,
        # repo-root inputs
        release=layout.interim_scored,
        pkl_dir=layout.pkl_dir,
        chezod117=layout.ext_chezod117 / "CheZOD117_test_set.fasta",
        chezod1325_txt=layout.ext_chezod_1325 / "allseqs1325.txt",
        bundle_readme=layout.root / "docs" / "260520" / "bundle-README.md",
        # work-dir artifacts
        final_dataset=work_dir / "final_dataset",
        mmseqs=work_dir / "mmseqs",
        testset=work_dir / "testset",
        release_bundle=work_dir / "release_bundle",
        # deploy-fasta output
        deploy_out=layout.processed_deploy / "disorder_trizod.fasta",
    )
