"""Manuscript figure generators for TriZOD.

Each module produces one manuscript figure (or family) from released pipeline
outputs. Import the specific generator you need — the modules are kept
independently importable so the lightweight helpers (``style``, ``chezod``) do
not pull in matplotlib unless a plotting module is used:

    from trizod.figures.fig2_lacs import plot_lacs_effect
    from trizod.figures.fig2_lacs_case_study import build_case_study_figure
    from trizod.figures.chezod import load_chezod, load_trizod, summarize

The reproducible manuscript entrypoint is
``scripts/figures/regenerate_manuscript_figures.py``.
"""
