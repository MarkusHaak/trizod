"""Shared style and data paths for the TriZOD paper figures (main + Supplementary).

Every generator in ``scripts/figures/paper/`` imports this module first, so all
figures share one look and one set of inputs:

* **Size.** Artwork is drawn at its final size and placed at 1:1: 160 mm for the
  full text width (``W_DOUBLE``), 78 mm for a half-width cell (``W_SINGLE``);
  both fit Nature's 180 mm / 88 mm limits. Use :func:`figure` with millimetres.
* **Type.** Sans-serif (Arial, falling back to Helvetica / DejaVu Sans). All text
  is 7 pt (``FS_BODY`` = ``FS_TICK`` = 7: axis titles, tick labels, legends,
  direct labels), panel letters are 8 pt bold lower case (``FS_PANEL``), and
  nothing is smaller than ``FS_SMALL`` = 6 pt (NCS sets no text below 5 pt at
  final size). :func:`save` enforces the floor on every text item, including
  generator-local sizes and sub/superscripts, and labels log axes with plain
  numbers (1, 10, 100, 1,000) instead of 10^n, whose exponent matplotlib sets at
  70 % of the tick size. Lay text out at ``FS_SMALL`` or larger, so the floor
  never moves it after layout. Fonts are embedded as TrueType
  (``pdf.fonttype = 42``) so the PDFs stay editable.
* **Colour.** One colour per entity, in every main and Supplementary figure
  (story blueprint 6.8; ``reports/story/STYLE_NOTES.md``), each with a second cue
  in :data:`ENTITY`: TriZOD release black filled, CheZOD1325 grey ``#8C8C8C``
  open, CheZOD117 vermillion triangle, TriZOD test set blue circle, training
  bluish green; removed / excluded items light grey with hatching (patches) or
  a dash (lines), never a hue. Replicate conditions use :data:`COND` (same black,
  different bluish green, different but state-clean bluish green open). The four
  nested tiers use one sequential reddish-purple ramp, light = permissive
  (unfiltered) to dark = strict, so "darker is stricter"; shift count k is a
  brown-to-gold ramp (``trizod_k``: none of the entity hues, no grey). Two groups of one entity (ordered and disordered residues) differ by
  marker and fill in ink (:data:`CONTRAST`), never by grey, which is CheZOD1325.
* **Output.** Vector PDF into the directory given by the required ``--out-dir``
  argument (the draft's ``Figures/paper``; :func:`set_out_dir`), data tables into
  its sub-folders; an optional PNG preview goes to a scratch directory for visual
  checks only.

Data are read through :mod:`trizod.paths`, by default from the frozen 2026-10 build. The
data root is ``$TRIZOD_DATA_ROOT`` when set, else the first of the code checkout and its
parents that holds ``data/interim/build`` (a git worktree shares the main
checkout's ``data/``), else the :mod:`trizod.paths` default. A later build is selected
with ``$TRIZOD_BUILD_DIR``, ``$TRIZOD_RELEASE_DIR``, ``$TRIZOD_BUNDLE`` and
``$TRIZOD_REBUILD_LOG``, and the caches are moved with ``$TRIZOD_CACHE_DIR``
(``README.md``, "Selecting the build").

Run generators from the code checkout with
``uv run --extra figures --with pyarrow python scripts/figures/paper/<script>.py --out-dir <draft>/Figures/paper``
(``README.md`` lists every generator). Never import pyarrow in a process that
runs the scoring pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import text as mtext  # noqa: E402
from matplotlib import ticker as mticker  # noqa: E402

from trizod import paths  # noqa: E402

# ----------------------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------------------
REPO = paths.ROOT  # the code checkout: scripts/, docs/, trizod/
# git-ignored caches of the generators (raw scan, mmseqs, PANAV loops); $TRIZOD_CACHE_DIR
# moves them, so the caches of two builds never mix
TMP = (
    Path(os.environ["TRIZOD_CACHE_DIR"]).expanduser()
    if os.environ.get("TRIZOD_CACHE_DIR")
    else paths.TMP
)


def _find_root(start: Path) -> Path:
    """The checkout that holds ``data/interim/build`` (worktrees share the main one)."""
    for p in [start, *start.parents]:
        if (p / "data" / "interim" / "build").is_dir():
            return p
    return start


def _env_path(name: str, default: Path) -> Path:
    """``$name`` as a path when set and non-empty, else ``default``."""
    v = os.environ.get(name)
    return Path(v).expanduser() if v else default


ROOT = Path(os.environ.get("TRIZOD_DATA_ROOT") or _find_root(REPO))
_L = paths.layout(ROOT)
DATA = _L.data
# The build the figures describe. The defaults are the frozen 2026-10 build; a later build
# is selected with environment variables (README.md, "Selecting the build"), all optional:
#   TRIZOD_BUILD_DIR    build directory (final_dataset/, testset/, mmseqs/)
#   TRIZOD_RELEASE_DIR  trizod_dataset.parquet + trizod_shifts.parquet
#   TRIZOD_BUNDLE       release bundle (MANIFEST.json, scores/, train/, test/)
#   TRIZOD_REBUILD_LOG  the build log with the four print_filter_losses() reports
# The last three default to fixed names inside the BUILD directory, so pointing only
# TRIZOD_BUILD_DIR at another build fails loudly instead of mixing two builds.
BUILD = _env_path("TRIZOD_BUILD_DIR", _L.interim_build)
PQ_DIR = _env_path("TRIZOD_RELEASE_DIR", BUILD / "parquet-2026-10-simplify")
PQ_MAIN = PQ_DIR / "trizod_dataset.parquet"
PQ_SHIFTS = PQ_DIR / "trizod_shifts.parquet"
BUNDLE = _env_path("TRIZOD_BUNDLE", BUILD / "release_bundle" / "trizod-dataset-2026-10")
REBUILD_LOG = _env_path("TRIZOD_REBUILD_LOG", BUILD / "rebuild_2026-10.log")
# The frozen build the draft was written on; its split predates rebuild fix (a).
FROZEN_PIPELINE = "trizod-5b8147a"


def _manifest() -> dict:
    try:
        return json.loads((BUNDLE / "MANIFEST.json").read_text())
    except (OSError, ValueError):
        return {}


_MANIFEST = _manifest()
RELEASE = _MANIFEST.get("version", "2026-10")  # release label, e.g. "2026-11"
PIPELINE = _MANIFEST.get("pipeline_version", FROZEN_PIPELINE)
IS_FROZEN = PIPELINE == FROZEN_PIPELINE  # True only for the pre-fix 2026-10 split
CHEZOD_VER = _L.interim_chezod_verification
CHEZOD = _L.ext_chezod
CHEZOD_1325 = _L.ext_chezod_1325
PANAV = _L.ext_panav
BMRB_LACS = _L.ext_bmrb_lacs
RAW_BMRB = _L.raw_bmrb

# Output directory (the draft's Figures/paper). Unset until a generator's main() parses
# its required --out-dir (parse_args), so importing a generator as a library -- as
# scripts/manuscript/make_numbers.py does -- never writes anywhere.
FIG_DIR: Path | None = None
# PNG previews only when TRIZOD_FIG_PREVIEW_DIR is set (Path("") is ".", which is truthy, so the
# old `Path(...) or None` wrote previews into the current directory; fixed 2026-10-31)
_preview = os.environ.get("TRIZOD_FIG_PREVIEW_DIR")
PREVIEW_DIR = Path(_preview) if _preview else None


def set_out_dir(path) -> Path:
    """Direct every later :func:`save` / :func:`out_dir` to ``path``."""
    global FIG_DIR
    FIG_DIR = Path(path)
    return FIG_DIR


def out_dir(sub: str | None = None) -> Path:
    """The output directory (``--out-dir``), or its sub-folder ``sub`` (not created)."""
    if FIG_DIR is None:
        raise RuntimeError(
            "no output directory: pass --out-dir <draft>/Figures/paper "
            "(or call style_paper.set_out_dir() first)"
        )
    return FIG_DIR if sub is None else FIG_DIR / sub


def cli(doc: str | None) -> argparse.ArgumentParser:
    """Argument parser of a generator, with the required ``--out-dir``."""
    ap = argparse.ArgumentParser(description=(doc or "").split("\n")[0])
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="figure directory of the manuscript draft (<draft>/Figures/paper); "
        "PDFs go here, data tables into its source_data/, fig3_data/ and "
        "suppfigs_*_data/ sub-folders",
    )
    return ap


def parse_args(ap: argparse.ArgumentParser, argv=None) -> argparse.Namespace:
    """Parse ``argv`` with ``ap`` (from :func:`cli`) and set the output directory."""
    args = ap.parse_args(argv)
    set_out_dir(args.out_dir)
    return args


# ----------------------------------------------------------------------------------
# constants shared by the analyses
# ----------------------------------------------------------------------------------
TIERS = ["unfiltered", "tolerant", "moderate", "strict"]  # permissive -> strict
RANK = {t: i for i, t in enumerate(TIERS)}
G_DIS = 0.65  # binary disorder cut used throughout the paper
BACKBONE_ATOMS = ["C", "CA", "CB", "H", "HA", "HB", "N"]

# ----------------------------------------------------------------------------------
# colours: one colour per entity, in every main and Supplementary figure
# ----------------------------------------------------------------------------------
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#8C8C8C",
}
INK = "#000000"  # text, axes and the TriZOD release
INK_SOFT = "#4D4D4D"  # secondary text, CI whiskers, separators
GREY_CHEZOD = OKABE_ITO["grey"]  # CheZOD1325 (3.4:1 on white), always with open markers
GREY_REMOVED = "#B3B3B3"  # removed / excluded items, always hatched or dashed
HATCH = "////"  # hatching of removed / excluded areas (hatch colour = edgecolor)
DASH_REMOVED = (0, (3.0, 1.5))  # the line counterpart of the hatching


class _Palette(dict):
    """``CAT``: a dict that warns when a legacy, ambiguous key is read."""

    ALIASES = {"chezod": "chezod1325"}

    def __getitem__(self, key):
        if key in self.ALIASES:
            warnings.warn(
                f"style_paper.CAT[{key!r}] is ambiguous: use CAT['chezod1325'] (grey, "
                "the CheZOD1325 set and the CheZOD method) or CAT['chezod117'] "
                "(vermillion, the CheZOD117 test set)",
                stacklevel=2,
            )
        return super().__getitem__(key)


CAT = _Palette(
    {
        # entities (story blueprint 6.8); their second cue (marker, fill, line) is in ENTITY
        "release": INK,  # TriZOD release, all chains: black, filled
        "chezod1325": GREY_CHEZOD,  # CheZOD1325 (and "run as CheZOD"): grey, open
        "chezod117": OKABE_ITO["vermillion"],  # CheZOD117 test set: vermillion triangle
        "test": OKABE_ITO["blue"],  # TriZOD test set: blue circle
        "train": OKABE_ITO["green"],  # training representatives: bluish green
        "removed": GREY_REMOVED,  # removed / excluded: light grey + HATCH / DASH_REMOVED
        "lacs": OKABE_ITO["skyblue"],  # LACS offset source (Supplementary Figs. 6, 15)
        # neutrals, not entities
        "neutral": INK_SOFT,  # secondary ink
        "light": "#BDBDBD",  # reference lines and guides (G = 0.65, medians)
        # legacy alias, kept so old call sites still run (warns): the CheZOD method /
        # CheZOD1325. New code names the set: chezod1325 or chezod117.
        "chezod": GREY_CHEZOD,
    }
)

# The second cue of each entity, so a figure stays readable in grey scale and for
# colour-blind readers. Keys are those of CAT; use marker_kw(), line_kw(), patch_kw().
ENTITY = {
    "release": {"color": INK, "marker": "o", "mfc": INK, "ls": "-", "label": "TriZOD"},
    "chezod1325": {
        "color": GREY_CHEZOD,
        "marker": "o",
        "mfc": "white",
        "ls": "--",
        "label": "CheZOD1325",
    },
    "chezod117": {
        "color": CAT["chezod117"],
        "marker": "^",
        "mfc": CAT["chezod117"],
        "ls": "-",
        "label": "CheZOD117",
    },
    "test": {
        "color": CAT["test"],
        "marker": "o",
        "mfc": CAT["test"],
        "ls": "-",
        "label": "TriZOD test set",
    },
    "train": {
        "color": CAT["train"],
        "marker": "o",
        "mfc": CAT["train"],
        "ls": "-",
        "label": "Training",
    },
    "removed": {
        "color": GREY_REMOVED,
        "marker": "x",
        "mfc": "none",
        "ls": DASH_REMOVED,
        "hatch": HATCH,
        "label": "Removed",
    },
}

# Replicate pairs by their conditions (Fig. 4a): same = black filled, different pH or
# temperature = bluish green filled, different but state-clean = bluish green open.
COND = {
    "same": {"color": INK, "marker": "o", "mfc": INK, "label": "Same conditions"},
    "different": {
        "color": OKABE_ITO["green"],
        "marker": "o",
        "mfc": OKABE_ITO["green"],
        "label": "Different pH or temperature",
    },
    "different_state_clean": {
        "color": OKABE_ITO["green"],
        "marker": "o",
        "mfc": "white",
        "label": "Different, without state differences",
    },
}

# Two groups of one entity (ordered vs disordered residues, chains that deviate from
# their replicates vs chains that agree): both in ink, told apart by marker and fill.
CONTRAST = {
    "focus": {"color": "#1A1A1A", "marker": "o", "mfc": "#1A1A1A"},
    "control": {"color": "#1A1A1A", "marker": "s", "mfc": "white"},
}


def _style(key_or_style) -> dict:
    """An ENTITY / COND / CONTRAST entry, given by its ENTITY key or as the dict itself."""
    return ENTITY[key_or_style] if isinstance(key_or_style, str) else key_or_style


def marker_kw(key_or_style, ms: float | None = None, mew: float = 0.8) -> dict:
    """``ax.plot`` keyword arguments for an entity's marker (no line).

    ``ax.plot(x, y, **sp.marker_kw("chezod117"))`` draws vermillion triangles;
    ``sp.marker_kw(sp.COND["different_state_clean"])`` an open bluish-green circle.
    """
    st = _style(key_or_style)
    kw = {
        "ls": "",
        "marker": st["marker"],
        "color": st["color"],
        "mfc": st["mfc"],
        "mec": st["color"],
        "mew": mew,
    }
    if ms is not None:
        kw["ms"] = ms
    return kw


def line_kw(key_or_style, lw: float | None = None) -> dict:
    """``ax.plot`` / ``ax.step`` keyword arguments for an entity's line."""
    st = _style(key_or_style)
    kw = {"color": st["color"], "ls": st.get("ls", "-")}
    if lw is not None:
        kw["lw"] = lw
    return kw


def patch_kw(key_or_style) -> dict:
    """Keyword arguments for a bar or area of an entity (removed: white, hatched)."""
    st = _style(key_or_style)
    if st.get("hatch"):
        return {
            "facecolor": "white",
            "edgecolor": st["color"],  # hatch colour; lw=0 hides the outline
            "hatch": st["hatch"],
            "lw": 0,
        }
    return {"facecolor": st["color"], "edgecolor": "none", "lw": 0}


def entity_handle(key_or_style, label: str | None = None, kind: str = "marker", **kw):
    """A legend handle (marker, line or patch) for an entity; prefer direct labels."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    st = _style(key_or_style)
    label = st.get("label", "") if label is None else label
    if kind == "patch":
        return Patch(label=label, **patch_kw(st), **kw)
    if kind == "line":
        return Line2D([], [], label=label, **line_kw(st), **kw)
    return Line2D([], [], label=label, **{"ms": 3.6, **marker_kw(st), **kw})


TIER_COLORS = {
    "unfiltered": "#D7A9CB",
    "tolerant": "#B86AA6",
    "moderate": "#86307A",
    "strict": "#3E0A45",
}
TIER_MARKERS = {"unfiltered": "o", "tolerant": "s", "moderate": "D", "strict": "^"}

# Per-residue shift count k (1-21) is drawn with one sequential map everywhere (the colour
# bar of Supplementary Fig. S2, the strips of Fig. 4b). Figures that bin chains by their
# median k (Fig. 3c) sample the map at each class centre, so a class colour matches the
# colour bar at its k. Since 12 Nov 2026 (figure review item 14) the map is a brown-to-gold
# ramp, "trizod_k": viridis put k <= 6 in the dark purple of the strict tier and k 7-15 in
# the bluish green of "different conditions" and training; cividis (the review's first
# option) has a neutral grey middle, which is CheZOD1325 (Fig. 3a, next to Fig. 3c), the
# "other sample" chains of Supplementary Fig. S8 and the density of Supplementary Fig. S2.
# Brown - ochre - gold avoids purple, green, blue, grey and the vermillion of CheZOD117,
# and its lightness rises monotonically (readable in grey scale and for colour-blind eyes).
K_CMAP_COLORS = ["#4a2a0a", "#9c6a1e", "#d9a82e", "#f3d75e"]  # k = 1 ... 21
K_CMAP = "trizod_k"


def _register_k_cmap() -> None:
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap

    if K_CMAP not in mpl.colormaps:
        mpl.colormaps.register(LinearSegmentedColormap.from_list(K_CMAP, K_CMAP_COLORS))


_register_k_cmap()
K_RANGE = (1, 21)
K_CLASSES = {"≤6": (0, 6), "7–15": (7, 15), "16–21": (16, 21)}  # (lo, hi], inclusive hi


def k_color(k: float) -> str:
    """Hex colour of shift count ``k`` on the shared k scale (:data:`K_CMAP`)."""
    import matplotlib as mpl

    lo, hi = K_RANGE
    return mpl.colors.to_hex(mpl.colormaps[K_CMAP]((k - lo) / (hi - lo)))


K_CLASS_COLORS = {
    name: k_color((max(lo, K_RANGE[0]) + hi) / 2)
    for name, (lo, hi) in K_CLASSES.items()
}

# ----------------------------------------------------------------------------------
# sizes and rcParams
# ----------------------------------------------------------------------------------
MM = 1 / 25.4
# The draft (rostlab layout, A4, 2.5 cm margins) has a 160 mm text width, so artwork is
# drawn at that width and placed at 1:1 (half width: 78 mm, placed at 0.4875\textwidth).
# Both fit Nature's 180 mm / 88 mm limits.
W_SINGLE = 78.0
W_DOUBLE = 160.0

# Text sizes. Nature sets figure text at 5-7 pt at final size; the paper uses the top of
# that range for everything (blueprint 6.8). The draft's text width is 160 mm and the
# artwork is drawn at that width (half width: 78 mm), so it is placed at 1:1 and a size
# here is the size on the page.
TEXTWIDTH_MM = 160.0
PLACE_SCALE = TEXTWIDTH_MM / W_DOUBLE  # 1.0: placed at 1:1
MIN_PT_PLACED = 5.0  # NCS minimum at final size
FS_SMALL = 6.0  # the floor: nothing smaller (save() enforces it); cell values, notes
FS_TICK = 7.0  # tick labels and legends: the same 7 pt as all other text
FS_BODY = 7.0  # axis titles, direct labels, body text: Nature's 7 pt ceiling
FS_PANEL = 8.0  # bold lower-case panel letters (Nature: 8 pt)
SCRIPT_SHRINK = 0.7  # mathtext sub/superscript size relative to its base (fixed in mpl)
COMPACT_FROM = 1e5  # log axes reaching this value label 1,000 as 1k, 10^6 as 1M
assert FS_SMALL * PLACE_SCALE >= MIN_PT_PLACED

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FS_BODY,
    "axes.titlesize": FS_BODY,
    "axes.labelsize": FS_BODY,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_TICK,
    "legend.title_fontsize": FS_TICK,
    "legend.frameon": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.2,
    "ytick.minor.size": 1.2,
    "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5,
    "axes.labelpad": 2.0,
    "axes.titlepad": 3.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.0,
    "patch.linewidth": 0.5,
    "hatch.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600,
    "figure.dpi": 150,
    "mathtext.default": "regular",
}


def apply_style() -> None:
    plt.rcParams.update(RC)


apply_style()


def figure(width_mm: float = W_DOUBLE, height_mm: float = 100.0, **kw):
    """A figure of the given size in millimetres (constrained layout off by default)."""
    return plt.figure(figsize=(width_mm * MM, height_mm * MM), **kw)


def panel_label(ax, letter: str, x: float = -0.18, y: float = 1.06, **kw) -> None:
    """Bold lower-case panel letter at the top-left of ``ax`` (axes fraction)."""
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=FS_PANEL,
        fontweight="bold",
        va="bottom",
        ha="left",
        **kw,
    )


def fig_label(fig, letter: str, x_mm: float, y_mm_from_top: float) -> None:
    """Panel letter placed in figure millimetres (for panels without a single axes)."""
    w, h = fig.get_size_inches() / MM
    fig.text(
        x_mm / w,
        1 - y_mm_from_top / h,
        letter,
        fontsize=FS_PANEL,
        fontweight="bold",
        va="top",
        ha="left",
    )


def direct_label(
    ax,
    x: float,
    y: float,
    text: str,
    dx_mm: float = 1.0,
    dy_mm: float = 0.0,
    ha: str = "left",
    va: str = "center",
    color: str = INK,
    fontsize: float = FS_BODY,
    **kw,
):
    """Write ``text`` beside the mark or line end at data point ``(x, y)``.

    The offset is in millimetres, so it does not change with the axis scale (log axes
    included). The text stays in ink; the coloured mark beside it carries the identity
    (blueprint 6.8). Use it instead of a legend wherever there is room.
    """
    from matplotlib.transforms import offset_copy

    tr = offset_copy(
        ax.transData, fig=ax.figure, x=dx_mm * MM, y=dy_mm * MM, units="inches"
    )
    kw.setdefault("clip_on", False)
    return ax.text(
        x, y, text, transform=tr, ha=ha, va=va, color=color, fontsize=fontsize, **kw
    )


# ----------------------------------------------------------------------------------
# minimum text size: plain log labels and a floor enforced at save time
# ----------------------------------------------------------------------------------
def plain_number(x: float, compact: bool = False) -> str:
    """``x`` without an exponent: 0.01, 1, 10, 1,000 (``compact``: 1k, 10k, 1M)."""
    if x == 0:
        return "0"
    sign = "\N{MINUS SIGN}" if x < 0 else ""
    a = abs(x)
    if compact and a >= 1e3:
        for div, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
            if a >= div:
                return f"{sign}{a / div:g}{suffix}"
    if a >= 1:
        return sign + (f"{a:,.0f}" if float(a).is_integer() else f"{a:,g}")
    return sign + np.format_float_positional(a, precision=12, trim="-")


class PlainLogFormatter(mticker.LogFormatter):
    """Log-axis tick labels as plain numbers (see :func:`plain_number`).

    Matplotlib's default labels are 10^n, whose exponent is set at 70 % of the tick
    size (3.8 pt for 5.5 pt ticks). Plain numbers keep every glyph at the tick size.
    Axes that reach ``COMPACT_FROM`` switch to 1k / 10k / 1M to keep labels short.
    Which ticks get a label is decided by :class:`matplotlib.ticker.LogFormatter`.
    """

    def __call__(self, x, pos=None):
        s = super().__call__(abs(x), pos)  # the base class drops the sign (symlog)
        return self.fix_minus("-" + s) if x < 0 and s else s

    def _num_to_string(self, x, vmin, vmax):
        return plain_number(x, compact=max(abs(vmin), abs(vmax)) >= COMPACT_FROM)


_POW10 = re.compile(
    r"^(?P<pre>[^$]*?)"
    r"(?:10\$\^\{(?P<a>[-+\N{MINUS SIGN}]?\d+)\}\$"
    r"|\$(?:\\mathdefault\{)?10\^\{(?P<b>[-+\N{MINUS SIGN}]?\d+)\}\}?\$)"
    r"(?P<post>[^$]*)$"
)


def plain_power_of_ten(s: str) -> str:
    """Rewrite a label such as ``10$^{-2}$`` or ``≤$10^{3}$`` as 0.01 / ≤1,000."""
    m = _POW10.match(s)
    if not m:
        return s
    e = int((m["a"] or m["b"]).replace("\N{MINUS SIGN}", "-"))
    return m["pre"] + plain_number(10.0**e) + m["post"]


def _script_depth(s: str) -> int:
    """Deepest sub/superscript level in the mathtext parts of ``s`` (0 = none)."""
    parts = re.split(r"(?<!\\)\$", s)
    if len(parts) < 3 or len(parts) % 2 == 0:  # mathtext needs paired $
        return 0
    best = 0
    for math_part in parts[1::2]:
        depth, opened, pending = 0, [], False
        for ch in math_part:
            if ch in "^_":
                pending = True
            elif ch == "{":
                opened.append(pending)
                if pending:
                    depth += 1
                    best = max(best, depth)
                pending = False
            elif ch == "}":
                if opened and opened.pop():
                    depth -= 1
            elif pending and not ch.isspace():
                best = max(best, depth + 1)  # single-character script, e.g. x^2
                pending = False
    return best


def min_glyph_pt(t) -> float:
    """Size of the smallest glyph of text artist ``t`` (its scripts included)."""
    return t.get_fontsize() * SCRIPT_SHRINK ** _script_depth(t.get_text())


class NoExponentFormatter(mticker.Formatter):
    """Wrap a tick formatter so power-of-ten labels come out as plain numbers.

    Catches labels a generator sets explicitly, e.g. ``set_xticks(t, ["10$^{-2}$"])``,
    which matplotlib turns into a formatter that would re-create them at every draw.
    """

    def __init__(self, inner):
        self.inner = inner

    def set_axis(self, axis):
        super().set_axis(axis)
        self.inner.set_axis(axis)

    def set_locs(self, locs):
        self.inner.set_locs(locs)

    def __call__(self, x, pos=None):
        return plain_power_of_ten(self.inner(x, pos))

    def format_ticks(self, values):
        return [plain_power_of_ten(s) for s in self.inner.format_ticks(values)]

    def get_offset(self):
        return self.inner.get_offset()

    def format_data(self, value):
        return self.inner.format_data(value)

    def format_data_short(self, value):
        return self.inner.format_data_short(value)


def _all_axes(fig) -> list:
    """Every axes of ``fig``, inset axes and colour bars of insets included."""
    from matplotlib.axes import Axes

    return fig.findobj(lambda a: isinstance(a, Axes))


def _axes_ticks(fig):
    """``(axis, which)`` for both axes of every axes of ``fig``."""
    for ax in _all_axes(fig):
        for axis in (ax.xaxis, ax.yaxis):
            for which in ("major", "minor"):
                yield axis, which


def _plain_log_labels(fig) -> int:
    """Replace default 10^n log/symlog formatters (colour bars included); count them."""
    n = 0
    for axis, which in _axes_ticks(fig):
        if axis.get_scale() not in ("log", "symlog"):
            continue
        fmt = getattr(axis, f"get_{which}_formatter")()
        if isinstance(fmt, mticker.LogFormatterMathtext):  # SciNotation too
            getattr(axis, f"set_{which}_formatter")(
                PlainLogFormatter(
                    base=getattr(fmt, "_base", 10.0),
                    labelOnlyBase=fmt.labelOnlyBase,
                    minor_thresholds=fmt.minor_thresholds,
                    linthresh=getattr(fmt, "_linthresh", None),
                )
            )
            n += 1
    return n


def _floor_pass(fig, floor: float) -> int:
    """Raise every drawn text item whose smallest glyph is below ``floor``; count them."""
    n = 0
    for axis, which in _axes_ticks(fig):
        labs = [
            t
            for t in axis.get_ticklabels(which=which)
            if t.get_visible() and t.get_text()
        ]
        # Explicit power-of-ten labels: fix them in the formatter, which re-creates
        # the label text at every draw.
        fmt = getattr(axis, f"get_{which}_formatter")()
        if not isinstance(fmt, NoExponentFormatter) and any(
            plain_power_of_ten(t.get_text()) != t.get_text() for t in labs
        ):
            getattr(axis, f"set_{which}_formatter")(NoExponentFormatter(fmt))
            n += 1
        # Tick labels are re-created when the layout changes the tick count, so raise
        # the axis-level label size too; new ticks then start at the floor.
        sizes = {round(t.get_fontsize(), 3) for t in labs}
        if len(sizes) == 1 and min(sizes) < floor - 1e-6:  # uniform labels only
            axis.set_tick_params(which=which, labelsize=floor)
            n += 1
    for t in fig.findobj(mtext.Text):
        s = t.get_text()
        if not t.get_visible() or not s or not s.strip():
            continue
        s_new = plain_power_of_ten(s)
        if s_new != s:
            t.set_text(s_new)
            n += 1
        need = floor / SCRIPT_SHRINK ** _script_depth(s_new)
        if t.get_fontsize() < need - 1e-6:
            t.set_fontsize(need)
            n += 1
    return n


def enforce_min_text(fig, floor: float = FS_SMALL) -> int:
    """Make every glyph of ``fig`` at least ``floor`` pt; return the number of edits.

    Log axes get plain-number labels (:class:`PlainLogFormatter`), power-of-ten
    labels are written out (``10$^{-2}$`` -> 0.01), and any text item smaller than
    ``floor`` -- generator-local sizes, cell values, legend notes, colour-bar ticks --
    is raised to it. A label with sub/superscripts is raised until its scripts reach
    ``floor``. The figure is drawn between passes so tick labels and the layout engine
    see the final sizes.
    """
    n = _plain_log_labels(fig)
    for _ in range(4):
        fig.canvas.draw()  # create tick labels, run the layout engine
        k = _floor_pass(fig, floor)
        n += k
        if k == 0:
            break
    small = [
        (round(min_glyph_pt(t), 2), t.get_text()[:30])
        for t in fig.findobj(mtext.Text)
        if t.get_visible() and t.get_text().strip() and min_glyph_pt(t) < floor - 1e-6
    ]
    if small:
        print(f"style_paper: still below {floor} pt: {small[:5]}", file=sys.stderr)
    return n


def _drawn_texts(fig):
    """The text items ``fig`` draws: tick labels inside the view, labels, legends, notes."""
    for ax in _all_axes(fig):
        for axis in (ax.xaxis, ax.yaxis):
            try:
                ticks = axis._update_ticks()  # only the ticks inside the view
            except AttributeError:  # pragma: no cover - older/newer matplotlib
                ticks = axis.get_major_ticks()
            for tick in ticks:
                yield tick.label1
                yield tick.label2
            yield axis.label
            yield axis.offsetText
        yield from (ax.title, ax._left_title, ax._right_title)
        yield from ax.texts
        if ax.get_legend() is not None:
            yield from ax.get_legend().get_texts()
            yield ax.get_legend().get_title()
    yield from fig.texts
    for leg in fig.legends:
        yield from leg.get_texts()
        yield leg.get_title()


def text_outside(fig, tol_mm: float = 0.1) -> list:
    """Visible text items that reach beyond the canvas, so the PDF would clip them.

    Returns ``(text, mm beyond the edge)`` pairs. Raising all text to 7 pt can push a
    key or a tick label over the edge of a tightly laid-out panel; :func:`save` reports
    them.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    w, h = fig.bbox.width, fig.bbox.height
    px_per_mm = fig.dpi * MM
    out = []
    for t in _drawn_texts(fig):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(r)
        over = max(-bb.x0, -bb.y0, bb.x1 - w, bb.y1 - h) / px_per_mm
        if over > tol_mm:
            out.append((t.get_text()[:30].replace("\n", " "), round(float(over), 1)))
    return out


def save(fig, name: str, preview: bool = True, min_text: bool = True) -> Path:
    """Write ``<out-dir>/<name>.pdf`` (vector); PNG preview to PREVIEW_DIR if set.

    With ``min_text`` (default) the figure first passes :func:`enforce_min_text`,
    so no glyph is smaller than ``FS_SMALL`` and log axes carry plain numbers.
    Text that reaches beyond the canvas is reported on stderr (:func:`text_outside`).
    """
    fig_dir = out_dir()  # fail before the (slow) text-floor pass
    if min_text:
        n = enforce_min_text(fig)
        if n:
            print(
                f"style_paper: {name}: {n} text edit(s) to reach the {FS_SMALL} pt floor",
                file=sys.stderr,
            )
    outside = text_outside(fig)
    if outside:
        print(
            f"style_paper: {name}: text beyond the canvas (mm): {outside[:6]}",
            file=sys.stderr,
        )
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"{name}.pdf"
    fig.savefig(
        out, metadata={"Creator": "scripts/figures/paper", "CreationDate": None}
    )
    if preview and PREVIEW_DIR is not None:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREVIEW_DIR / f"{name}.png", dpi=200)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------------
# small data helpers
# ----------------------------------------------------------------------------------
def arr(v) -> np.ndarray:
    """List with None -> float array with NaN."""
    return np.array([np.nan if x is None else x for x in v], dtype=float)


def load_main(columns=None):
    """The one-row-per-chain release Parquet as pandas (object strings, not arrow)."""
    import pandas as pd
    import pyarrow.parquet as pq

    df = pq.read_table(PQ_MAIN, columns=columns).to_pandas()
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) and df[col].dtype != object:
            df[col] = df[col].astype(object)
    return df


def load_tier_scores(tier: str, fields=("gscores", "zscores", "k")) -> dict:
    """Per-tier labels from the bundle: ``{ID: {field: array}}`` (JSONL)."""
    out = {}
    with open(BUNDLE / "scores" / tier / "scores.json") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            out[r["ID"]] = {
                f: arr(r[f]) if isinstance(r[f], list) else r[f] for f in fields
            }
    return out


def load_tier_ids() -> dict:
    """``{tier: set(IDs scored at that tier)}`` from the bundle's per-tier scores.json."""
    ids = {}
    for t in TIERS:
        s = set()
        with open(BUNDLE / "scores" / t / "scores.json") as fh:
            for line in fh:
                if line.strip():
                    s.add(json.loads(line)["ID"])
        ids[t] = s
    return ids


def strictest_tier(tier_ids: dict) -> dict:
    """``{ID: strictest tier at which it is scored}``."""
    out = {}
    for t in TIERS:
        for i in tier_ids[t]:
            out[i] = t
    return out


def read_fasta(path) -> dict:
    out, key = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            key = line[1:].split()[0]
            out[key] = ""
        elif key is not None:
            out[key] += line
    return out


def tier_legend_handles(tiers=TIERS, marker=True):
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [],
            [],
            color=TIER_COLORS[t],
            marker=TIER_MARKERS[t] if marker else None,
            lw=0.8,
            ms=3,
            label=t,
        )
        for t in tiers
    ]
