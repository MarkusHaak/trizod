// =============================================================================
// TriZOD — Final Pipeline & Re-Referenced Dataset
// 6 May 2026 project meeting talk · 14 slides · target ~14 min
// touying 0.7.3 + metropolis theme
//
// Iterated 2026-05-06: workflow centerpiece, no step-number framing,
// numbering disabled, body 22pt, max-CSP-per-pair slide added.
// =============================================================================

// ── 1. IMPORTS ────────────────────────────────────────────────────────────────
#import "@preview/touying:0.7.3": *
#import themes.metropolis: *


// ── 2. THEME SETUP ────────────────────────────────────────────────────────────
#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  footer: self => self.info.title,
  font: ("Aptos",),

  config-info(
    title: [TriZOD — Final Pipeline & Re-Referenced Dataset],
    subtitle: [Finalized pipeline · α-synuclein case study · chemical-shift perturbations],
    author: [Tobias Senoner],
    date: [6 May 2026],
    institution: [TUM · TriZOD project meeting],
  ),

  config-colors(
    primary:          rgb("#ba82ff"),
    primary-light:    rgb("#ffffff"),
    secondary:        rgb("#361a54"),
    neutral-lightest: rgb("#fafafa"),
    neutral-dark:     rgb("#361a54"),
    neutral-darkest:  rgb("#361a54"),
  ),
)


// ── 3. GLOBAL TEXT / STYLE RULES ─────────────────────────────────────────────
#set text(font: "Aptos", size: 22pt)
#set strong(delta: 100)
#show strong: it => text(weight: "bold", it.body)
#set par(justify: true)
#set heading(numbering: none)


// =============================================================================
// ── SLIDES BEGIN ──────────────────────────────────────────────────────────────
// =============================================================================

// Slide 1 — Title
#title-slide()


// Slide 2 — Workflow (centerpiece)
== TriZOD pipeline at a glance

#align(center)[#image("figures/workflow.png", height: 78%)]

#text(size: 15pt)[Goal: per-residue disorder scores (Z-score, G-score) from BMRB NMR shifts. Inputs at the left; output `.str` + JSON at the right; deferred work in the dashed branch.]


// Slide 3 — What's new since 22 April
== What's new since 22 April

- Methyl wildcards in the parser (Leu `CD1`/`CD2` → `CDx`, Val `CG1`/`CG2` → `CGx`)
- LACS pre-correction baked into the scoring pipeline
- New flag: `--rereference-mode {none, lacs, potenci-only, both}` (default `both`)
- New flag: `--emit-str <dir>` writes one re-referenced NMR-STAR per scored entity
- *Zenodo deposit metadata* in repo (DOI on first tagged release)


// Slide 4 — Filter improvements + per-tier deltas
== Filter improvements and per-tier dataset deltas

#align(center)[#image("figures/per_tier_deltas.png", height: 68%)]

#text(size: 15pt)[Removed denaturant false-positives, fixed the solid-state regex, added a paramagnetic-sample filter, relaxed `min-backbone-shift-types` 5→4 in strict, and broadened the Celsius heuristic.]


// Slide 5 — Methyl wildcards explained
== Methyl wildcards: what and why

#text(size: 19pt)[
- Leucine: two methyls `CD1` + `CD2` (γ-carbon).
- Valine: two methyls `CG1` + `CG2` (β-carbon).
- *Geminal pairs are NMR-equivalent* — most pulse sequences cannot tell them apart.
- BMRB depositors must label one of each pair anyway: arbitrary stereospecificity.
]

#v(0.4em)
#align(center)[#text(size: 18pt)[
  *Before:* `LEU CD1 23.4 ppm` · `LEU CD2 24.1 ppm` (false claim of stereospecificity)

  *After (ambiguity ≠ 1):* `LEU CDx 23.4 ppm` · `LEU CDx 24.1 ppm`
]]

#v(0.4em)
#text(size: 14pt)[Backbone scoring is unchanged (these are side-chain methyls). The wildcards surface in the emitted `.str` files so downstream auto-assignment tools no longer propagate the false stereospecificity.]


// Slide 6 — Re-referencing pipeline diagram
== Re-referencing in the pipeline

#align(center)[#image("figures/architecture.png", width: 92%)]

#v(0.4em)
#text(size: 15pt)[raw shifts → *LACS pre-correction* (Wishart RC tables, robust line fits) → *POTENCI residual* (AIC-gated rolling 9-window) → Z / G-scores. Default mode is `both`.]


// Slide 7 — LACS vs POTENCI residual capture
== LACS vs POTENCI residual capture

#slide(composer: (1.2fr, 1fr))[
  #image("figures/lacs_vs_potenci_overlap.png", width: 100%)
][
  #text(size: 18pt)[
    Each dot is *one entry × one atom* (CA / CB / C, strict tier).
    Not accumulated.

    *LACS* catches the large systematic referencing bias.

    *POTENCI/AIC residual* mops up the remainder — most points land near the origin once LACS has done its job.
  ]
]


// Slide 8 — Entries materially affected by LACS
== Entries materially affected by LACS

#align(center)[#image("figures/flip_count_by_tier.png", height: 68%)]

#text(size: 15pt)[Red = entries with $|"LACS offset"| > 0.5$ ppm on at least one of C/CA/CB. Re-referencing meaningfully changes the input shifts for *21%* (tolerant), *25%* (moderate), *32%* (strict).]


// Slide 9 — Reid #2: alpha-synuclein + top-3 flippers
== α-synuclein and top-3 G-score flippers

#align(center)[#image("figures/gscore_flips.png", height: 70%)]

#text(size: 15pt)[Panel A · BMRB *17665* (αSyn, mis-referenced) raw vs re-referenced + BMRB *6968* (αSyn ground truth). Panels B-D · top-3 flippers BMRB *51068* / *52619* / *51262*. Gaps come from `k = 0` triplets (insufficient comparable shifts → G-score is NaN). αSyn LACS offset CA/CB = +2.82 ppm.]


// Slide 10 — Reid #1: residue-level CSPs
== Reid \#1: chemical-shift perturbations (residue-level)

#slide(composer: (1fr, 1fr))[
  #image("figures/csp_histogram.png", width: 100%)

  #text(size: 13pt)[All 61,063 per-residue CSPs across 581 pairs. Threshold *0.224 ppm* = trimmed-mean+SD on the bottom 90% (top 10% dropped so the threshold isn't inflated by interface residues).]
][
  #image("figures/csp_interface_example.png", width: 100%)

  #text(size: 13pt)[FKBP12 apo (bmr16925) vs bound (bmr16931). Two binding-pocket residues (55, 58) far above threshold.]
]


// Slide 11 — Max CSP per pair
== Max CSP per pair · how strong is the strongest binding shift?

#align(center)[#image("figures/max_csp_per_pair.png", height: 75%)]

#text(size: 15pt)[One value per pair (581 points): the *maximum* HN/N CSP within each pair. Pairs below the 0.224-ppm threshold are essentially silent; those above are real binding events.]


// Slide 12 — What's still TODO
== What's still TODO

- *Multi-molecule entries distort G-scores* — exclude entries with a non-polymer or nucleic-acid binding partner from the dataset.
- *Per-sequence representative selection* — for the same protein with multiple BMRB entries, pick the one with best experimental conditions; tiebreak by median G-score.
- *mmseqs2 sequence clustering* for the ML train/val/test split (per the original CheZOD/TriZOD report).
- All three are flagged as TODO in the workflow on slide 2.


// Slide 13 — Released artefacts
== Released artefacts

- `data/release/<tier>/scores.json` — per-residue Z/G + LACS + POTENCI offsets
- `data/release/<tier>/str/bmr*_*_rereferenced.str` — re-referenced NMR-STAR
- `.zenodo.json` + `CITATION.cff` — DOI on first tagged release

#v(0.5em)
*Per-tier counts after the rerun:*

#text(size: 18pt)[
  #table(
    columns: (auto, auto, auto),
    align: (left, right, right),
    table.header([*Tier*], [*Entries*], [*Coverage*]),
    [strict],     [3,033],  [17.00%],
    [moderate],   [10,107], [56.64%],
    [tolerant],   [15,433], [86.49%],
    [unfiltered], [16,851], [94.45%],
  )
]


// Slide 14 — Next steps
== Next steps

- Push deposit to Zenodo (DOI placeholder until first tag)
- Iterate on Reid's CSP — extend to multi-atom $Delta omega_"RMS"$ (Schumann/Williamson 2008)
- Reid follow-up with John Markley on why BMRB stopped LACS reports (frozen July 2020)
- Possible auto-assignment partner who'd benefit from `CDx`/`CGx` outputs
