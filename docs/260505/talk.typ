// =============================================================================
// TriZOD — Final Pipeline & Re-Referenced Dataset
// 6 May 2026 project meeting talk (~13-14 min)
// touying 0.7.3 + metropolis theme
// =============================================================================

// ── 1. IMPORTS ────────────────────────────────────────────────────────────────
#import "@preview/touying:0.7.3": *
#import themes.metropolis: *
#import "@preview/numbly:0.1.0": numbly


// ── 2. THEME SETUP ────────────────────────────────────────────────────────────
#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  footer: self => self.info.title,
  font: ("Aptos",),

  config-info(
    title: [TriZOD — Final Pipeline & Re-Referenced Dataset],
    subtitle: [Step 8 wildcards · LACS in scoring · .str emission · alpha-synuclein],
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
#set text(font: "Aptos", size: 18pt)
#set strong(delta: 100)
#show strong: it => text(weight: "bold", it.body)
#set par(justify: true)
#set heading(numbering: numbly("{1}.", default: "1.1"))


// =============================================================================
// ── SLIDES BEGIN ──────────────────────────────────────────────────────────────
// =============================================================================

// Slide 1 — Title
#title-slide()


// Slide 2 — TriZOD in one slide
== TriZOD in one slide

- *Goal:* per-residue disorder scores from BMRB NMR shifts (CheZOD-style)
- *Inputs:* 17,388 BMRB NMR-STAR entries
- *Output:* per-tier filtered JSON/CSV + (new) re-referenced `.str` files
- *Z-score:* deviation from POTENCI random-coil prediction
- *G-score:* $[0,1]$ geometric-mean form combining backbone atom Z-scores


// Slide 3 — What's new since 22 April
== What's new since 22 April

- *Step 8:* Leu/Val ambiguous methyl wildcards (`CDx` / `CGx`)
- *Step 9:* LACS pre-correction baked into the scoring pipeline
- New flag: `--rereference-mode {none, lacs, potenci-only, both}` (default `both`)
- New flag: `--emit-str <dir>` emits per-entry re-referenced NMR-STAR files
- *Zenodo deposit metadata* in repo (DOI on first tagged release)


// Slide 4 — Filter improvements + per-tier deltas
== Filter improvements (Steps 4–7) and per-tier dataset deltas

#align(center)[#image("figures/per_tier_deltas.png", height: 70%)]

#text(size: 12pt)[Removed denaturant false-positives, fixed solid-state regex, added paramagnetic exclusion, relaxed `min-backbone-shift-types` 5→4 (strict), broadened the Celsius heuristic.]


// Slide 5 — Step 8: methyl wildcards
== Step 8: methyl wildcards

- LEU `CD1`/`CD2` -> `CDx`, VAL `CG1`/`CG2` -> `CGx` for non-stereospecific ambiguity codes
- Stereospecific assignments (ambiguity code `1`) preserved as-is
- *Backbone scoring unchanged* — wildcards surface only in the emitted `.str` files
- Restores otherwise-discarded sidechain methyl shifts for downstream consumers


// Slide 6 — Re-referencing in the pipeline
== Re-referencing in the pipeline

#align(center)[#image("figures/architecture.png", width: 92%)]

#v(0.5em)
#text(size: 13pt)[raw shifts → *LACS pre-correction* (Wishart RC tables, robust line fits) → *POTENCI residual* (AIC-gated rolling 9-window) → Z / G-scores]


// Slide 7 — LACS vs POTENCI residual capture
== LACS vs POTENCI residual capture

#slide(composer: (1.2fr, 1fr))[
  #image("figures/lacs_vs_potenci_overlap.png", width: 100%)
][
  #text(size: 14pt)[
    Per-entry CA / CB / C offsets in the strict tier.

    *LACS* catches the large systematic referencing bias.

    *POTENCI/AIC residual* captures the remainder — most entries land near the origin once LACS has done its job.
  ]
]


// Slide 8 — Entries materially affected by LACS
== Entries materially affected by LACS, by tier

#align(center)[#image("figures/flip_count_by_tier.png", height: 68%)]

#text(size: 12pt)[Red = entries with $|"LACS offset"| > 0.5$ ppm on at least one of C/CA/CB. Re-referencing meaningfully changes the input shifts for 21% (tolerant), 25% (moderate), 32% (strict).]


// Slide 9 — Reid #2: alpha-synuclein + top-3 flippers
== Reid \#2: alpha-synuclein and top-3 G-score flippers

#align(center)[#image("figures/gscore_flips.png", height: 70%)]

#text(size: 12pt)[BMRB 17665 raw → looks helical (matches original helical-tetramer paper); re-referenced → disordered, matching BMRB 6968 ground truth. αSyn LACS offset CA/CB = +2.82 ppm. Top-3 flippers 51068/52619/51262 show the same pattern.]


// Slide 10 — Reid #1: chemical shift perturbations
== Reid \#1: chemical shift perturbations on bound/unbound pairs

#slide(composer: (1fr, 1fr))[
  #image("figures/csp_histogram.png", width: 100%)

  #text(size: 11pt)[CSP across 581 pairs · 61,063 values · threshold 0.224 ppm]
][
  #image("figures/csp_interface_example.png", width: 100%)

  #text(size: 11pt)[FKBP12 apo (bmr16925) vs bound (bmr16931) — interface at res. 55, 58]
]


// Slide 11 — Final pipeline architecture
== Final pipeline architecture

#align(center)[#image("figures/architecture.png", width: 85%)]

#text(size: 13pt)[
- `--rereference-mode {none, lacs, potenci-only, both}`
- `--emit-str` writes one `.str` per scored entity (two saveframes)
- JSON now carries both `lacs_off_<atom>` and `off_<atom>`
- All four filter tiers re-run on the finalized pipeline
]


// Slide 12 — Released artefacts
== Released artefacts

- `data/release/<tier>/scores.json` — per-residue Z/G + LACS + POTENCI offsets
- `data/release/<tier>/str/bmr*_*_rereferenced.str` — re-referenced NMR-STAR
- `.zenodo.json` + `CITATION.cff` — DOI on first tagged release

*Per-tier counts after rerun:*

#table(
  columns: (auto, auto, auto),
  align: (left, right, right),
  table.header([*Tier*], [*Entries*], [*Coverage*]),
  [strict],     [3,033],  [17.00%],
  [moderate],   [10,107], [56.64%],
  [tolerant],   [15,433], [86.49%],
  [unfiltered], [16,851], [94.45%],
)


// Slide 13 — Next steps
== Next steps

- Push deposit to Zenodo (DOI placeholder until first tag)
- Iterate on Reid's CSP — extend to multi-atom $Delta omega_"RMS"$ (Schumann/Williamson 2008)
- Step 8 downstream: any auto-assignment partner who'd benefit from `CDx`/`CGx`?
- Reid follow-up with John Markley on why BMRB stopped LACS reports (frozen July 2020)
