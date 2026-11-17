# Side-chain chemical-shift coverage

> **Example output, not maintained per release.** Written by `scripts/sidechain_coverage.py` on the 2026-11-17b rebuild (scoring commit bb177b2, BMRB snapshot bmrb_entries_2026-11-11), the build the v0.4.0 audit reviewed. A later build has other counts: regenerate it next to the build (`--output <release dir>/sidechain-coverage.md`), which is the copy `scripts/manuscript/make_numbers.py` reads first.

Values are AS DEPOSITED in BMRB: no re-referencing and no offset correction is applied to side-chain shifts.

## Bulk

| tier | chains | chains with side chains | total side-chain shifts | median / chain | IQR | 1H | 13C | 15N |
|---|---|---|---|---|---|---|---|---|
| unfiltered | 17403 | 12946 (74.4 %) | 3520256 | 105 | 0–359 | 2491621 | 964577 | 64058 |
| tolerant | 15272 | 11364 (74.4 %) | 3342626 | 131 | 0–387 | 2369951 | 911339 | 61336 |
| moderate | 10722 | 8330 (77.7 %) | 2802264 | 225 | 8–438 | 1969514 | 780501 | 52249 |
| strict | 3939 | 3158 (80.2 %) | 1218794 | 320 | 47–487 | 839957 | 355578 | 23259 |

Fully identical re-statements of the same value dropped corpus-wide: 1 (unfiltered). Genuinely conflicting values for the same (chain, seq_id, atom_id) are kept.

## Per-residue side-chain completeness

`all residues` divides by every canonical residue of the sequence; `assigned` divides only by residues carrying at least one deposited shift. `non-exchangeable` drops the labile side-chain protons from both numerator and denominator; `C/N only` keeps just the side-chain carbon/nitrogen skeleton.

| tier | all residues | all, non-exchangeable | all, C/N only | assigned residues | assigned, non-exchangeable | assigned, C/N only |
|---|---|---|---|---|---|---|
| unfiltered | 0.303 | 0.337 | 0.237 | 0.352 | 0.392 | 0.277 |
| tolerant | 0.315 | 0.351 | 0.246 | 0.362 | 0.403 | 0.284 |
| moderate | 0.368 | 0.410 | 0.293 | 0.391 | 0.436 | 0.313 |
| strict | 0.421 | 0.470 | 0.351 | 0.439 | 0.489 | 0.368 |

### By residue type (all residues, full expected set)

| residue | expected atoms | unfiltered | tolerant | moderate | strict |
|---|---|---|---|---|---|
| ARG | 15 | 0.170 | 0.176 | 0.204 | 0.228 |
| ASN | 4 | 0.261 | 0.272 | 0.313 | 0.361 |
| ASP | 1 | 0.012 | 0.006 | 0.007 | 0.005 |
| CYS | 1 | 0.009 | 0.010 | 0.011 | 0.014 |
| GLN | 7 | 0.304 | 0.318 | 0.372 | 0.440 |
| GLU | 4 | 0.305 | 0.318 | 0.372 | 0.425 |
| HIS | 9 | 0.086 | 0.089 | 0.104 | 0.124 |
| ILE | 11 | 0.457 | 0.476 | 0.554 | 0.628 |
| LEU | 10 | 0.441 | 0.460 | 0.534 | 0.598 |
| LYS | 13 | 0.253 | 0.264 | 0.311 | 0.357 |
| MET | 7 | 0.303 | 0.317 | 0.367 | 0.429 |
| PHE | 11 | 0.226 | 0.238 | 0.278 | 0.324 |
| PRO | 6 | 0.354 | 0.368 | 0.427 | 0.495 |
| SER | 1 | 0.009 | 0.010 | 0.011 | 0.011 |
| THR | 5 | 0.365 | 0.381 | 0.442 | 0.495 |
| TRP | 15 | 0.269 | 0.282 | 0.333 | 0.380 |
| TYR | 11 | 0.206 | 0.215 | 0.257 | 0.302 |
| VAL | 8 | 0.472 | 0.493 | 0.570 | 0.641 |

## Methyl-TROSY (ILVMA) 13C coverage

ILE CD1, LEU CD1/CD2, VAL CG1/CG2, MET CE, ALA CB. ALA CB is a backbone-whitelisted atom, so it is counted from the backbone mask.

| tier | overall | ALA | ILE | LEU | MET | VAL |
|---|---|---|---|---|---|---|
| unfiltered | 46.2 % | 69.6 % | 45.2 % | 41.3 % | 25.5 % | 43.3 % |
| tolerant | 47.9 % | 72.0 % | 46.7 % | 43.1 % | 26.5 % | 44.7 % |
| moderate | 56.1 % | 84.8 % | 54.5 % | 50.5 % | 30.7 % | 52.3 % |
| strict | 64.1 % | 91.7 % | 63.4 % | 58.5 % | 36.6 % | 61.3 % |

## Aromatic ring 13C/15N coverage

| tier | overall | HIS | PHE | TRP | TYR |
|---|---|---|---|---|---|
| unfiltered | 0.134 | 0.065 | 0.152 | 0.185 | 0.139 |
| tolerant | 0.141 | 0.067 | 0.161 | 0.194 | 0.145 |
| moderate | 0.170 | 0.080 | 0.193 | 0.237 | 0.179 |
| strict | 0.214 | 0.102 | 0.242 | 0.291 | 0.228 |
