# Side-chain chemical-shift coverage

Values are AS DEPOSITED in BMRB: no re-referencing and no offset correction is applied to side-chain shifts.

## Bulk

| tier | chains | chains with side chains | total side-chain shifts | median / chain | IQR | 1H | 13C | 15N |
|---|---|---|---|---|---|---|---|---|
| unfiltered | 17403 | 12946 (74.4 %) | 3520256 | 105 | 0–359 | 2491621 | 964577 | 64058 |
| tolerant | 15279 | 11367 (74.4 %) | 3343658 | 131 | 0–387 | 2370734 | 911578 | 61346 |
| moderate | 11305 | 8772 (77.6 %) | 2864339 | 208 | 8–428 | 2017822 | 793534 | 52983 |
| strict | 4245 | 3381 (79.6 %) | 1277382 | 307 | 35–479 | 881315 | 371882 | 24185 |

Fully identical re-statements of the same value dropped corpus-wide: 1 (unfiltered). Genuinely conflicting values for the same (chain, seq_id, atom_id) are kept.

## Per-residue side-chain completeness

`all residues` divides by every canonical residue of the sequence; `assigned` divides only by residues carrying at least one deposited shift. `non-exchangeable` drops the labile side-chain protons from both numerator and denominator; `C/N only` keeps just the side-chain carbon/nitrogen skeleton.

| tier | all residues | all, non-exchangeable | all, C/N only | assigned residues | assigned, non-exchangeable | assigned, C/N only |
|---|---|---|---|---|---|---|
| unfiltered | 0.303 | 0.337 | 0.237 | 0.352 | 0.392 | 0.277 |
| tolerant | 0.315 | 0.351 | 0.246 | 0.362 | 0.403 | 0.284 |
| moderate | 0.364 | 0.406 | 0.288 | 0.387 | 0.431 | 0.308 |
| strict | 0.412 | 0.460 | 0.343 | 0.431 | 0.480 | 0.360 |

### By residue type (all residues, full expected set)

| residue | expected atoms | unfiltered | tolerant | moderate | strict |
|---|---|---|---|---|---|
| ARG | 15 | 0.170 | 0.176 | 0.203 | 0.223 |
| ASN | 4 | 0.261 | 0.272 | 0.310 | 0.351 |
| ASP | 1 | 0.012 | 0.006 | 0.007 | 0.005 |
| CYS | 1 | 0.009 | 0.010 | 0.011 | 0.014 |
| GLN | 7 | 0.304 | 0.318 | 0.367 | 0.431 |
| GLU | 4 | 0.305 | 0.318 | 0.369 | 0.419 |
| HIS | 9 | 0.086 | 0.089 | 0.102 | 0.119 |
| ILE | 11 | 0.457 | 0.476 | 0.548 | 0.617 |
| LEU | 10 | 0.441 | 0.460 | 0.528 | 0.584 |
| LYS | 13 | 0.253 | 0.264 | 0.308 | 0.350 |
| MET | 7 | 0.303 | 0.317 | 0.363 | 0.422 |
| PHE | 11 | 0.226 | 0.238 | 0.273 | 0.315 |
| PRO | 6 | 0.354 | 0.368 | 0.423 | 0.486 |
| SER | 1 | 0.009 | 0.010 | 0.011 | 0.011 |
| THR | 5 | 0.365 | 0.381 | 0.437 | 0.486 |
| TRP | 15 | 0.269 | 0.282 | 0.325 | 0.368 |
| TYR | 11 | 0.206 | 0.215 | 0.252 | 0.292 |
| VAL | 8 | 0.472 | 0.493 | 0.565 | 0.631 |

## Methyl-TROSY (ILVMA) 13C coverage

ILE CD1, LEU CD1/CD2, VAL CG1/CG2, MET CE, ALA CB. ALA CB is a backbone-whitelisted atom, so it is counted from the backbone mask.

| tier | overall | ALA | ILE | LEU | MET | VAL |
|---|---|---|---|---|---|---|
| unfiltered | 46.2 % | 69.6 % | 45.2 % | 41.3 % | 25.5 % | 43.3 % |
| tolerant | 47.9 % | 72.0 % | 46.7 % | 43.0 % | 26.5 % | 44.7 % |
| moderate | 55.4 % | 84.4 % | 53.7 % | 49.6 % | 30.3 % | 51.6 % |
| strict | 63.0 % | 91.4 % | 62.2 % | 57.1 % | 36.0 % | 60.2 % |

## Aromatic ring 13C/15N coverage

| tier | overall | HIS | PHE | TRP | TYR |
|---|---|---|---|---|---|
| unfiltered | 0.134 | 0.065 | 0.152 | 0.185 | 0.139 |
| tolerant | 0.141 | 0.067 | 0.161 | 0.194 | 0.145 |
| moderate | 0.166 | 0.078 | 0.188 | 0.230 | 0.175 |
| strict | 0.206 | 0.097 | 0.234 | 0.281 | 0.219 |
