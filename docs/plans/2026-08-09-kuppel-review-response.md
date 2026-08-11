# TriZOD — Response Plan to the Kuppel Review

**Status:** proposal, pending Tobias sign-off
**Branch:** `develop` @ `28be333` (clean, read-only audit)
**Date of measurement:** 2026-08-09, against `data/interim/scored/*` as regenerated 2026-07-18

**Measurement provenance convention.** Every number below is tagged:
`[run]` = real `uv run trizod` pipeline execution against a hardlinked cache copy;
`[harness]` = faithful reimplementation of `fill_row_data`/`prefilter_dataframe`/`compute_scores`/`postfilter_dataframe`, validated to reproduce all four released tiers ID-for-ID (0 missing, 0 extra);
`[data]` = read directly from a released artefact (`scores.json`, `_composition_cache.csv`, `trizod_dataset.parquet`, raw `.str`);
`[meta]` = derived from the 16,963-entry `bmrb_meta.jsonl` extraction;
`[est]` = estimated, not measured end-to-end — flagged as such everywhere it appears.

Scripts live under the session scratchpad `…/bb1b9aef-…/scratchpad/` (`joint/`, `rel/`, `c1_step*`, `c2_*`, `c3_*`, `c4_step*`, `c5/`, `v5/`, `adv/`, `verify*/`). They are session-scoped and will not survive; the numbers, not the scripts, are the deliverable.

---

## 1. Executive summary

### 1.1 What is real

All six claims are at least partially confirmed. Two are outright code defects with reproducible root causes; two more are dead code; one is a design gap; one is a reframing of a real problem Sandro diagnosed correctly but attributed to the wrong tag.

**The most concrete finding is not in Sandro's list.** BMRB **5158** — apo-myoglobin molten globule — is `split=train, train_tier=moderate` in the published v0.3.0 deposit `[data]`. Its `_Entry.Title` and `_Citation.Title` both say "Molten Globule state"; `_Entity_assembly.Physical_state` is literally `molten globule`; TriZOD scores 52 % of its residues as ordered (mean Z 6.97). It misses strict only on T = 323 K, pH 4.1 and a missing method subtype. No state filter was ever involved. Reid's hypothetical is a training chain right now.

The twist: 5158's `_Citation_keyword` and `_Struct_keywords` are **both empty lists** `[meta]`. Issue #23 — the bug Sandro filed — does nothing for it. What catches 5158 is the `_Entity_assembly.Physical_state` tag, which is the half of his finding with no issue number. Siblings: **5119** (ATP synthase subunit c in chloroform/methanol, `molten globule`, 100 % ordered) and **16948** (dynamin GED in DMSO, `denatured`, 96 % ordered), both in train.

### 1.2 What is not real, or is smaller than claimed

| Sandro's framing | What we measured |
|---|---|
| "the exp-method whitelist is missing reliable spellings" | No spelling is missing. Every real solution-NMR spelling already passes. The entire loss is 4,103 entries whose subtype **tag is absent**. `[harness]` |
| "~800 entries recoverable to strict" | **+438** `[run]`. His ~800 almost exactly matches the **ionic-strength/unit gate** (+793 `[run]`), a different filter he did not name. |
| "add TFA to the denaturant list" | TFA is **already** in the strict list — and is a false positive: 56/57 percent-unit components are ≤ 0.2 % (median 0.1 %), i.e. HPLC counterion. `[meta]` |
| "~300 new files from denaturants" | 841 / 755 / 439 / **76** rows at unfiltered/tolerant/moderate/strict `[harness]`. 2.8× his estimate at tolerant, **4× lower** at strict. DPC alone is 299 entries ≈ his ~300. |
| "SDS/DPC/dodecyl are denaturants" | They are membrane mimetics. 61–72 % of the impact of his 7-token list is micelle chemistry, not denaturation. `[harness]` |
| "re-referencing cannot be applied to side chains" | **Sandro is right for the regime that matters.** Our own investigator's r = 0.925 counter-claim is an artifact of a mixed-unit sum and one broken chain — see §3.6. This claim is **withdrawn**. |

### 1.3 Headline numbers

**Three generations of tier counts are currently in circulation, and none of them match the data on disk.**

| artefact | strict | moderate | tolerant | unfiltered | TriZOD test |
|---|--:|--:|--:|--:|--:|
| `docs/dataset/datasheet.md`, `dataset-construction.md` | 3,033 | 10,107 | 15,433 | 16,851 | 344 |
| shipped v0.3.0 README, `bundle-README.md`, Fig 1a | 3,271 | 10,941 | 15,440 | 16,851 | 365 |
| **`data/interim/scored/` today** `[data]` | **3,514** | **11,306** | **15,446** | **16,851** | — |

`final_dataset_summary.json` records `initial_rows` strict = 3,271, proving the on-disk build (2026-07-14) predates the current scores (2026-07-18). **Rebuilding today with zero code changes drops 15 of the 365 published test chains (4.1 %)** `[rel/r1, rel/r2]` — the release is already broken before any of C1–C6 lands. C1–C6 are a smaller perturbation than the debt already on disk.

**Projected counts.** Three configurations, priced end-to-end unless marked:

| configuration | unfiltered | tolerant | moderate | strict | TriZOD test |
|---|--:|--:|--:|--:|--:|
| **today** `[data]` | 16,851 | 15,446 | 11,306 | 3,514 | 350/365 resolvable |
| **Set A** — bug fixes + C5 recovery + C2 denaturants, `interacti` **kept**, all state signals as columns only `[harness]` | 16,851 | 15,264 | 11,193 | **3,878** (+10.4 %) | 344/365 |
| **Set A + `interacti` dropped + physical_state deny list** — *recommended* `[est]` | 16,851 | ~15,210 | ~11,153 | **3,878 – 4,300, unpriced** | ~349/365 |
| **Set B** — Set A + deny list + `interacti`→complex-phrase swap `[harness]` | 16,850 | 15,212 | 11,158 | 4,011 | 345/365 |

Set A arithmetic is internally consistent to the row: 15,446 − 2 (C1a) − 27 (C5a) − 153 (TFE) = 15,264 exactly; 11,306 − 19 − 18 − 76 = 11,193 exactly; 3,514 − 50 − 7 + 1 + 434 = 3,892, minus the **14-row interaction** where C5-recovered rows are killed by the reactivated `interacti` keyword = 3,878 exactly `[harness]`.

**The recommended configuration's strict count is the single biggest unpriced number in this plan.** Dropping `interacti` re-admits 379 rows in the C1a context `[harness]` / 535 rows at `pass_pre` in the adversary's context, and interacts super-additively with C5 (28 strict rows are admitted only if both fixes land). Pricing it end-to-end is an explicit acceptance criterion for PR6.

**Cost of doing all of this**: ~14–18 developer-days plus regeneration. The regeneration itself is cheap — a full cold rebuild from raw `.str` is **under 40 minutes** `[joint/cold, cold2, parse_cost]`. The long pole is figures and manuscript numbers, not compute.

### 1.4 The three decisions that need a human

1. **`interacti` in the strict keyword blacklist — keep, drop, or swap for `in complex with`/`complexed with`?** Shipping the issue-#23 bugfix *without* touching this list is measurably the worst available option: it activates a keyword whose collateral is 54 %, deletes 6 of 365 published test chains (all collateral), and kills 14 of the 434 chains C5 recovers (barnase, cardiac troponin C, an A629P disease mutant ×4). **Recommended default: drop it.** Consequence: strict grows by 379–535 rows, of which ~30 % are genuine complexes that `detect_bound()` will drop at build time anyway — but they *will* be in `scores.json`, which Sandro consumes directly.

2. **The published 365-chain test set — preserve it or let it shrink?** 15 chains are already unresolvable for reasons unrelated to this review. A two-line change to `testset.py:280` (resolve the pin against the **tolerant** pool instead of strict, plus a `label_tier` column) preserves 365/365; all 365 pinned sequences are present in tolerant `[rel/r4]`. **Recommended default: re-pin against tolerant.** Never `--redraw` — the seeded draw is not stable under a pool change even with seed 42.

3. **State / membrane / oligomeric signals — filter by default, or ship as columns?** Filtering costs 10–16 test chains for membrane mimetics alone `[rel/r3]`; shipping columns costs zero and gives Sandro exactly what he asked for. **Recommended default: metadata-only in v0.4.0, with one exception** — the `_Entity_assembly.Physical_state` exact-match deny list, which is the only mechanism that catches 5158/5119/16948, removes 43 of 5,900 train representatives, and has zero measured IDP collateral. Everything else (membrane mimetics, `molten globule` as free text, `amyloid fibril`) goes to a post-release RFC.

---

## 2. Findings table

| # | Claim | Verdict | Measured impact | Ours vs Sandro | Class | Recommendation |
|---|---|---|---|---|---|---|
| **C1a** | `fields.extend(el)` explodes keyword lists into single characters; `_Citation_keyword`/`_Struct_keywords` have never matched anything (issue #23) | **Confirmed** | rows removed: tol **−2**, mod **−19**, strict **−50** `[harness]`; strict split: `interacti` 46, `unfold` 3, `misfold` 1 | no estimate given | **bug fix** | Ship — but **only together** with a decision on `interacti` (§7.1) |
| **C1b** | `_Entity_assembly.Physical_state` parsed at `bmrb.py:98`, used nowhere | **Confirmed** | exact-match deny list, marginal over C1a: tol **−54**, mod **−40**, strict **−7** `[harness]`; 47 distinct non-null values over 26,885 records | "~100 additional files" → **not confirmed, ~2× high** at recommended scope | **new metadata** + optional policy | Ship `physical_state` **column** at all four tiers. Ship the exact-match deny list at tolerant+ (no `unfolded` below moderate). Do **not** also add the field to the substring `fields` list |
| **C1c** | "molten" would catch ~30 more | **Confirmed at unfiltered/tolerant only** | 31 / 27 / 14 / **2** rows `[harness]`; **both strict hits are false positives** (19560 = folded Hsp90, matched via a citation title about p53) | "~30 more" — right for unfiltered/tolerant, wrong for strict | **policy — reject** | Do **not** add `molten globule` or `amyloid fibril` as free-text keywords. The physical_state route reaches 21 of the 31 with no topic-label FPs |
| **C2** | Chemical-denaturant list incomplete | **Partially confirmed** | his 7 tokens: **841 / 755 / 439 / 76** rows `[harness]`. TFE alone 175/153/76/7; DPC 309/283/185/46 | "~300 new files" — 2.8× low at tolerant, **4× high at strict** | **mixed** | Add `TFE` + `trifluoroethanol` (tol+, −217 tolerant); `DMSO` **ungated** (mod+, −62/−15); **remove** `TFA` and `Potassium Pyrophosphate`; membrane mimetics as a **column only** |
| **C2′** | (ours) `TFA` and `Potassium Pyrophosphate` are not denaturants | **Confirmed** | removing both gains **+1 strict row** (17958_1_1_1), not +47/+50 `[run]` | not raised by Sandro | **bug fix / hygiene** | Remove both. Pitch as a labelling correction with negligible data impact — **not** as data recovery |
| **C3** | `multi_protein_assembly` counts distinct Entity_IDs, so homo-oligomers never fire it | **Confirmed** | 157 entries / **178 rows** strict, 911 rows unfiltered, on current scores `[harness]`; 29/365 test_trizod, 7/115 CheZOD117, 416/5,900 train `[data]` | "~150 strict, ~770 unfiltered" — confirmed within 2.5 % | **new metadata** | Ship `n_entity_assembly_rows` (raw) + a conservatively-derived copy count using `Magnetic_equivalence_group_code`. **Do not** ship a column named `is_homo_oligomer` on the naive rule — it mislabels ≥39 monomeric conformer depositions incl. published test chain **15711 apoSOD1** |
| **C4a** | `has_metal` is dead code | **Confirmed** | **0 of 21,842** entity records typed `metal` `[meta]`; `has_metal=False` on all 16,963 rows of the shipped cache `[data]`; membership impact **0 in every tier** | histogram reproduced **exactly** | **bug fix + new metadata** | Parse `_Entity.Nonpolymer_comp_ID`; derive metal from `_Chem_comp.Formula` primarily, comp-ID allow-list only as tie-break; ship `ligand_comp_ids`/`ligand_names` separately |
| **C4b** | `has_nucleic` misses the DNA/RNA hybrid polymer type | **Confirmed** | **16 entity records in 14 entries** `[meta]`; 8 flip `has_nucleic`, 4 flip `is_bound`; **0 rows in any tier** | no magnitude given | **bug fix** | Ship. Replace the two-value exact match with a `NUCLEIC_POLYMER_TYPES` allow-list |
| **C5** | exp-method whitelist too narrow | **Reframed, then confirmed** | 4,466 rows / 4,103 entries lost solely to an **absent** subtype tag `[harness]`; recovery **+438** `[run]` / +434 with `require-solution` `[harness]` | "~800" → matches the **unit-assumptions** gate (+793 `[run]`), not this filter | **bug fix + policy** | Fix the `""`-sentinel regex bug (see C5′). Admit null-subtype entries to strict on positive `_Sample.Type` evidence, **restricted to subtype IS NULL** |
| **C5′** | (ours) `""` in the whitelist becomes a regex wildcard | **Confirmed** | `"\|solution\|structures"` matches every string → tolerant/moderate whitelists are **dead code**; 27 tolerant / 18 moderate junk-subtype rows pass today (X-RAY DIFFRACTION, THEORETICAL, STATE) `[harness]` | not raised | **bug fix** | Strip `""` before joining. This is what makes the manuscript's "excludes solid-state NMR" claim false |
| **C5″** | (ours) genuine solid-state entries in the released strict tier | **Confirmed** | **3 rows / 2 entries**: 25289 ×2 (Aβ fibrils, MAS, DARR/PAIN, `_Sample.Type=solid`) and **27211** (P. horikoshii TET2, all 5 samples `solid`) `[data]` | not raised | **bug fix (defect in v0.3.0)** | Add a refined solid veto (solid evidence **and** no solution-type experiment names); 100 % measured precision |
| **C6** | Side-chain shifts are discarded | **Confirmed, and larger** | **3,458,852** side-chain shifts discarded on unfiltered (2.45 M ¹H / 0.95 M ¹³C / 63 k ¹⁵N) `[meta]`; median 109/chain; 75 % of chains affected. **The release ships no shifts at all** — 42 Parquet columns, none of them a shift `[data]` | no estimate given | **new feature** | Ship a companion Parquet of raw side-chain shifts (`val`, `val_err`, `ambiguity_code`), ~11.9 MB. **Do not** ship `val_rereferenced` (§3.6) |
| **C6′** | (ours) `--include-shifts` emits nothing | **Confirmed** | `shifts` is assigned only inside the `no_shift_averaging` branch; verified by direct call `[run]` | not raised | **bug fix** | One-line fix, ships immediately and independently |
| **C6″** | (ours) `off_*` is documented as ppm and is not | **Confirmed** | `off_A` = mean of `diff/REFINED_WEIGHTS` → σ units; `lacs_off_A` is ppm. max \|off_HA\| = 25.22, impossible as ppm `[data]` | not raised | **doc bug in shipped v0.3.0** | Correct the v0.3.0 README column table, `datasheet.md:122`, and `str_writer.py`'s `Offset_ppm` tag. This error already caused one downstream analysis mistake |

---

## 3. Per-claim detail

Throughout this section, **verifier corrections are called out explicitly**. Where the original investigator and the adversarial verifier did not converge, the item is marked **UNRESOLVED** and no side is silently chosen.

---

### 3.1 C1 — keyword filter fields (`trizod/trizod.py:188–213`)

#### What is wrong

```python
# trizod/trizod.py:188-199  (current)
if entry.citation_keywords is not None:
    if isinstance(entry.citation_keywords, list):
        for el in entry.citation_keywords:
            fields.extend(el)          # <-- el is a str; extend() iterates its characters
    else:
        fields.extend(entry.citation_keywords)   # <-- same bug on the scalar path
```

`fields` is then substring-matched at L208–213. `citation_keywords = ['Protein misfolding', 'protein-protein interaction', 'membrane bound']` yields `fields = [None,'P','r','o','t','e','i','n', …]` (60 single characters) and all three blacklist flags `False`. Corrected, all three are `True`.

`_Entity_assembly.Physical_state` is parsed at `trizod/bmrb/bmrb.py:98` into `Assembly.entities[i][3]` and referenced **nowhere else** in `trizod/`, `scripts/` or `tests/`.

`docs/filtering.md:25` promises the blacklist searches text "mentioned anywhere in the BMRB file". That has never been true.

#### Evidence

- `citation_keywords` and `struct_keywords` are `list` in **all 16,963** entries (`Counter({'list': 16963})`), all 39,693 elements are `str`, longest 126 chars, only four single-character elements (`E`,`E`,`P`,`Z`) `[meta]`. The `else:` branch is dead code today — a latent bug, not an active one. No tier keyword is 1 character, so the bug is **fully inert**.
- 4,780 entries carry citation keywords, 6,101 carry struct keywords, 10,572 carry at least one `[meta]`.
- `physical_state` histogram over 26,885 entity-assembly records: `native` 19,598 · null 6,500 · `intrinsically disordered` 196 · `denatured` 156 · `na` 148 · `unfolded` 101 · `partially disordered` 52 · `molten globule` 38 · `reduced` 22 `[meta]`. **Sandro's histogram reproduced exactly.**
- Fixing the bug removes **2 / 19 / 50** rows from tolerant / moderate / strict `[harness]`. Cross-validated against a real strict pipeline run: 75 `pass_pre` rows removed, exactly 50 of them in the published release, 100 % ID agreement `[run]`.
- Of the 50 strict drops, 46 are `interacti`. **27/50 (54 %) have no independent complex signal** — `detect_bound()==False` *and* no complex/bound/with/fusion word in title, assembly name or entity name. Example: *"Assignments of free human Tsg101 UEV domain"*, matched by the citation keyword *"Protein-protein interactions"* `[meta]`.
- **6 of 365** published test chains are lost (15503, 15569, 16090, 50126, 50543, 50998); **all six are collateral**; **0** if `interacti` is dropped `[rel/r3]`.
- Dropping `interacti` re-admits **379 rows** in the C1a context `[harness]` / **535 `pass_pre` rows** in the adversary's context, of which 376/535 (70 %) are `detect_bound()==False` free monomers and 159/535 are genuine complexes `[harness]`.
- Exact-match physical_state deny list, marginal over C1a: **54 / 40 / 7** rows at tolerant / moderate / strict `[harness]`.
- `physical_state=='intrinsically disordered'` rows: 224 / 218 / 159 / 29 across the tiers. **Zero are lost** under either mechanism — the KEEP set protects them explicitly `[harness]`.
- The deny list is what catches **26816** (TDP-43 IDR in 8 M urea), **27324** (Cdc37 CTD in 8 M urea) and **6227** (urea-unfolded barstar). None lists urea as a sample component, so the denaturant filter misses all three; 26816's title contains neither "denatur" nor "unfold" `[meta]`.
- Train-pool impact of the deny list: **56 of 5,900** train representatives carry a non-native state signal; the deny list removes **43**; **1 of 365** test chains moves `[data]`.

#### Verifier corrections — stated plainly

| investigator said | verifier measured | status |
|---|---|---|
| tolerant deny list = **129 rows**, "CONFIRMS Sandro's ~100" | **54 rows** at the actually-recommended scope (129 is the full list *including* `unfolded` and the moderate/strict-only terms). The investigator's own summary line already said 54 — the write-up contradicted itself | **corrected.** Sandro's ~100 is **not** confirmed; it is ~2× high |
| "129, or 121 excluding reconstituted/micelle-bound" | **123** (the excluded values contribute exactly 6 rows) | corrected |
| BMRB **50211** is a mislabelled solid-state entry in strict | **False.** 50211's `_Entry.Title` is *"Solution-state NMR assignments of the patient FOR005 λ-III … variable domain"*; the fibril text is the **companion paper's citation title**. Real examples: **25289** (`_Sample.Type=solid`, subtype "NMR, 20 STRUCTURES") and **34178** (subtype "STATE", `_Sample.Type=fiber`) | **corrected.** The investigator made the exact topic-vs-state conflation it warned about elsewhere |
| 18 of 31 "molten" rows recoverable via physical_state | **21** | corrected |
| 48 distinct physical_state values | **47** non-null (48 counts the null) | corrected |
| recommended config: moderate 82 = kw 45 + ps 40 | 45+40 = 85 ≠ 82. Correct disjoint split: kw-only 42 + ps 40. Same for strict (298+7, not 299+7) | corrected; totals reproduce |
| "recommended config removes 302/305 rows" | It removes 302 **and adds 350**. Strict *grows* 1.4 %; churn is 652 rows = 18.6 % of the tier | **corrected — the headline omitted the additions** |
| "could not measure how many rows dropping `interacti` re-admits; upper bound 1,208" | Directly measurable: **535** | corrected |
| add `molten globule` + `amyloid fibril` to moderate/strict defaults | `molten globule` at strict is **0/2 precision** (both hits are 19560, folded Hsp90). `amyloid fibril` at strict is 2/8. At moderate, `amyloid fibril` deletes **51322** — Aβ(1-42) whose own `Physical_state` is `intrinsically disordered`, matched purely on its citation title | **rejected.** These fail the same test used to reject `interacti` |
| guard test `test_idp_states_never_blacklisted` protects IDPs | It does not. 51322 passes both its assertions and is still dropped, because the free-text keyword matched the **citation title**, not the state field | **corrected.** The guard must be row-level: no row whose own `physical_state` ∈ KEEP may be dropped by the keyword filter |
| keep `bound` in the strict list | Unaudited. 824 moderate rows match `bound`; in 19 of them *every* occurrence is `unbound`/`boundaries` — e.g. *"Unbound Med25ACID"*, *"Human Pdx1 Homeodomain in the Unbound State"*. Same failure class as `apo`→`apoptosis`, which the investigator rejected | **open — see §7.1** |

**UNRESOLVED:** whether the strict tier should do complex removal via keywords **at all**, given that `detect_bound()` already drops 85.8 % of what the complex keywords remove, at build time. The investigator raised it as an open question and did not answer it; verifier 2 answered "mostly no" from the same data. This is a scientific scope call, not a measurement.

#### Proposed patch

```python
# trizod/trizod.py, fill_row_data()
-    if entry.citation_keywords is not None:
-        if isinstance(entry.citation_keywords, list):
-            for el in entry.citation_keywords:
-                fields.extend(el)
-        else:
-            fields.extend(entry.citation_keywords)
-    if entry.struct_keywords is not None:
-        ...same...
+    # issue #23: these are LISTS OF STRINGS. fields.extend(el) on a str explodes
+    # it into single characters, so no multi-character keyword could ever match.
+    for kw_field in (entry.citation_keywords, entry.struct_keywords):
+        if kw_field is None:
+            continue
+        if isinstance(kw_field, list):
+            fields.extend(kw_field)
+        else:
+            fields.append(kw_field)

+    # _Entity_assembly.Physical_state, resolved to THIS row's entity assembly.
+    # Entry-wide OR over-filters by 10-28 rows per tier.
+    physical_state = None
+    for e_assem_ID, e_ID, _label, state in assembly.entities:
+        if e_assem_ID == row["entity_assemID"] and e_ID in ("", None, row["entityID"]):
+            physical_state = (state or None)
+            break
+    row["physical_state"] = physical_state or pd.NA
+    # EXACT match, case-insensitive. Deliberately NOT added to `fields` -- substring
+    # matching on free text is precisely what makes 'unfold' dangerous.
+    row["physical_state_blacklisted"] = bool(
+        physical_state and physical_state.strip().lower() in _deny
+    )
```

```python
# trizod/trizod.py, filter_defaults
-        "keywords-blacklist": [[], ["denatur"], ["denatur","unfold","misfold"],
-                               ["denatur","unfold","misfold","interacti","bound"]],
+        "keywords-blacklist": [[], ["denatur"], ["denatur","unfold","misfold"],
+                               ["denatur","unfold","misfold","bound"]],
+        # 'interacti' removed: 46 of its 50 hits are paper-topic labels
+        # ("protein-protein interaction") on free monomers -- 54% collateral,
+        # 6/365 published test chains, 14 of C5's 434 recovered rows.
+        "physical-state-blacklist": [[], _PS_TOLERANT, _PS_MODERATE, _PS_STRICT],
+
+PHYSICAL_STATE_KEEP = frozenset({
+    "native", "intrinsically disordered", "partially disordered",
+    "natively unstructured", "folded", "reduced", "recombinant",
+    "synthetic", "mutant",
+})
```

#### Tier policy

| tier | keywords | physical-state deny |
|---|---|---|
| unfiltered | `[]` (unchanged) | `[]` — but **emit the `physical_state` column** |
| tolerant | `["denatur"]` (unchanged) | `denatured`, `partially denatured`, `misfolded`, `non-native`, `aggregated`, `amyloid*`, `fibril*`, `molten globule` — **no `unfolded`** |
| moderate | `["denatur","unfold","misfold"]` (unchanged) | tolerant + `unfolded`, `partially unfolded`, `folding intermediate`, `intermediate` |
| strict | drop `interacti`; keep `bound` pending audit | moderate + `bound`, `micelle-bound`, `SLAS micelle-bound`, `Reconstituted*` |

Rationale for excluding `unfolded` at tolerant: of 67 tolerant rows with `physical_state=='unfolded'`, **0** have any tolerant-list chemical denaturant in the sample components, and ≥7 are canonical IDPs in native buffer (α-synuclein 16300/16301/25227/25228, γ-synuclein 7244, Tau 52309/52401, amelogenin 15662, NS5A D2 15225) `[meta]`. *(The investigator reported "only 1 of 67"; the verifier measured 0 — the one hit is 26680, which uses trifluoroethanol, a strict-only token. The argument gets stronger, not weaker.)*

#### Tests

- `test_citation_keywords_matched_whole` — `citation_keywords=['Protein misfolding']`, `keywords=['misfold']` → `True`. **Fails on develop today.**
- `test_struct_keywords_matched_whole`, `test_scalar_keyword_string_appended_not_exploded`, `test_single_character_keyword_still_matches`.
- `test_physical_state_row_resolved` — one assembly, entity_assembly `1='native'` / `2='denatured'`; row `1` must not fire, row `2` must.
- `test_physical_state_deny_is_exact_not_substring` — `['denatured']` must not fire on `'partially denatured'` or `'not denatured'`.
- **`test_keep_state_never_dropped_by_any_filter`** — row-level, not set-level: no row whose own `physical_state` ∈ `PHYSICAL_STATE_KEEP` may be dropped by the keyword filter. This is the guard that would catch the 51322 case; the originally proposed set-level test would not.
- `test_unseen_physical_state_warns` — any value above a count threshold in neither KEEP nor deny logs a warning.
- Regression pin: 15503/15569/16090/50126/50543/50998 remain in strict.

#### Risks

- Dropping `interacti` re-admits **159 genuine complexes** to `scores.json` (adversary measurement). `detect_bound()` removes them at build time, but any consumer reading `scores.json` directly — **including Sandro** — sees a *less* pure strict tier. This must be stated in the release note.
- `physical_state` is depositor free text with a growing vocabulary (`fibril`/`fibrils`/`Fibrillar`/`amyloid fibril`/`amyloid fibrils` are five separate values). Exact match will silently miss new spellings. Mitigated by the warn-on-unseen test.
- `citation_keywords`/`citation_title`/`struct_keywords` are **topic metadata about the paper, not descriptions of the sample**. Fixing C1a doubles down on searching them. `citation_title` alone is the exclusive source of 1,221 of the 1,810 corpus `interacti` matches. **The root design issue — one flat substring list applied to a bag of state-bearing and topic-bearing fields — is not solved by this patch.** A field-scoping boolean (exclude citation/struct metadata from the *state* blacklist) is one line in `fill_row_data` and would fix 51322, 19560, and all 6 test-set losses at once. Verifier 2 argues this is "arguably the actual fix". It is **not** in the recommended patch and should be an RFC item.

---

### 3.2 C2 — chemical denaturants (`trizod/trizod.py:214–227`)

#### What is wrong

The matcher is:

```python
for den_comp in chemical_denaturants:
    for sID in sampleIDs:
        for comp in entry.samples[sID].components:
            if comp[3] and not comp[2] and den_comp.lower() in comp[3].lower():
```

`comp[2]` = `_Sample_component.Entity_ID`, `comp[3]` = `Mol_common_name`. The list is incomplete (no `TFE` abbreviation, no `DMSO`), contains a non-denaturant that was never justified (`Potassium Pyrophosphate`, added 2023-09-14 in `ba5586b` inside an unreviewed 12-token grab-bag, 9 of which were later purged in `8eb7ad3`), and contains a false positive (`TFA`).

#### Evidence

- Sandro's 7 tokens remove **841 / 755 / 439 / 76** rows (771 / 691 / 420 / 74 entries) `[harness]`. Sanity residual is **0** in all four tiers — the replay reproduces the released datasets exactly. Both independent verifiers reproduced every per-token figure bit-for-bit.
- `TFE` is genuinely new: `'trifluoroethanol'.lower()` does not contain `'tfe'`. 149 of 151 TFE entries are missed by the existing token; union 216 `[meta]`.
- **TFA concentrations**: 57 percent-unit components, min 0.01, **median 0.1 %**, max 1.0; 56/57 ≤ 0.2 % `[meta]`. HPLC counterion. Its acidification pathway is already covered by the independent pH filter.
- **Removing TFA gains exactly 1 strict row** (17958_1_1_1) `[run]`. Only 6 of the 47 TFA entries survive even the moderate tier; of those, 5 fail strict on pH NaN / pH 4.0 / pH 5.2 / shift-fraction 0.653.
- **Removing Potassium Pyrophosphate gains 0 strict rows** `[run]`. 4881 and 4886 have a missing `exp_method_subtype` (rejected by the strict whitelist regardless); 16596 hits `interacti` and has \|off_HB\| = 2.545 > max-offset 2.0.
- DMSO is bimodal: 99 entries ≥ 5 % v/v, 16 < 5 %, 16 unparseable `[meta]`. Ungated at strict it removes 15 rows, 6 of which are 1–2.5 % ligand-stock co-solvent.
- Zero string-level false positives for Sandro's 7 tokens: `'SDS-PAGE'` occurs **0 times** in the 14,981-value `Mol_common_name` corpus `[meta]`.
- Membrane mimetics: SDS+DPC+dodecyl = 515 / 477 / 302 / **55** rows; the full 16-token vocabulary = **766 / 695 / 454 / 95** rows via the real matcher `[harness]`.

#### Verifier corrections — stated plainly

Both verifiers independently reproduced the entire measurement block bit-for-bit, then found the **derived policy arithmetic** broken:

| investigator said | verifiers measured | status |
|---|---|---|
| removing TFA "gains ~47 entries" / "+47ish rows" | **+1 row** | **corrected, ~47×.** Both verifiers. The investigator's own evidence block said "~6"; the plan said 47 |
| removing KPP: "+3 strict rows, 3514 → 3517" | **+0 rows** | **corrected.** Both verifiers |
| tolerant cost of the proposal: 153 rows | **217** (the proposal adds both `TFE` *and* `trifluoroethanol`; the second is worth 64 more) | corrected |
| moderate cost: ~116 | **158** | corrected |
| strict net: ~−20 | **−69** (−1.96 %) | **corrected, 3.5×** |
| membrane-FULL: 816/744/489/**100** rows | 816/744/489/100 is an **entry-level** upper bound; the real matcher gives **766/695/454/95** | corrected — methodology switch presented as measurement |
| Entity_ID guard hides 39 entries | **27** newly captured; no definition reaches 39. The actionable subset (7 entries referencing a non-polymer entity) reproduces exactly — **but verifier 2 found 9**, adding 26756/26757 (`CHAPS`, Entity_ID=4, non-polymer) | **corrected; 7-vs-9 UNRESOLVED** (depends on which token vocabulary is used) |
| `tests/reference/unfiltered.json` has 10 flagged rows | **12** (omits 21014, 36079, both TFA). Conclusion unaffected | corrected |
| the test-set pin is drawn from the **tolerant** tier | It is resolved against the **strict** pool (`testset.py:280` reads `final_dataset/strict/strict.fasta`). Tolerant losses are irrelevant to it | **corrected.** And the real impact is worse than stated: membrane exclusion drops **8** (narrow) / **14** (full) pinned records, none substitutable |
| hoisting the sampleIDs fallback "changes keyword results for ~0.94 % of rows" | 0.94 % is the rate at which the *code path* differs. Outcome differs for **0** of the 20 fallback rows under any tier's keyword list | corrected — the risk text invites a reviewer to block a harmless refactor |

**Verifier finding neither party had:** `Sample.components` is built by `zip()` over six `get_tag_vals(..., default=[])` calls (`bmrb.py:239–251`). If any one of the six `_Sample_component` tags is absent, the entire component list collapses to empty. **1,117 of 16,963 entries (6.6 %)** have a sample with an empty component list, and in every one of those entries *all* samples are empty `[meta]`. No token-list change can ever see those entries. If the goal is "catch denaturants", fixing this is worth more than adding TFE.

**Latent KeyError, in both current code and the proposed rewrite:** `bmrb.py` applies `.strip()` *after* the membership test — `[sID.strip() for sID in set(sampleIDs) if sID in self.samples]` — so a whitespace-padded sample ID yields a key that is no longer in `entry.samples`. The `try/except Found` does not catch `KeyError`. No occurrence in the current corpus. Use `entry.samples.get(sID)`.

#### Proposed patch

```python
-        "chemical-denaturants": [
-            [], ["guanidin","GdmCl","Gdn-Hcl","urea"],
-            ["guanidin","GdmCl","Gdn-Hcl","urea"],
-            ["guanidin","GdmCl","Gdn-Hcl","urea","TFA","trifluoroethanol",
-             "Potassium Pyrophosphate"],
-        ],
+        "chemical-denaturants": [
+            [],
+            ["guanidin","GdmCl","Gdn-Hcl","urea","TFE","trifluoroethanol"],
+            ["guanidin","GdmCl","Gdn-Hcl","urea","TFE","trifluoroethanol","DMSO"],
+            ["guanidin","GdmCl","Gdn-Hcl","urea","TFE","trifluoroethanol","DMSO"],
+        ],  # 'TFA' (HPLC counterion, median 0.1%) and 'Potassium Pyrophosphate'
+            # (buffer) intentionally REMOVED. Net data impact: +1 strict row.
+
+# Standalone vocabulary -- the tier default references THIS, not the reverse.
+MEMBRANE_MIMETIC_TOKENS = ("SDS","DPC","dodecyl","LPPG","LMPG","DHPC","DMPC",
+                           "POPC","bicelle","micelle","Triton","CHAPS","octyl",
+                           "maltoside","digitonin","nanodisc")
+        "membrane-mimetics": [[], [], [], []],   # metadata only in v0.4.0
```

Plus: always-emitted `membrane_mimetic` column carrying the **matched token(s) as a string**, not a bool — a bool collapses the SDS-micelle vs DDM-solubilisation distinction the plan itself argues matters.

**No DMSO concentration gate.** The gate is ~40 lines of unit normalisation over depositor free text (`'%'`, `'% v/v'`, `'v/v'` meaning both 0.2 and 20, `'uL'`, `'mg/ml'`, `'-'`) and buys 6 spared rows at strict and 12 at moderate. Its boundary is undefined at the only value that matters (bmr36172 is exactly 5.0 %) and 19355 has a blank concentration with unit `mM`. Ship ungated; record `dmso_pct` in metadata if anyone wants it.

#### Tier policy

unfiltered `[]` (preserves `tests/reference/unfiltered.json`) · tolerant +TFE/+trifluoroethanol · moderate/strict +DMSO ungated, −TFA, −KPP · membrane mimetics `[]` at every tier, column emitted everywhere.

#### Tests

Table-driven on stub samples: `TFE-d2` matches `TFE`; `trifluoroethanol` does **not** match `TFE`; `sodium dodecyl sulfate` matches `dodecyl` but not `SDS`; Entity_ID guard blocks `4-guanidinophenyl 4-guanidinobenzoate` (31202), `bis-pyridylurea inhibitor` (26609), `NTFecA` (6803). `test_potassium_pyrophosphate_no_longer_filters` and `test_tfa_removed_from_strict` must assert the **denaturant flag**, not tier membership — 4881/4886/16596/15195/16148 are all rejected by unrelated strict filters.

#### Risks

- **Framing risk, highest priority:** if TFA/KPP removal is pitched to Reid as "recovers ~47 entries", the first person to check finds 1. Pitch as correctness with negligible impact.
- The three strict-only tokens `TFA`, `trifluoroethanol` and `Potassium Pyrophosphate` currently remove **0 strict rows each**. The entire strict-only extension of the denaturant list costs the strict tier nothing today — that is the cleanest framing for Reid.
- Membrane mimetics as a default-on strict filter would be a one-sided removal of ordered examples: the removed rows are 93.7 % ordered and contain **zero** disordered chains (baseline disordered fraction 4.1 %) `[data]`. This is the strongest argument for column-only.

---

### 3.3 C3 — homo-oligomer detection (`trizod/dataset/composition.py:12–40`)

#### What is wrong

`multi_protein_assembly` is `len({e[1] for e in asm.entities}) > 1` — a homo-oligomer references **one** Entity n times, so it never fires. `n_entities = len(entry.entities)` is 1 for a homodimer. This is **deliberate and documented** at `composition.py:13-15` and `:32-34`. The defect is discoverability: no `is_homo_oligomer` / `n_copies` column exists in any TSV or in the published Parquet, so a consumer cannot filter even though they should be allowed to.

#### Evidence

- Homo-oligomeric rows surviving `is_bound` + `len≥20`: **178 strict / 157 entries**, 582 moderate, 790 tolerant, 911 unfiltered — on the **current** scores `[harness]`.
- Published v0.3.0 splits: train 416/5,900 (7.05 %), test_trizod **29/365**, test_chezod117 **7/115**, redundant 175/2,337, excluded 735/8,134 `[data]`.
- Cost of excluding: 6.45–6.73 % of unique sequences in every tier `[data]`.
- Homo-oligomers are significantly **more ordered**: strict mean G 0.132 (n=178) vs 0.172 (n=2,409), one-sided MWU p = 4.95e-4 `[data]`. Excluding them shifts the label distribution toward disorder and removes well-determined ordered examples.
- The published v0.3.0 Parquet has **42 columns and no composition column at all** — no `is_bound`, no `n_entities` `[data]`. Adding composition metadata is a first, not a widening.
- Only **2 of 16,963** entries have >1 `_Assembly` record, and **0** have an entity spanning assemblies `[meta]` — the "same entity in separate assemblies" confound is empirically non-existent.

#### Verifier corrections — stated plainly

| investigator said | verifiers measured | status |
|---|---|---|
| strict 152 entries / 172 rows | **157 / 178** on the current scores. The investigator read the **stale 2026-07-14 build**; recomputing on the Jul-18 scores moves every tier | **corrected.** Quote 157/178, or state the as-of date |
| flag-OFF path needs no rerun of the redundancy chain | **False.** Regenerating `_composition_cache.csv` requires `trizod dataset build`, which reads `data/interim/scored` — 4 days newer than the build on disk — and adds ~212 strict / ~307 moderate rows, changing the FASTAs | **corrected** |
| per-`(Entity_ID, Physical_state)` grouping "demotes exactly these 10 [alt-state depositions] and no genuine oligomer" | **Wrong in both directions.** (a) 11 divergent assemblies in **11** entries, not 10 (bmr19104 missing). (b) bmr19104 is a **live false positive**: 3 rows, states `native`/`unfolded`/`native` — two share a state, so it yields `n_copies=2`. Title: *"Induced folding in RNA recognition by A. thaliana DCL1"*. It is `split=train, train_tier=tolerant` in the release | **corrected** |
| — | Verifier 2, from the **raw `.str`** (tags `Entity_assembly_name`, `Conformational_isomer`, `Magnetic_equivalence_group_code`, all present in 1,276/1,276 files and **all discarded by `bmrb.py`**): **92 entity-groups in 91 entries** carry a conformer annotation while sharing an identical `Physical_state`; **39 groups / 38 entries** are unambiguous. Verbatim: bmr15303 *"bb-PDI monomer, major conformer"/"minor conformer"*; bmr26987 *"HIV1 protease monomer, folded"/"monomer, unfolded"*; bmr19491 Galectin-3 *"trans form (major)"* + 4 *"cis form"* rows; bmr51008/51009 Aβ40 major/minor; bmr51635/51636 MYC trans/cis; bmr52628 p14ARF state 1/2 | **new, material** |
| — | **15711 SOD1** gets `is_homo_oligomer=True, n_copies=2` under the proposed rule. Its `_Entry.Title` is *"Backbone chemical shift assignments for **monomeric** apoSOD1"* and its two rows are named *"chain 1, proline cis conformer"* / *"chain 2, proline trans conformer"*. **It is in the published `test_chezod117` split** | **new, material** |
| `n_copies` is "stoichiometry as deposited" | Often flatly wrong. Using `Magnetic_equivalence_group_code`: bmr27605 Nek2 LZ D3=4, meq `[1,1,2,2]`, Details *"two parallel, symmetric coiled coils that adopt two conformations"* → a **dimer**; bmr6743 CcdA D3=6 → dimer in three conformers; **bmr19377 IL-10 D3=3 → 2, and it is a published `test_trizod` row** | **corrected** |
| bmr27605 is a "leucine-zipper tetramer" and bmr51492 a "TDP-43 dimer" — cited as reasons to reject a `Conformational_isomer` refinement | Both misread. 27605 is a dimer (its own Details say so). 51492's rows are *"TDP-43, form A"/"form B"* with `Physical_state='partially disordered'` — two conformational forms, not stoichiometry. **The conclusion still stands** (7091 GroES ×7, 53193/53194 PF4 ×4, 6851 XPA ×2 all set the tag while being genuine oligomers) but the supporting examples must not be quoted to Sandro | **corrected** |
| flagship claim: per-state grouping catches folded/unfolded depositions | Disproven within its own family: **bmr51327** lists *"drkN SH3 folded state"/"unfolded state"* with the **same** `Physical_state`, so it survives — while 25500/25501 (same protein, same experiment) are demoted only because those depositors filled the field in. **The discriminator is depositor formatting, not biology** | **corrected** |

**Consequence: the recommended classifier changes.** Do **not** ship the naive per-state rule under the name `is_homo_oligomer`.

#### Proposed patch

1. **Parse three more `_Entity_assembly` tags** in `bmrb.py` (~4 lines): `Entity_assembly_name`, `Conformational_isomer`, `Magnetic_equivalence_group_code`. Costs a re-parse of 17,388 `.str` files (~20 s at 10 workers `[joint/parse_cost]`), which is required for C4 anyway.
2. **Ship `n_entity_assembly_rows`** — raw, no interpretation.
3. **Derive `n_copies` conservatively**, in priority order: (a) if `Magnetic_equivalence_group_code` is populated for all rows of the entity, `n_copies` = largest group size; (b) else count per `(Entity_ID, Physical_state)`; (c) demote when repeated rows' `Entity_assembly_name`/`Details` differ only by a conformer token (`cis|trans|major|minor|conformer|isomer|form A|B|state 1|2|folded|unfolded`). Rule (a) fixes 19 groups including two published test rows; rule (c) removes 39 unambiguous FPs. Expect ~53 residual ambiguous groups.
4. **`--exclude-homo-oligomers`, default OFF**, all four tiers, uniform (a tier-conditional composition rule would break the `strict ⊆ moderate ⊆ tolerant ⊆ unfiltered` nesting `build.py`'s `tier_rank` relies on).
5. `n_copies` must be **NA, not 1**, when the composition entry errored — `build.py`'s except branch writes `{"is_bound": True, "error": …}` and those rows would otherwise read as confidently monomeric.

#### Tier policy

Metadata in all four tiers; **no tier's default membership changes**. The tiers grade NMR data quality; oligomeric state is an orthogonal biological axis. Folding it into `strict` would make `strict` mean something different from what the manuscript and datasheet say.

#### Tests

`test_homodimer_flagged` · `test_alternative_states_not_oligomer` · **`test_two_native_one_unfolded_not_oligomer`** (the bmr19104 3-rows-2-states shape — the originally proposed 2-rows-2-states test passes while the real-world shape fails) · **`test_conformer_named_rows_not_oligomer`** (the bmr15711 apoSOD1 shape) · `test_ligand_repeat_not_oligomer` · `test_hetero_complex_unchanged` · real-entry parametrisation rebuilt **from raw-file evidence**, not from the investigator's list. Assert a **required subset** of `detect_bound()` keys, not the exact key set — the exact-set test breaks on every future additive change.

#### Risks

- `is_homo_oligomer` is **both a lower bound and an upper bound**: it under-counts (~10–28 under-annotated homodimers; bmr15021's assembly is literally named `homodimer` with one row and `Number_of_components=2`) and over-counts (conformer rows inflate it). The datasheet must state both directions. The investigator documented only the first, because it did not know about the second.
- Naming: `is_homo_oligomer` over-claims for a field that means "this entity appears on more than one `_Entity_assembly` row". A consumer taking `is_homo_oligomer == False` as a monomeric guarantee would simultaneously discard monomeric MYC, MAX, Aβ40, p14ARF, Galectin-3 and apoSOD1.

---

### 3.4 C4 — entity classification (`trizod/dataset/composition.py:24, :42`)

#### What is wrong

```python
has_metal = any((e.type or "").lower().startswith("metal") for e in entities)   # :24
bound = has_non_polymer or has_nucleic or has_metal or multi_protein_assembly   # :42
```

**Zero of 21,842 entity records** has an `_Entity.Type` starting with `metal` `[meta]`. Sandro's histogram — `polymer 18897 / non-polymer 2926 / water 16 / null 1 / D-SACCHARIDE 1 / SACCHARIDE 1` — reproduces **exactly**. `has_metal` is `False` on all 16,963 rows of the shipped `_composition_cache.csv` `[data]`, and `build.py`'s `keep_cols` doesn't even export it.

`has_nucleic`'s two-value exact match misses `polydeoxyribonucleotide/polyribonucleotide hybrid`.

#### Evidence

- Independent reimplementation of `detect_bound()` reproduces the shipped cache on **all 16,963 entries, 0 mismatches** on all six fields `[meta]` — every number below describes the released artefact.
- `_Assembly.Metal_ions` is a **count** tag: `{null 13683, '0' 2932, '1' 243, '2' 65, '3' 26, '4' 23, …}`. Only 360 entries have >0, vs 1,661 real metal entries — it **under-reports 4.6×**. Reject as the primary source `[meta]`.
- `_Entity.Nonpolymer_comp_ID` is populated on **2,763/2,926 = 94.4 %** of non-polymer entities, 536 distinct PDB chem-comp codes (ZN 773, CA 302, MG 133, HEC 120, HEM 84, FES 33 …), and is **not parsed** by `bmrb.py` `[meta]`.
- Metal fix membership impact: **0 rows in every tier**. All 1,661 metal entries are already `has_non_polymer` — but this is **true by construction**, not an empirical property (see corrections).
- Hybrid fix: `has_nucleic` 1,235 → 1,243 entries; `is_bound` 4,558 → 4,562. The 4 newly-bound entries (17351, 19226, 30184, 34228) are protein-free and appear in **zero** tiers `[meta]`.

#### Verifier corrections — stated plainly

| investigator said | verifiers measured | status |
|---|---|---|
| "16 hybrid entity records in **8** entries" | **14** entries. The 8 listed are those whose `has_nucleic` label *flips*; 30105, 30113–30117 also carry hybrids but already have `has_nucleic=True` via a co-deposited plain DNA entity | corrected |
| the stale-pickle `AttributeError` guard "turns a silent wrong answer into a hard failure — intended" | **The opposite.** `build.py:69–74` wraps `detect_bound` in a bare `except Exception` storing `{"is_bound": True, "error": …}` with no output. Simulating a stale cache: **2,558 entries** take the except path, `_composition_cache.csv` is written with NaN on every composition column, no `KeyError` fires, and **the build completes silently** with `has_metal=True` on 0 of 16,963 rows | **corrected — the fix's own safety mechanism is inert.** Both verifiers |
| "Containment proof: has_metal ∧ ¬has_non_polymer = 0 over all 16,963 — 0 exceptions" | A **tautology**: `metal_ids ⊆ ligand_comp_ids`, which is built only from `type == 'non-polymer'` entities. The conclusion (zero membership change) is right but follows from the code, not the corpus. The proposed `test_metal_is_subset_of_non_polymer` **can never fail** | **corrected.** Drop the test |
| `has_other_nonpolymer_type` makes "19 SACCHARIDE/null-typed entities visible" | **3.** 16 of the 19 are `water`, which the rule explicitly excludes | corrected |
| "Zero false positives found" in the metal comp-ID list | Wrong test — the list's problem is **false negatives**, measurable from `_Chem_comp.Formula` (same files, never consulted): **≥16 entries** missed — ZNH/ZEM (Zn-protoporphyrin IX, the direct analogue of HEM which *is* listed), GIX (Ga-porphyrin), 7BU, 9F0/H9C/LN8 (Pt), RUL/3UQ (Ru), RE1O1, BF2, and comp ID `3` = MAGNESIUM ION in bmr17610. **64 of the 107 proposed codes never occur in the corpus** | **corrected.** Use `_Chem_comp.Formula` as the primary rule |
| `_Assembly.Metal_ions` is "only a cross-check" | It is the **only** signal for 6 metal-bearing entries currently **in** the dataset: 50235 (cIAP1-Bir3) and 50588 (WRKY1-N) are in **all four tiers including strict**; 50635, 52090, 53330, 6698 are caught by nothing else. Precision caveat: 50635 is titled *"gallium binding peptide C3.15 (**WITHOUT** gallium)"* | **corrected — this is where C4 touches membership** |
| water handling | Inconsistent: water is excluded from `has_other_nonpolymer_type` on the grounds that it is not a ligand, but still counts toward `multi_protein_assembly`. **Three protein-only depositions are discarded from every tier solely because a WATER entity is listed**: bmr25640 (oxidized horse-heart cytochrome c, 105 aa), bmr34125 (cytotoxin-1, 60 aa), bmr34240 (engrailed homeodomain, 64 aa). Excluding water from the assembly member set **adds** 3/3/2/0 build rows | **new — the only membership-gaining correction in C4** |
| regression test pins kept-row counts 11307/10471/7690/2375 | Those are **stale** (2026-07-14). Current: 11307/10477/7997/2587. Also `wc -l` gives 11314/10478/7691/2375 because `entity_name` contains embedded newlines — the test must use a real CSV reader | corrected |

**UNRESOLVED:** the count of Entity_ID-guard-hidden non-polymer entities safe to un-hide — 7 (verifier 1) vs 9 (verifier 2, adding 26756/26757 `CHAPS`). The difference is which token vocabulary is applied; both are correct for their own vocabulary. Pin the vocabulary first.

#### Proposed patch

1. `bmrb.py`: parse `_Entity.Nonpolymer_comp_ID` and `_Entity.Nonpolymer_comp_label`.
2. `composition.py`: metal rule = `_Chem_comp.Formula` element scan (**primary**, covers 2,766/2,926) → comp-ID allow-list (tie-break) → explicit `METAL_ENTITY_NAMES` (163 entities without a comp ID; recovers 61). Emit `has_metal`, `has_metal_ion`, `has_metal_cofactor`, `metal_comp_ids`, `nonpolymer_comp_ids` (controlled vocabulary only, blank when absent) and `ligand_names` (free text) as **separate columns** — the proposed single `ligand_comp_ids` mixes them and 9 of its 73 fallback tokens collide with genuine comp IDs (`3`, `CA`, `ZN`, `HEM`, …).
3. `NUCLEIC_POLYMER_TYPES` allow-list including the hybrid; drop the `e.type == "polymer"` guard so a missing Type tag cannot hide a nucleic acid.
4. Exclude `water` entities from the assembly member set feeding `multi_protein_assembly` (+3/+3/+2/0 build rows).
5. Key the "unrecognised Type" rule on **declared ligand evidence** (bmr6123 `CTO`, bmr7114 `BCD`, bmr16669 `$chem_comp_GTPgS`), not on the Type value being unrecognised. Ship as a **label**, not in `bound`.
6. **Prerequisite (PR0):** narrow `build.py:69–74`'s bare `except Exception`, or add an up-front assertion in `build_composition_cache()`.

#### Tier policy

Uniform across all four tiers. `detect_bound()` runs once per entry and `is_bound` is applied as a single universal mask before tiers are split (`build.py:188`). No new stringency knob.

#### Tests

`test_hybrid_polymer_type_counts_as_nucleic` · `test_metal_flag_actually_fires` (`ZN` → True; `GDP`/`ATP` → False but present in ligand columns) · **`test_metal_matches_chem_comp_formula`** — for every non-polymer entity with a resolvable Formula, assert `has_metal == bool(metal elements in Formula)`. This single test fails today on the proposed comp-ID list for the 16 entries above · `test_water_is_not_a_ligand` · **`test_build_surfaces_classifier_errors`** — assert the error branch is not hit, or `sum(has_metal) > 0` after a build.

#### Risks

- Purging `tmp/bmrb_entries/` is **safe**: every pickle was written 2026-07-13 14:57, seven minutes after the last commit touching `bmrb.py` (`661298d`, 14:50:14), and nothing has touched that file since. Re-parsing reproduces identical `Entity` state plus the new attributes.
- Backward-compat: `has_metal` flips `False→True` on 1,661 entries. `_composition_cache.csv` is not committed and not in the release bundle, so no published contract breaks — but a notebook filtering `has_metal == False` as a no-op suddenly excludes ~10 % of entries.

---

### 3.5 C5 — experimental-method whitelist (`trizod/pipeline.py:117–140`)

#### What is wrong

Two separate defects.

**(i) The `""` sentinel becomes a regex wildcard.** `whitelist = ['', 'solution', 'structures']` builds the pattern `"|solution|structures"`, whose empty alternative matches **every** string. The tolerant and moderate whitelists are therefore **dead code** — only the `solid` blacklist bites. 27 tolerant / 18 moderate rows with subtypes `X-RAY DIFFRACTION`, `THEORETICAL`, `STATE`, `1H-15N-HSQC`, `Magic angle spinning NMR` pass today `[harness]`.

**(ii) `method_sel &= ~pd.isna(subtype)` drops the whole null-subtype class from strict.** 4,466 rows / **4,103 entries** `[harness]` — 90.8 % of everything the strict method filter rejects.

#### Evidence

- **No spelling is missing.** All 37 observed `(exp_method, exp_method_subtype)` combinations were enumerated; every real solution-NMR spelling (`solution`, `SOLUTION`, `SOLUTION NMR`, `Solution state`, `NMR, N STRUCTURES`) already passes strict via the two substrings `[harness]`.
- 97.5 % of null-subtype entries have `_Sample.Type == 'solution'`; **zero** carry solid evidence in `_Sample.Type` or `_Experiment.Sample_state`; exactly one (BMRB 5815, fd bacteriophage coat protein) is caught by experiment name (`PISEMA`, `Inversion and Spin Exchange at the Magic Angle`) `[meta]`.
- Era argument: **90.7 %** of null-subtype entries are BMRB ID < 7000 (pre-2006), while **94.8 %** of solid-state-labelled entries are ID ≥ 15000 `[meta]`.
- Recovery: **+438 rows / 423 entries** unrestricted `[run]`; **+434** with `require-solution` restricted to null subtypes `[harness]`. Baseline reproduction is exact — the same harness produces 3,514 ID-for-ID.
- **All 438 recovered rows have bit-identical `(T, pH, ionic_strength)` and bit-identical `gscores` versus the moderate tier** `[run]`. No rescoring.
- Completeness proof: running the strict prefilter over all 17,843 candidate rows with the method filter neutralised, 33 rows outside moderate pass every strict criterion — and **all 33 carry an explicit `SOLID-STATE` subtype**, i.e. they are blacklisted in both tiers. **+438 is a hard ceiling, not a sample** `[harness]`.
- Referencing quality of the recovered pool is **better**, not worse: candidate pool frac(max\|offset\|>2) = 0.080 vs 0.139 for the informative-subtype control `[harness]`.

#### Verifier corrections — stated plainly

| investigator said | verifiers measured | status |
|---|---|---|
| net recovery +420 rows / 407 entries (calibrated +426) | **+438 / 423** by real pipeline run. The calibration machinery (r = 1.0150, "52 false negatives") was unnecessary — the experiment is a 6-minute cache-hot run | **corrected** |
| tolerant and moderate both lose 28 rows | Impossible — moderate is a strict subset of tolerant. Bugfix alone: **−27 tolerant / −18 moderate**. With the proposed `reject-solid` fallback: **−13 / −7**, because the fallback **re-admits 14 of the 27** (all X-RAY DIFFRACTION rows, THEORETICAL, 1H-15N-HSQC) whose `_Sample.Type` is `solution` | **corrected — the plan's own fallback cancels half its own bugfix** |
| BMRB 5815 is among the 28 removed | 5815 is in **neither** the released tolerant nor moderate tier — double-counted | corrected |
| "the 407 recovered entries: none are ≥ 15000" | **3 rows / 6 entries** are (16206, 25264, 30663) | corrected |
| widening `'structures'` → `'structure'` recovers 3 rows | **0.** None of the `NMR, 1 STRUCTURE` rows appear in the 443-row delta | **corrected — drop the change** |
| the ~800 gap is `unit-assumptions=False`; relaxing it gives +1,141 | `--unit-assumptions` alone gives **+7** rows. Relaxing `--default-conditions` too gives **+1,184**. And holding the method filter fixed, `--unit-assumptions` alone recovers **+793 rows** — a near-exact match for Sandro's ~800 | **corrected. The bottleneck is `default-conditions`, and Sandro's number is a different filter entirely** |
| solid-state rows currently in strict: 2 rows / 1 entry (25289) | **3 rows / 2 entries** — 27211 (*"Solid-state NMR assignment of P. horikoshii TET2"*, all 5 samples `_Sample.Type=solid`) also sits in strict with subtype `solution` | **corrected** |
| the diff sketch | `def prefilter_dataframe(df, method_whitelist, method_blacklist, method_fallback="off", temperature_range, …)` is a **SyntaxError** (verified with `py_compile`) — the sketch was never executed | corrected |
| bump the `3514` constant in `test_full_dataset_regression.py` | No such constant exists; the test diffs against `data/interim/baseline/*.json` (dated 2026-03-09, pre-#17 and pre-#20 — **it already fails today**) | corrected |
| a new `sample_state.py` classifier with tri-state policy | Verifier 2: the whole classifier is worth **+2 net rows** over adding `''` to the strict whitelist cell. Its 6-row "gain" is 4 junk-subtype rows (`X-RAY DIFFRACTION`, `THEORETICAL`, `NMR`, `STATE`) admitted to **strict** on `_Sample.Type=='solution'`, minus 2 legitimate solution rows it wrongly drops (5813 `bicell_solution`, 6040 `micelles`) — because `_SOLUTION_SAMPLE` is fully anchored (`^…$`) and so violates its own docstring's promise to treat bicelles/micelles as solution | **material** |

#### Proposed patch

```python
-    if "" in whitelist_lower and "" not in blacklist_lower:
-        method_sel |= ...pd.isna(df.exp_method_subtype)
-    else:
-        method_sel &= ~pd.isna(df.exp_method_subtype)
-        missing_vals &= ~pd.isna(df.exp_method_subtype)
+    # "" is a SENTINEL meaning "accept a missing subtype". It must never reach
+    # str.contains(): the empty alternative in "|solution|structures" matches
+    # EVERY string, which silently turned the tolerant/moderate whitelist into
+    # a no-op -- only the blacklist was doing any work.
+    allow_missing = "" in [e.lower() for e in method_whitelist]
+    wl = [e.lower() for e in method_whitelist if e]
+    ...
+    # Fallback applies ONLY when the subtype IS NULL, never when it is present
+    # but uninformative -- otherwise 'X-RAY DIFFRACTION' and 'THEORETICAL'
+    # rows enter strict, and the tolerant/moderate bugfix cancels itself.
```

Ship the evidence classifier as a **column** (`sample_state_evidence`) and as the **strict** null-subtype gate, but do not apply it to present-but-uninformative subtypes. Anchoring in `_SOLUTION_SAMPLE` must be relaxed so `bicell_solution`, `micelles`, `liquid crystal` resolve to `solution` (they are isotropic solution NMR).

Additionally, ship the **refined solid veto** (solid `_Sample.Type`/`Sample_state`/experiment-name evidence **and** no solution-type experiment names), which removes 25289 ×2 and 27211 at 100 % measured precision. Do **not** ship the naive veto — it removes 8 strict rows of which only 3 are genuinely solid (34067 *"Solution structure of the RBM5 OCRE domain"*, 36119, 30293, 34330, 16340 are real solution structures with a mis-typed `_Sample.Type`).

#### Tier policy

unfiltered `off` (preserves "unfiltered = raw corpus"; a fallback here costs exactly 1 row, 5815) · tolerant/moderate `reject-solid`, **null-subtype only** (−13 / −7) · strict `require-solution`, **null-subtype only** (+434).

#### Tests

`test_empty_whitelist_term_is_a_sentinel_not_a_wildcard` — 4-row frame `['X-RAY DIFFRACTION','solution',None,'SOLID-STATE']` → `[False, True, True, False]`. **Fails on develop today.** · `test_fallback_only_applies_to_null_subtype` · `test_alignment_media_are_solution` (bicelle, liquid crystal, micelle) · `test_real_entries` — 5815 → solid, 25289 → solid, **27211 → solid**, 6191/10035 → solution · assert no strict row has `sample_state_evidence == 'solid'`.

#### Risks

- **Composition shift, and this is the sentence Reid and Iva will react to first:** the released strict tier contains **exactly 1** entry with BMRB ID < 7000. After recovery it is **342 of ~3,940 (8.7 %)** `[harness]`. Referencing quality is measurably fine, but the shift must be stated in the datasheet, not discovered by a reviewer.
- The honest framing of what strict now means: "declares solution NMR, **or** declares nothing and `_Sample.Type` says solution". `_Sample.Type` is a required, default-filled field (16,250 entries say `solution`), so the positive-evidence guard excludes only 5 of 443 rows. It is nearly inert; the real protection is the era argument.
- **`train_tier == strict` in the published Parquet changes meaning across a concept-DOI version bump.** Same column, same label, different guarantee. Needs an explicit datasheet sentence.

---

### 3.6 C6 — side-chain shifts (`trizod/bmrb/bmrb.py:747`)

#### What is wrong

`get_valid_bbshifts()` whitelists 12 atom IDs `{C,CA,CB,H,HA,HB,N,HA2,HA3,HB1,HB2,HB3}`. Everything else — every side-chain carbon, methyl, aromatic ring atom and side-chain nitrogen — is dropped and never reaches `fill_row_data()`, `output_dataset()`, the `.str` writer or the Parquet.

**And the release ships no shifts at all.** `trizod_dataset.parquet` (9.07 MB) has 42 columns: scores, offsets, metadata. Raw shifts exist only in the optional 1.4 GB `.str` bundle. `scripts/build_dataset.sh` never passes `--include-shifts`.

**And `--include-shifts` is dead code** — `trizod/trizod.py:368-371` assigns `shifts` only inside the `no_shift_averaging` branch, so `--include-shifts` alone emits zero shift columns. Verified by direct call `[run]`.

#### Evidence

- **3,458,852** side-chain shifts discarded on the unfiltered tier (tolerant 3,360,929 / moderate 2,863,031 / strict 1,069,688); nucleus split 2,449,316 ¹H / 946,576 ¹³C / 62,960 ¹⁵N `[meta]`. Median 109/chain (strict 310); 75.0 % of chains carry ≥ 1 (strict 80.0 %).
- Per-residue side-chain completeness: 0.399 / 0.415 / 0.480 / 0.527 across tiers. Methyl-TROSY (ILVMA) ¹³C coverage 46.7 % → 64.3 %. Aromatic ring ¹³C/¹⁵N: PHE 0.155 → 0.247 `[meta]`.
- Data hygiene is excellent: 99.96 % have `Val_err ≤ 1.3` or missing; only **26** duplicated `(chain, seq_id, atom)` keys corpus-wide, 19 conflicting, confined to 1 chain `[meta]`.
- Ambiguity codes must **not** reuse the backbone rule: applying `ambc ∈ {1,2,'','.'}` would discard 107,774 values (3.08 %), including **all 85,831 aromatic ring-degenerate** (code 3) values `[meta]`.
- Companion Parquet sizes at zstd-19 (same codec as the deposit): side-chain flat long table **11.89 MB**; nested list<struct> 10.50 MB; all shifts 43.39 MB `[est — pyarrow is not a project dependency and this was not independently re-measured]`.
- Bonus cross-finding, real and in the **strict** tier today: `17640_1_1_1` SER71 CB = **−939.28 ppm** (and THR161 CB = −939.28), `51128_1_1_1` LEU24 HB3 = **1579.0 ppm** `[data]`. Both carry `Val_err = 0.0000` and `Ambiguity_code = 1`, so they pass both existing guards and reach scoring. Impact is bounded by `np.minimum(diffs, 4.0)` at `scoring.py:165` — one saturated atom per residue — but they are in the published dataset.

#### Verifier corrections — stated plainly, because this one matters

| investigator said | verifiers measured | status |
|---|---|---|
| ship `val_rereferenced = val − (off_CA + lacs_off_CA)` for ¹³C, justified by r = 0.925 | **`off_*` is not in ppm.** `compute_offsets` returns `nanmean(weighted_diffs)` where `weighted_diffs = diff_arr / REFINED_WEIGHTS` (`scoring.py:115, 150-153`), so `off_A` is in σ units; `lacs_off_A` **is** raw ppm (`scoring.py:263`). The proposed column adds two different units. Empirical proof: max \|off_HA\| = **25.22** — impossible as ppm for a proton; × `REFINED_WEIGHTS['HA']` = 0.66 ppm, sane. Shipping it would inject up to **21.2 ppm** of pure unit error; 5.5 % of chains >0.5 ppm, 2.9 % >1 ppm | **REJECTED. Both verifiers, independently.** |
| r = 0.925 refutes Sandro's premise that re-referencing cannot be applied to side chains | The correlation is carried by **one chain**: removing 25442_1_2_2 (`lacs_off_CA = 300.12 ppm`) drops r to **0.353**; top-1 % removal → 0.602; Spearman is **0.531**; for the 79.8 % of chains with \|offset\| < 0.5 ppm, r = **0.113**. Redone in correct units by verifier 2: `lacs_off_CA` **alone** gives r = 0.963, slope 0.981, RMS residual 0.953 ppm; `off_CA` alone gives r = **0.009**. The proposed sum gives RMS 1.390 ppm — **46 % worse than LACS alone**. Binned: the correction makes things **worse** for 7,357 of 7,955 chains and 44.6 % of chains end with a *larger* absolute residual than doing nothing | **REJECTED. The claim "REFUTES Sandro's premise" is withdrawn. For the regime nearly all users occupy, Sandro is right.** |
| the 0.932 slope raises a question for Reid ("close enough to 1.0, or scale it?") | The slope is 0.981 once units are fixed. **The question is manufactured by the arithmetic error.** Do not send it | corrected |
| ¹H and ¹⁵N "do not transfer" (r = 0.088, −0.009) | Tested against the **carbon** offset — a strawman. Against the proper proton offsets, side-chain ¹H vs HB offset r = **0.331** (0.358 in the \|off\|<1 ppm regime), ~4× the reported figure. ¹⁵N vs `off_N` r = 0.045. The ¹⁵N conclusion survives; the ¹H figure does not | corrected |
| `OUTLIER_NUCLEUS_IMPOSSIBLE`, ranges "wider than any physically reported protein shift" | **It is a paramagnetic-protein detector.** 359 of its 393 ¹H hits (**91.3 %**) are in entries whose title/ligands name a paramagnetic or metallo system — an 18.8× enrichment. Verified by title: 3084 cobalt carbonic anhydrase; 811/812/814 Cu-Zn SOD; 330/1192/1783–1789 ferricytochrome c (1783–1789 are literally titled *"hyperfine-shifted resonances"*); 1808 cyanide-inhibited HRP. Only 6.9 % carry BMRB's `paramagnetic` tag | **REJECTED as specified.** Rename to `OUTLIER_OUT_OF_TYPICAL_RANGE` and widen ¹H to ~[−40, 120] |
| k = 10 flags "far outliers", 0.30–0.41 % | **~30 % of the flag list is real physics**: 20.5 % paramagnetic + 9.6 % ARG NE aliasing (which the investigator itself concedes are correct measurements, merely folded). Handing k = 10 to Sandro without that number will make him delete good data | **corrected** |
| stats table = 206 `(comp, atom)` rows, 11 with n < 30 | On the actually-dropped set: **154** rows, **14** with n < 30. The proposed extractor and the proposed test assert different things | corrected — internally inconsistent |
| "discarded at exactly one line" | `bmrb.py:741` (`max_err ≤ 1.3`) and `:744` (ambiguity whitelist) run **before** `:747` | corrected |
| `sidechain_outlier_frac` policy: ">20 % drops 196 chains / 4,888 shifts" | Corpus-wide, not tier-restricted. Within the unfiltered tier: **67 chains / 3,303 shifts**. And 27 of those 67 have fewer than 10 side-chain shifts total. The column is **undefined for 25.0 %** of unfiltered rows | corrected |

**New finding neither party had, worth its own triage item:** `lacs_off_CA = 300.12 ppm` on entry 25442 is not a referencing offset — it is a broken shift table or a broken LACS fit. Maxima across the corpus: CA 300.12, N 300.58, C 160.73 ppm `[data]`. That is a latent LACS quality issue independent of C6.

#### Proposed patch — reduced scope

1. **`--include-shifts` bugfix** (`trizod/trizod.py:368-371`), one line, ships alone and immediately. Column-to-atom mapping verified: `get_valid_bbshifts` fills column *i* with `BACKBONE_ATOMS[i]`, so `list(BACKBONE_ATOMS)` in the averaging branch is correct names in correct order.
2. **`bmrb.get_sidechain_shifts(shifts, seq)`** — parallel read of the same `ShiftTable.shifts` tuples; reuses the Seq_ID / Comp_ID / Val guards but **not** the atom whitelist, `max_err` cut or ambiguity whitelist. `get_valid_bbshifts()` untouched → scoring bit-identical.
3. **Companion Parquet** `trizod_sidechain_shifts.parquet`: `id, seq_id, comp_id, atom_id, atom_type, val, val_err, ambiguity_code`. Flat long, sorted by `(id, seq_id, atom_id)`, zstd-19. Joins 1:1 on `id` with zero orphans (verified: unfiltered is an exact superset of every other tier, and the released Parquet has all 16,851 rows with the identical id format).
4. **NOT shipping:** `val_rereferenced`, `sidechain_outlier_frac` on the main table, the `OUTLIER_NUCLEUS_IMPOSSIBLE` bit as specified, the wheel-shipped stats table, and the maintainer-only regeneration subcommand. The outlier flagger is a **separable, contestable second PR** — the evidence supports the raw table now.
5. **Documentation fix:** correct `off_*` units in the v0.3.0 README column table, `docs/dataset/datasheet.md:122` ("max-offset … 3/3/2 **ppm**" — it is σ units), and `trizod/io/str_writer.py:41-42/113-117` (`Offset_ppm` tag fed from `row['off_{atom}']`).

#### Tier policy

Tier-independent. Emitted once at unfiltered scope; users subset via `split`/`train_tier`. Measured justification: the k = 6 flag rate is 1.133 / 1.100 / 1.109 / 1.119 % across tiers — flat. But note the *coverage* is strongly tier-dependent (median 109 → 310), which is a different and more useful fact for Sandro.

#### Tests

`test_include_shifts_without_no_averaging` (fails on develop) · `test_get_sidechain_shifts_disjoint_from_backbone` · `test_scoring_unchanged_by_sidechain_extraction` (bit-identical z/g/k/offsets) · `test_sidechain_ambiguity_preserved` (code 3 survives) · `test_sidechain_parquet_join` (every `id` in the main table; `sequence[seq_id-1] == one_letter(comp_id)`).

**Do not** write the proposed `n_backbone + n_sidechain == raw row count` assertion — `get_valid_bbshifts` returns `None` for the *entire* table on an AA mismatch (`bmrb.py:722`) or conflicting duplicates (`:773-778`), so it is false for exactly those chains.

#### Risks

- A reviewer can argue that publishing 3.46 M side-chain values with no referencing assessment is worse than not publishing them. Mitigation: `lacs_off_*` already ships in the Parquet and joins on `id`, so the ~600 chains that genuinely need a correction can self-serve. The datasheet must say plainly: **side-chain values are AS DEPOSITED, with no referencing correction.**
- Stereo swaps are invisible to any marginal rule: 34.7 % of 59,777 LEU CD1/CD2 pairs and 41.3 % of 48,884 VAL CG1/CG2 pairs have the inverted ordering. The honest handle is `Ambiguity_code == 2` (29.49 % of the dropped set), which is already deposited — carry it as a column plus one datasheet sentence.
- Adding the companion is safe for `test_pipeline_regression` (field-by-field comparison, not byte-wise). The real schema break is downstream: the Parquet packaging step and any strict-schema consumer of `scores.json`.

---

## 4. Execution plan

Nine PRs. Dependencies:

```
PR0 ─┐
PR1 ─┼─ land immediately, zero membership change, no regen
PR2 ─┘
       PR3 ─┐
       PR4 ─┼─ metadata / feature, no scores.json membership change
            │
       PR5 ──► PR6 ──► PR7 ──► PR8
       (bugs)  (policy) (regen) (RFC)
```

| PR | Title | Scope / files | Tests | Regen? | Effort |
|---|---|---|---|---|---|
| **PR0** | `fix(build): surface composition-classifier errors instead of swallowing them` | `trizod/dataset/build.py:69-74` — narrow the bare `except Exception`, or assert up front in `build_composition_cache()`. **Blocks PR3.** Without it a stale-pickle run writes NaN composition metadata on 2,558 entries and completes silently | `test_build_surfaces_classifier_errors` | no | **0.5 d** |
| **PR1** | `fix(output): --include-shifts emits nothing without --no-shift-averaging` | `trizod/trizod.py:368-371` | `test_include_shifts_without_no_averaging` (fails on develop) | no | **0.5 d** |
| **PR2** | `docs: off_* offsets are in sigma units, not ppm` | v0.3.0 README column table, `docs/dataset/datasheet.md:122`, `trizod/io/str_writer.py:41-42/113-117`. Also `docs/filtering.md:25` ("mentioned anywhere in the BMRB file" — false) | doc-only | no | **0.5 d** |
| **PR3** | `feat(composition): revive has_metal, catch DNA/RNA hybrid, exclude water from assembly count, emit oligomer + ligand metadata` (C3 + C4) | `bmrb/bmrb.py` (parse `Nonpolymer_comp_ID`, `Nonpolymer_comp_label`, `Entity_assembly_name`, `Conformational_isomer`, `Magnetic_equivalence_group_code`), `dataset/composition.py`, `dataset/build.py` keep_cols, `scripts/build_parquet_dataset.py`, `cli/main.py`. **Requires purging `tmp/bmrb_entries/`** (~20 s re-parse at 10 workers) | new `tests/test_composition.py` (~12 cases, incl. **bmr19104** 3-rows-2-states, **bmr15711** apoSOD1 cis/trans, **bmr19377** IL-10 meq, bmr17351 hybrid, ZN/HEM/ZNH via `_Chem_comp.Formula`, water) | `_composition_cache.csv` + `<tier>_all_ranked.tsv`; **build stage +3/+3/+2/0 rows from the water fix**. ⚠️ This build also silently absorbs the pending post-#20 regen (strict build pool 2,375 → 2,587) — state it in the PR body or the diff is unattributable | **3 d** |
| **PR4** | `feat(sidechain): companion side-chain shift table` (C6, no re-referencing, no flagger) | new `trizod/sidechain/`, `bmrb/bmrb.py::get_sidechain_shifts`, `scripts/build_parquet_dataset.py`, `dataset/package_release.py` | `tests/test_sidechain.py`; assert z/g/k/offsets bit-identical with and without | new companion Parquet (~11.9 MB) | **3 d** |
| **PR5** | `fix(prefilter): keyword fields exploded into characters; empty whitelist term acts as regex wildcard` (C1a + C5′) | `trizod/trizod.py` `fill_row_data`, `trizod/pipeline.py:117-140`. **Also update `scripts/filter_impact_report.py`** (calls both `create_peptide_dataframe` and `prefilter_dataframe`) | `tests/test_keyword_fields.py`, `tests/test_prefilter_method.py::test_empty_whitelist_term_is_a_sentinel_not_a_wildcard` — both fail on develop | **yes.** tol −29, mod −37, strict −50 `[harness]`. **Must not merge before Decision 1 (`interacti`) is made** | **2 d** |
| **PR6** | `feat(filter): admit undeclared-method entries at strict; denaturant list corrections; state metadata columns` (C5b + C2 + C1b column + physical_state deny + `interacti` decision) | `trizod/trizod.py` `filter_defaults` + `fill_row_data` + `output_dataset` column list, `trizod/pipeline.py`, `trizod/cli/main.py`, new `trizod/bmrb/sample_state.py`, `docs/filtering.md` | `tests/test_sample_state.py`, `tests/test_denaturants.py`, `test_keep_state_never_dropped_by_any_filter`. **Assert 5815, 25289 ×2, 27211 are NOT in strict** | **yes** — this is the +364 (or more). **Acceptance criterion: price the final configuration end-to-end with a real run before merge; the recommended config's strict count is currently unpriced** | **4 d** |
| **PR7** | `chore(release): regenerate scored tiers + dataset chain → v0.4.0` | run the chain; `data/interim/*`, `data/release/*`; regenerate `data/interim/baseline/*.json` (dated 2026-03-09, **already failing**); re-cut `tests/reference/*` | `test_pipeline_regression`, `test_full_dataset_regression`, `test_testset_pin` | **the regen** | **2 d** + review |
| **PR8** | `RFC: contested filter policy` — membrane mimetics, `molten globule`/`amyloid fibril` as free text, `bound`/`unbound` audit, field-scoping (exclude citation/struct metadata from the state blacklist), DMSO gate, solid veto scope | as PR6 | policy tests | second regen | **hold** |

**Why PR5 and PR6 are separate and must not be squashed:** PR5 only *removes* rows and is a pure correctness fix; PR6 *adds* 364+ and is policy. Keeping them separate makes the regeneration diff bisectable.

**Total: ~16 developer-days** plus regeneration and the figure work in §6.

**Re-run cost, measured** `[joint/cold, cold2, parse_cost]`:

| step | cost |
|---|---|
| cache-hot `trizod score`, one tier (real run, 3,514 rows out) | **240 s** — dominated by `create_peptide_dataframe`, **0 cache writes** |
| all four tiers cache-hot | **15–20 min** |
| `.str` re-parse (needed for PR3 only) | 3.1 GB at 26.4 MB/s → ~2 min serial, **~20 s at 10 workers** |
| cold POTENCI (`tmp/potenci` purged) | 0.075–0.089 s/row → **~2.5 min at 10 workers** |
| cold LACS + scoring (`tmp/wSCS` purged) | 0.015 s/row → **~25 s at 10 workers** |
| **worst case, everything cold, from raw `.str`** | **< 40 min** |

**No fix in the accepted set changes a single score value.** `cache.py:16` keys on `sha256(f"{seq}|{temperature}|{pH}|{ion}")`; `pipeline.py:388` keys on `{entry.id}_{stID}_{entity_assemID}_{entityID}_{rereference_mode}_v{scoring_cache_version()}`. Neither key sees any filter list, keyword, token, whitelist, composition flag or `max_offset`. Verified: all 438 C5-recovered rows have bit-identical `(T, pH, I)` **and** bit-identical `gscores` versus the moderate tier `[run]`.

*(Note the wSCS key is condition-blind, so "cache keys unchanged" is not by itself evidence that scores are unchanged. The gscore identity check is.)*

**Latent bug to fold in while touching prefilter/postfilter:** `trizod/pipeline.py:221` builds `any_offsets_too_large = pd.Series(np.full((df.shape[0],), False))` with a fresh `RangeIndex` and ORs it against index-aligned columns of `df`. It works today only because the pipeline's frame happens to carry a `RangeIndex`; any caller passing a filtered subset gets silent misalignment and a wrong `pass_post` (reproduced: 18/398 vs the correct 302/398). One-line fix: `index=df.index`.

---

## 5. Dataset regeneration + release plan

### 5.1 Pre-step (blocking)

1. Merge **PR0**.
2. Purge `tmp/bmrb_entries/`. The pickle cache is **unversioned** — `pipeline.load_bmrb_entries` unpickles whatever exists with no hash check, so PR3's new `Entity` attributes are a silent no-op against a warm cache.

### 5.2 The chain

`scripts/build_dataset.sh` is the spine but is **not sufficient** — three artefacts live outside it.

| # | command | writes | note |
|---|---|---|---|
| 0 | `trizod --filter-defaults <tier> --rereference-mode both --cache-dir tmp` ×4 | `data/interim/scored/<tier>/scores.json` | cache-hot, ~15–20 min total |
| 1 | `trizod dataset build` | `final_dataset/<tier>/*`, `_composition_cache.csv`, `final_dataset_summary.json` | reads `paths.scored` |
| 2 | `trizod dataset test-set` | `testset/TriZOD_test_set.fasta` | **default = emit the pin.** `--redraw` is destructive — never invoke |
| 3 | `trizod dataset redundancy` | `mmseqs/` | needs `mmseqs` in PATH; the only slow step after scoring |
| 4 | `trizod dataset representatives` | `train_<tier>_best.fasta`, `train_<tier>_clu_best.tsv`, `cluster_repr_overrides_<tier>.tsv` | |
| 5 | `trizod dataset package --version v0.4.0` | `release_bundle/…` + `MANIFEST.json` + **leakage gate** | |
| **6** | `uv run --with pyarrow python scripts/build_parquet_dataset.py` | `trizod_dataset.parquet` | **not in the shell script** |
| **7** | `trizod dataset deploy` | `data/processed/deploy/*.fasta` | UdonPred handoff; **not in the shell script** |
| **8** | hand-edit | `CHECKSUMS.txt`, `zenodo.json`, `CITATION.cff`, `docs/dataset/bundle-README.md` | `CHECKSUMS.txt` is generated by **no script in the repo** |

**`package_release.main()` hard-fails** unless all of these exist: `docs/dataset/bundle-README.md` (staged as the bundle README — this is the datasheet users actually get), and per tier ×4 `train_<tier>_best.fasta`, `train_<tier>.fasta`, `train_<tier>_clu_best.tsv`, `train_<tier>_clu.tsv`, `scores/<tier>/scores.json`, plus `test/CheZOD117_test_set.fasta` and `test/TriZOD_test_set.fasta`.

**`assert_no_leakage()` is a hard gate** (`package_release.py:61-99`): it raises if any record in any `train/**/*.fasta` shares an ID or an exact sequence with any record in `test/*.fasta`. A shared BMRB *entry* ID (different shift record) is only reported. This is why any test-set change forces the whole redundancy chain to re-run rather than be patched.

### 5.3 Test-set decision

**How the code actually works.** The test set is **pinned**, not redrawn. Default mode loads `trizod/dataset/pinned/TriZOD_test_set.fasta` (365 seqs, seed 42, 25 % sample, written 2026-07-14) and calls `resolve_pinned_testset(pinned, strict)` — matching **by sequence** against `final_dataset/strict/strict.fasta`. A pinned sequence keeps its ID if the ID is still in the strict pool, else takes the lowest-numbered current strict entry with an identical sequence, else is **dropped**.

**Measured impact:**

| scenario | test set | note |
|---|--:|---|
| today, no code changes `[rel/r1, r2]` | **350/365** | 15 already lost to the post-#20 rescore (52075, 18726, 16557, 17139, 10301, 50686, 50777, 17606, 34155, 26752, 50809, 30129, 11000, 25798, 18351) — all left strict on offsets/`max-offset`; all 15 remain in tolerant |
| C1 with `interacti` **kept** `[rel/r3]` | −6 more | all six are collateral |
| C1 with `interacti` **dropped** | **−0** | |
| C1 recommended (drop `interacti` + complex phrases + deny list) | −4 | |
| C2 denaturants only | −1 | |
| C2 + membrane mimetics, narrow (SDS/DPC/dodecyl) | **−10** | none substitutable |
| C2 + membrane mimetics, full 16-token | **−16** | none substitutable |
| C3 / C4 / C5 / C6 | −0 | |

**Two facts dominate:** dropping `interacti` costs **zero** test chains while keeping it costs **six**; and the entire remaining damage is the membrane-mimetic decision.

**Recommendation.**

1. **Never `--redraw`.** The seeded draw is not stable under a pool change even with seed 42 — `rng.sample(free_reps, n_sample)` depends on `len(free_reps)` and the sorted representative list, and the CheZOD-free cluster count moves with the pool (currently 1,298 clusters / 324 sampled / 431 member seqs from a 2,232-sequence pool; the pool is already 2,437 before any fix, and ~+438 rows after C5). A redraw produces a *different* 365 and destroys comparability with v0.3.0, with anything Sandro is training now, and with every DisProt/ROC number in the manuscript.

2. **Repair the pin so the test set stays at 365.** Change `testset.py:280` to resolve against the **tolerant** pool instead of strict (two lines), and emit a `label_tier` column marking the 15–16 chains that no longer meet strict criteria. All 365 pinned sequences are present in tolerant `[rel/r4]`. Justification: the test set is a set of *sequences*, and its defining property — CheZOD-disjointness at 30/80 — is a property of the sequences, invariant under any filter change. "Drawn from the strict tier" describes the 2026-07 *construction*, not an ongoing invariant; the per-residue labels come from `scores.json` either way.

3. **Fallback if that is judged too clever:** accept **349/365** (MINIMAL scenario) and document the 16 dropped IDs in the changelog. Do **not** silently ship a 336- or 349-chain set still called "the 365-sequence TriZOD test set".

4. **Close the ID-substitution trap either way.** C5 admits ~438 rows to strict. If a newly admitted sequence is *identical* to a pinned test sequence and has a lower entry number, `resolve_pinned_testset` substitutes the ID — the sequence is unchanged but the **ID changes**, breaking ID-based joins for anyone holding v0.3.0. Either pin IDs as well as sequences, or ship a `v0.3.0 → v0.4.0` test-ID mapping table in the bundle.

**What changes regardless:** the *training* sets move no matter what. Strict unique sequences go 2,232 → 2,437 on the baseline alone, plus C5's ~+438 rows, plus C1/C2 churn. Cluster membership, `clusterupdate` ordering and quality-best representatives all shift. Communicate that split explicitly: **test numbers stay comparable, training sets do not.**

### 5.4 Versioning and changelog

**v0.4.0 now, v1.0.0 at acceptance.** Zenodo versions are immutable and all remain available; the concept DOI `10.5281/zenodo.21309963` resolves to the latest, so v0.3.0 (record `21381698`) stays up permanently and anyone mid-flight can keep citing it. Staying on 0.x signals "pre-publication, still moving", which is honest. Cut v1.0.0 at acceptance, frozen, cross-referenced to the paper's DOI. Only the parenthetical "(current version v0.3.0)" at `Article.tex:323` needs the bump — the manuscript already cites the concept DOI, which is correct.

**One Zenodo record, plus an immediate non-DOI handoff build for Sandro** off `develop` after PR3/PR4/PR6-columns. This keeps the DOI graph clean while decoupling his timeline from the policy debate. See §7.5.

**Changelog** (ship as `CHANGELOG.md` in the bundle and in the Zenodo description):

1. **Filter-behaviour corrections** — (a) `_Citation_keyword`/`_Struct_keywords` were exploded into single characters and never matched (issue #23); (b) the `""` whitelist sentinel acted as a regex wildcard, disabling the tolerant/moderate experiment-method whitelist; (c) `has_metal` was dead code (no BMRB entity is typed `metal`); (d) `has_nucleic` missed `polydeoxyribonucleotide/polyribonucleotide hybrid`; (e) a `water` entity in the assembly wrongly made three protein-only depositions "bound"; (f) `--include-shifts` emitted nothing without `--no-shift-averaging`.
2. **Filter-policy changes** — `interacti` removed from strict (54 % collateral: it matched paper-topic labels like "protein–protein interaction" on free monomers); denaturant list corrected (TFE/DMSO added; TFA and potassium pyrophosphate removed as non-denaturants); `_Entity_assembly.Physical_state` now denies non-native states at tolerant+; entries with no `_Entry.Experimental_method_subtype` are admitted to strict on positive solution evidence (previously dropped wholesale for a tag 24 % of BMRB never filled in).
3. **New metadata columns** (additive; no existing column changes meaning): `physical_state`, `membrane_mimetic`, `sample_state_evidence`, `n_entity_assembly_rows`/`n_copies`, `has_metal*`, `nonpolymer_comp_ids`, `ligand_names`.
4. **New companion file** — side-chain chemical shifts (~3.5 M values previously discarded).
5. **Explicit statement of what moved** — per-tier counts, training representatives, cluster membership; and one unambiguous sentence about the test set with the ID list.
6. **Known carry-over** — v0.3.0's tier counts already reflected a superseded scoring run; v0.4.0 is the first release built on the post-#20 scores.
7. **Documentation corrections** — `off_*` is in σ units (÷ `REFINED_WEIGHTS`), not ppm; the v0.3.0 README column table and `str_writer.py`'s `Offset_ppm` tag both mislabelled it, and that mislabelling already caused one downstream analysis error.
8. **Known defects fixed in the data itself** — 25289 ×2 and 27211 (genuine solid-state depositions) removed from the strict tier.

---

## 6. Manuscript impact checklist

Read-only pass over `publication/manuscript/Article.tex` and `Figures/2026-06-22_TriZOD_redundancy_reduction.tex`. Nothing modified.

### A. Pre-existing errors — independent of C1–C6, fix regardless

| # | loc | issue |
|---|---|---|
| **A1** | `Article.tex:112` Table 1 | `min-backbone-shift-types` strict listed as **5**; code is **4** (changed in the Typer rebuild `e95aa7c`) |
| **A2** | `:256` | Consequential text: "moderate and strict filters require at least 3 and **5** backbone-shift types (so most windowed positions carry k ≥ 9 or **≥ 15** of the maximum 21)" → becomes 4 and k ≥ 12 |
| **A3** | `:118, 131-133` Table 1 + footnote b | "**15 agents**: … BME/2-ME, TFA, mercaptoethanol, trifluoroethanol, potassium pyrophosphate, acetic acid, DTT, DSS and deuterated sodium acetate." **The strict list was cut to 7 agents in `8eb7ad3` (2026-03-25, "remove non-denaturants from blacklist").** The manuscript documents a superseded code state and names DTT/DSS/acetic acid/mercaptoethanol as denaturants |
| **A4** | Table 1 | `exclude-paramagnetic` (F/T/T/T) is a real filter and is **absent** from the table |
| **A5** | `:119, 301` | "exp-method (solid-state): exclude" and "the experiment method, which excludes solid-state NMR". **Materially false** — the `""` sentinel makes the tolerant/moderate whitelist match everything, and 25289 ×2 + 27211 are in the released **strict** tier |

### B. Changed by the fixes

| # | loc | current | action |
|---|---|---|---|
| B1 | `:117` Table 1 keywords | `denatur, unfold, misfold, interacti, bound` | drop `interacti`; add the physical-state deny row |
| B2 | `:118` + fn a/b | 4 / 4 / 15 agents | recount: TFE/DMSO in, TFA/KPP out |
| B3 | `:301` Methods | "keyword blacklist (denature, unfold, misfold, **interact**, bound)" | same as B1 |
| B4 | `:301` | "chemical denaturants such as urea and guanidinium chloride" | add TFE/DMSO; state membrane mimetics are **flagged, not filtered** |
| B5 | `:197` | "strict set holds **1,685** sequences in **1,381** clusters, moderate **5,980** in **4,491**" | all four numbers move |
| B6 | `:252` | "only **240** of **5,900** representatives fail the tolerant criteria, and **4,491** pass even the moderate ones" | recompute — this is the load-bearing claim of the "nested sets hide filtering's effect" subsection |
| B7 | `:65, 155, 266, 307` | "**365**-sequence test set" (×4) | preserved if the pin repair lands (footnote only); otherwise update all four |
| B8 | `:65, 187, 268` | "more than **15,000** protein NMR experiments" | still true (16,851) — verify, don't assume |
| B9 | `:323` Data availability | "current version **v0.3.0**" | → v0.4.0 |
| B10 | `:326` Code availability | commit `6b17b3b` | → the release commit |
| B11 | `:309` | "CheZOD117 (**115** of 117 mapped)" | re-verify after rebuild |

### C. Figures — all require regeneration

| fig | file | what is baked in |
|---|---|---|
| **1a** | `Figures/2026-06-22_TriZOD_redundancy_reduction.tex` | **13 hard-coded TikZ numbers** — BMRB 17,388; unique seqs 9,480/9,040/7,009/2,232; after test-removal 8,237/7,824/5,980/1,685; training reps 5,900/5,660/4,491/1,381; "1,298 CheZOD-free"; "324 clusters (431 seq)"; "TriZOD test 365". The baseline rebuild already gives 9,480/**9,044**/**7,266**/**2,437**. **There is no generator** — every node must be re-typed by hand |
| **1b** | `disorder_content_PDF_vertical` | per-tier G-score distributions → regenerate |
| **2a** | `Z-score_G-score_synthetic` | **synthetic → unaffected** |
| **2b** | `G-scores_vs_Z-scores` | empirical → regenerate |
| **3a** | `25640_Cytochrome_C` | "**34** aligned entries" (`:205, 230`). ⚠️ Note bmr25640 is one of the three entries the C4 water fix **un-bounds** — this figure's subject changes status |
| **3b** | `mean_acc_vs_DisProt_disorder_frac` | DisProt **0.76** (`:205, 235, 248, 319`) |
| **3c** | `ROC_curves` | AUC **0.777–0.793** (`:207, 237, 256, 319`) and abstract "0.79" (`:65`) |

The memory note `manuscript-finalize-figure-regen` already flagged DisProt 0.76, ROC 0.777–0.793, CytC 34 and Figs 1b/2b/3 as needing regeneration post-#17, deferred for lack of local DisProt data and a Fig 1/3 generator. **That blocker is unchanged and is now on the critical path** — C1/C2/C5 move tier membership, so it cannot be deferred again. Fig 1a additionally has no generator at all.

### D. Docs to rewrite in the same pass

`docs/dataset/datasheet.md` (tier table §1; test set 344 →; §3 step 5 bound-complex wording; §4 leakage counts 1,190/1,168/921/492 and 148/143/112/31; §6 "max-offset … 3/3/2 **ppm**" → σ units) · `docs/dataset/dataset-construction.md` (all per-tier tables) · `docs/dataset/bundle-README.md` (the shipped datasheet) · `docs/filtering.md:25` · `trizod/dataset/deploy_fasta.py:25` (docstring says 5,684 seqs).

---

## 7. Open decisions for Tobias

### 7.1 `interacti` in the strict keyword blacklist — keep, drop, or swap?

**Recommended default: DROP.** Do not swap for `in complex with`/`complexed with` in v0.4.0.

Evidence for dropping: 46 of C1a's 50 strict drops are `interacti`; 54 % of them have no independent complex signal; 6/365 test chains die, all collateral; 14 of C5's 434 recovered chains die (barnase, cardiac troponin C, an A629P disease mutant ×4); `detect_bound()` already removes 85.8 % of what complex keywords catch, at build time.

Cost of dropping: re-admits 379–535 rows to strict, of which ~30 % **are** genuine complexes. They will be in `scores.json`, which Sandro reads directly, even though `detect_bound()` drops them at build.

**Shipping the issue-#23 bugfix while keeping `interacti` is measurably the worst option and should not be on the table.**

⚠️ **This decision must be made before PR5 merges,** and PR6 must price the resulting configuration end-to-end. Bracket: strict lands between 3,878 (keep) and ~4,300 (drop) `[est]`.

### 7.2 The 365-chain test set — preserve or shrink?

**Recommended default: preserve, by re-pinning against the tolerant pool + a `label_tier` column.** Two lines in `testset.py:280`. All 365 sequences are present in tolerant. Fallback: accept 349 and publish the 16 dropped IDs. Under no circumstances `--redraw`.

### 7.3 State / membrane / oligomeric signals — filter or annotate?

**Recommended default: annotate everywhere; filter only on `_Entity_assembly.Physical_state`.**

The physical_state exact-match deny list is the one exception because it is the only mechanism that catches 5158/5119/16948, costs 54/40/7 rows, removes 43 of 5,900 train representatives, has zero measured IDP collateral, and is the only route to 26816/27324/6227 (urea-denatured with no urea sample component). Both adversarial verifiers endorsed it in exact-match-only form.

Membrane mimetics, `molten globule`/`amyloid fibril` as free text, and the field-scoping question all go to PR8.

### 7.4 Does `bound` stay in the strict keyword list?

**Recommended default: keep for v0.4.0, audit in PR8.** It has the same failure mode as `apo`→`apoptosis`: 19 moderate rows match only via `unbound`/`boundaries` — *"Unbound Med25ACID"*, *"Human Pdx1 Homeodomain in the Unbound State"* — which are precisely the free-state deposits strict wants to **keep**. But it was never audited at strict scope, and removing an unaudited keyword in the same release as removing `interacti` compounds the change.

### 7.5 Immediate handoff to Sandro — non-DOI build or a real v0.3.1?

**Recommended default: non-DOI build off `develop`** after PR3/PR4/PR6-columns, delivered as a tarball or branch. It gives him the new metadata columns and the side-chain companion within ~2 weeks, decoupled from the policy debate, without minting a Zenodo record that will be superseded a month later.

Caveat to state to him: it is built on the post-#20 scores, so it already differs from v0.3.0 (strict 3,271 → 3,514) **before** any of these fixes. It cannot honestly be presented as "same data, new columns".

### 7.6 Version number

**Recommended default: v0.4.0 now, v1.0.0 at acceptance.** v0.3.0 stays up permanently under the same concept DOI.

### 7.7 DMSO concentration gate

**Recommended default: no gate.** ~40 lines of unit-normalisation heuristic over depositor free text, buying 6 spared rows at strict and 12 at moderate, with an undefined boundary at the only value that matters (bmr36172 = exactly 5.0 %). Ship ungated; record `dmso_pct` in metadata.

### 7.8 Solid-state veto scope

**Recommended default: ship the refined veto** (solid evidence **and** no solution-type experiment names) — 100 % measured precision, removes 25289 ×2 and 27211 from strict. **Do not** ship the naive universal veto: 37.5 % precision, kills 5 real solution structures.

### 7.9 Credit for Sandro Kuppel

**Facts, not a recommendation** — this call belongs to Tobias, Reid and Burkhard.

Reid and Iva are already co-authors (Reid: co-senior + corresponding, affil 7/8; Iva: affil 4), so their input on this review needs no separate action. Sandro is neither an author nor in the Acknowledgements — `Article.tex:328-330` is a placeholder: *"[Funding sources, collaborator thanks, and tool/resource credits to be added before submission.]"*

- He reported six distinct claims; all six were at least partially confirmed by independent audit.
- Two are genuine code defects with reproducible root causes (`fields.extend(el)`; the dead `startswith("metal")` test); a third (`has_nucleic` hybrid) is also real.
- His `_Entity.Type` histogram reproduced **exactly** against 21,842 records; his `_Entity_assembly.Physical_state` histogram reproduced exactly against 26,885; his homo-oligomer counts reproduced within 2.5 %.
- Several magnitude estimates were wrong (~300 vs 841; ~800 vs 438) and several proposed terms are net-negative on inspection.
- He did not contribute to conception, design, analysis or drafting; his use case is downstream and separate.
- If he contributes implementation, or if the C6 side-chain coverage analysis becomes a manuscript section or supplementary table, the profile changes materially.

Under ICMJE, bug reports and data-quality feedback conventionally map to **Acknowledgements**. On the facts as they stand that threshold is not met for authorship, but it is close enough that the call should be made **before** he is asked to review the v0.4.0 bundle, not after.

### 7.10 Process items surfaced in passing

There is no `CONTRIBUTING.md` and no CI, so `CLAUDE.md`'s three-command gate (`ruff check`, `ruff format --check`, `pytest`) is unenforced. PR #10 has been open against `develop` since March. If external PRs are going to be invited, both should be addressed first. **Recommended default: add a minimal GitHub Actions workflow running the three commands, plus a short `CONTRIBUTING.md`, in PR0's week.**

---

## 8. Draft reply to Sandro

> *Ready to send after Tobias edits. Assumes decisions 7.1 (drop `interacti`), 7.2 (preserve 365), 7.3 (annotate + physical_state deny), 7.5 (non-DOI handoff build). Adjust if any of those go the other way. Deliberately does not touch the credit question — that should come from Tobias separately.*

---

Hi Sandro,

Thanks — this was a genuinely useful review, and it landed on things we would not have found ourselves. We went through all six points properly, measured each one against the full 16,963-entry corpus, and then had a second pass adversarially check the numbers. Summary below, including the places where our measurements came out differently from your estimates.

**Where you were exactly right**

Your `_Entity.Type` histogram reproduces to the entry: `polymer 18897 / non-polymer 2926 / water 16 / null 1 / D-SACCHARIDE 1 / SACCHARIDE 1`, and **zero** entities are typed `metal`. `has_metal` has never fired. Same for the `_Entity_assembly.Physical_state` histogram (26,885 records — `native` 19,598, `denatured` 156, `unfolded` 101, `molten globule` 38, `intrinsically disordered` 196). Your homo-oligomer counts are within 2.5 % of ours (157 strict entries / 178 rows on the current scores, 751 unfiltered).

The `fields.extend(el)` bug is real and verbatim-reproducible: `citation_keywords = ['Protein misfolding']` becomes `['P','r','o','t','e','i','n', …]`, so those two fields have never contributed to the keyword blacklist. Same for the DNA/RNA hybrid polymer type, which the two-value exact match misses (16 entity records in 14 entries).

**The concrete example that came out of this**

BMRB 5158 — apo-myoglobin, molten globule — is `split=train, train_tier=moderate` in the published v0.3.0 deposit. Its title *and* its citation title say "Molten Globule state", `_Entity_assembly.Physical_state` is literally `molten globule`, and TriZOD scores 52 % of its residues as ordered. It misses strict only on temperature, pH and a missing method-subtype tag — no state filter was ever involved. Siblings: 5119 (ATP synthase subunit c in chloroform/methanol, 100 % ordered) and 16948 (dynamin GED in DMSO, 96 % ordered), both also in train.

Here is the twist that changed our plan: **5158's `citation_keywords` and `struct_keywords` are both empty lists.** Issue #23 — the bug you filed — does nothing for it. What catches 5158 is the `Physical_state` tag, which is the half of your finding that had no issue number. So your instinct was right, but the mechanism you filed for is not the one that catches the case that matters. We are shipping both, with the state tag as an exact-match deny list rather than folded into the substring search.

**Where our numbers differ from yours**

- **Denaturants, "~300 new files":** your seven tokens remove 841 / 755 / 439 / **76** rows at unfiltered / tolerant / moderate / strict. That is 2.8× your estimate at tolerant but **4× lower at strict** — most of those entries are already gone for other reasons. DPC alone is 299 entries, which is probably where the ~300 came from.
- **TFA:** it is already in the strict list, and it is a false positive — 56 of 57 percent-unit components are ≤ 0.2 % (median 0.1 %), i.e. HPLC counterion. We are **removing** it, not keeping it. Same for `Potassium Pyrophosphate` (a buffer, added in an unreviewed 2023 grab-bag). Combined effect on the strict tier: **+1 row**.
- **SDS / DPC / dodecyl:** these are membrane mimetics, not denaturants — 61–72 % of the impact of your list is micelle chemistry. We are giving them their own `membrane_mimetic` column rather than folding them into `chemical-denaturants`, so you can drop micelle-bound chains yourself with one line of pandas and membrane-NMR users are not forced into our choice.
- **Method whitelist, "~800 recoverable":** no spelling is missing. Every real solution-NMR spelling already passes. The whole loss is 4,103 entries whose subtype **tag is absent** — strict had no sentinel for that and dropped them wholesale. Recovering them (on positive `_Sample.Type` evidence) is **+438 rows**, about half your estimate. Interestingly, +793 rows is almost exactly what you get from relaxing strict's ionic-strength *unit* requirement, which is a different filter entirely — that may be what your ~800 was actually measuring.
- **"Molten" as a free-text keyword:** ~30 more at unfiltered/tolerant, as you said. But only 14 at moderate and **2** at strict — and both strict hits are false positives (BMRB 19560 is folded Hsp90; the phrase is in a citation title about p53). The `Physical_state` route reaches 21 of the 31 with none of that risk, which is why we went that way.
- **Re-referencing side chains:** we briefly thought we had a result showing the backbone ¹³C offset transfers to side-chain carbons, which would have contradicted your premise. It does not survive scrutiny — the correlation was carried by a single chain with a 300 ppm LACS blowup, and the arithmetic mixed two different unit conventions. **You were right.** We are shipping side-chain values raw, with `lacs_off_*` already in the table so anyone who needs a correction can apply it themselves.

**On side chains generally** — you understated this one. The release does not just discard side chains; it ships **no chemical shifts at all**. 42 columns of scores, offsets and metadata; raw shifts only in the optional 1.4 GB `.str` bundle. We are adding a companion Parquet with the ~3.5 M discarded side-chain values (`id`, `seq_id`, `comp_id`, `atom_id`, `val`, `val_err`, `ambiguity_code`), ~12 MB, joining 1:1 on `id`.

One warning if you build an outlier filter on those: a naive "physically impossible ppm" rule is really a paramagnetic-protein detector — 91 % of its ¹H hits are hyperfine-shifted resonances in cytochromes, Cu-Zn SODs and peroxidases, i.e. correct measurements. About 30 % of what a k=10 robust-z rule flags is real physics (paramagnetics plus ARG NE aliasing, where 15 % of ARG NE values are folded by one ¹⁵N spectral width). We will ship the raw values plus the ambiguity codes and let you decide.

**What we are doing, and when**

*Merging now (no change to which chains are in which tier):*
- issue #23 (`fields.extend`) and the `""`-sentinel regex bug in the method whitelist
- `has_metal` rebuilt on `_Entity.Nonpolymer_comp_ID` + `_Chem_comp.Formula`; DNA/RNA hybrid caught; a water-entity bug that was wrongly marking three protein-only depositions as complexes
- new columns at every tier: `physical_state`, `membrane_mimetic`, `sample_state_evidence`, ligand comp-IDs, and per-row oligomer counts
- the side-chain companion table

*Then, as a policy change:* `interacti` comes out of the strict keyword list (46 of its 50 hits were paper-topic labels like "protein–protein interaction" on free monomers), TFE and DMSO go in as denaturants, TFA and potassium pyrophosphate come out, the `Physical_state` deny list goes in at tolerant and above, and strict admits the undeclared-method entries. Net: strict grows ~10 %, tolerant and moderate shrink ~1 %.

*Deliberately deferred to a separate RFC:* filtering on membrane mimetics by default, and `molten globule`/`amyloid fibril` as free-text keywords. Both remove more good data than bad in our measurements, and both would cost published test-set chains. You get the columns either way.

**We can unblock you before the policy debate finishes.** The metadata columns and the side-chain table change no membership at all, so we can hand you a build off `develop` in ~2 weeks with all of it, ahead of the Zenodo v0.4.0 respin. One caveat: that build sits on a rescoring run we did in July (issue #20), so the strict tier is already 3,514 rows rather than v0.3.0's 3,271 — it is not "v0.3.0 plus columns". Tell us if you would rather wait for the tagged release.

**Two questions for you**

1. What exactly do you want from the oligomer flag? "As deposited" is what BMRB gives us, and it is noisy in both directions — it under-counts (~10–28 homodimers declare a single assembly row; one entry is literally *named* "homodimer" with one row) and it over-counts, because a fair number of depositors use repeated assembly rows for cis/trans or major/minor **conformers of a monomer**. BMRB 15711 is *"monomeric apoSOD1"* with rows named "proline cis conformer"/"proline trans conformer" — a naive rule calls it a dimer, and it is in the published CheZOD117 test split. We are building a more conservative counter using the magnetic-equivalence tag, but if you need a curated experimental oligomeric state that is a much bigger curation project and we would want to scope it separately.
2. On the ColabFold validation idea — worth doing, with one caveat and one practical snag. The caveat: raw NMR-vs-structure disagreement is not by itself evidence of bad data, since a real IDP disagrees too. It only reads as filter validation *conditioned on* TriZOD calling the chain ordered, which is a narrower and more defensible claim. The snag: AlphaFold2 parameters are CC BY 4.0, so predicted structures cannot ship under the dataset's MIT licence — we would need to distribute them separately or ship only derived metrics. Happy to talk through how to set it up.

Thanks again — several of these had been sitting in the code for years and would not have surfaced from inside the project.

Best,
Tobias

---

*[Note for Tobias, not for sending: §7.9 sets out the facts bearing on whether Sandro belongs in the Acknowledgements or higher. Worth settling with Reid and Burkhard before he is asked to review the v0.4.0 bundle.]*

---

## 9. Outcome (appended 2026-08-11)

*The sections above are the historical record of what was proposed and decided on
2026-08-09 and are left unedited. This section records what actually shipped on branch
`fix/kuppel-review` and where it diverged from the plan. Numbers are read from the
regenerated artefacts under `data/interim/scored/` and `data/interim/build/`.*

### 9.1 Decisions as resolved

| § | question | resolution |
|---|---|---|
| 7.1 | `interacti` — keep, drop, or swap | **Dropped *and* swapped.** The plan recommended dropping without a replacement; the shipped strict list drops `interacti` and adds `in complex with` / `complexed with`, matched on sample-descriptive fields only |
| 7.2 | preserve the 365-chain test set | **Preserved by re-pinning against the tolerant pool**, plus a `label_tier` column. 364 of 365 resolve (see 9.3) |
| 7.3 | state / membrane / oligomeric signals | **Annotate everywhere; filter only on `physical_state`** — as recommended. The keyword **field-scoping** boolean, which §3.1 had deferred to an RFC, shipped as well (`--keyword-search-scope`, default `sample`) |
| 7.4 | does `bound` stay at strict | **Kept, and made whole-word.** The `unbound`/`boundaries` failure mode the section flagged is fixed rather than deferred |
| 7.7 | DMSO concentration gate | **No gate** — as recommended |
| 7.8 | solid-state veto scope | **Refined veto** (solid evidence *and* no solution-type experiment name) — as recommended |
| 7.6 / 5.4 | version number | **Not settled here.** The build is named `2026-08`; no Zenodo record has been minted and no version number is claimed in the docs |

### 9.2 Divergence from the plan: ambiguous physical states are corroborated

The plan's tier policy (§3.1) denies `denatured` / `partially denatured` at tolerant and
`unfolded` / `partially unfolded` at moderate **on the deposited tag alone**, and §3.1's
own evidence block noted the collateral without resolving it.

That is not what shipped. Those four values are now
`PHYSICAL_STATE_AMBIGUOUS` and are denied **only where the entry independently names a
denaturant** (`has_denaturant_evidence`, exposed as the `denaturant_evidence` column).
Everything else in the deny lists still stands on the tag alone, which is what preserves
the mechanism's original purpose: 5158, 5119 and 16948 are still removed.

Re-measured on the regenerated pre-filter frame (17,843 rows), as `filtered (unique)`:

| rule | tolerant | moderate | strict |
|---|---|---|---|
| deny on the tag alone (as §3.1 proposed) | 214 (65) | 323 (46) | 331 (14) |
| **deny only where corroborated** (shipped) | 180 (39) | 221 (12) | 229 (4) |
| never deny an ambiguous value | 41 (20) | 45 (11) | 53 (4) |

The `unique` column of the tag-alone row — **65 / 46 / 14** — is where §1.3's and §3.1's
"−65 / −46 / −14" came from; those figures were correct for the rule as planned. The
shipped rule costs **39 / 12 / 4**, and the rows it still uniquely removes are dominated
by `molten globule` (18 tolerant, 10 moderate), i.e. exactly the 5158 class.

Measured on the released tiers, the corroboration rule retains **24 / 28 / 5** records at
tolerant / moderate / strict that a tag-alone rule would have deleted. They include
α-synuclein (6968 — whose entity is literally named *"intrinsically disordered
alpha-synuclein"* — plus 16342, 25227, 25228), Tau (52309, 52401), γ-synuclein (7244),
endosulfine α (15136), ACTR/CBP (15397/15398), NS5A D2 (15225), the T-cell-receptor
ζ chain (15409) and the yeast SNAREs Snc1/Sso1 (4286/4287). The
`test_alpha_synuclein_survives_the_tolerant_deny_list` /
`test_apo_myoglobin_molten_globule_is_still_denied` pair in
`tests/test_physical_state.py` pins both directions.

### 9.3 Final numbers

Scored records per tier, and the record-level churn against v0.3.0:

| tier | v0.3.0 | 2026-08 | removed | added | net | training reps |
|---|--:|--:|--:|--:|--:|--:|
| unfiltered | 16,851 | 16,851 | 0 | 0 | 0 | 5,907 |
| tolerant | 15,446 | 15,193 | 273 | 20 | −253 | 5,590 |
| moderate | 11,306 | 11,175 | 198 | 67 | −131 | 4,625 |
| strict | 3,514 | **4,113** | 285 | 884 | **+599** | 1,998 |

Strict landed at 4,113, inside §7.1's 3,878–4,300 bracket. `unfiltered` is the identical
record set ID for ID, as §7.3 predicted.

**Test set — one correction to §7.2.** The plan asserted that *"all 365 pinned sequences
are present in tolerant"*, so re-pinning would preserve 365/365. It preserves **364**:
`19342_1_1_1` ("Transmembrane-cytosolic part of Trop2") is measured in **70 %
trifluoroethanol** and is removed by the C2 TFE token, which the §7.2 measurement
predated. This is a correction rather than collateral — at that concentration the shifts
report a solvent-forced helical conformation, not the aqueous state. Additionally
`50998_1_1_1` was ID-substituted to `5599_1_1_1` (identical sequence, lower entry
number), and **17** retained chains no longer meet strict criteria (`label_tier`: 10
moderate, 7 tolerant). Leakage gate: 0 shared IDs, 0 exact-sequence matches.

**Release Parquet**: 16,851 rows × **63** columns (was 42). `split` = train 5,907 /
excluded 8,131 / redundant 2,334 / test_trizod 364 / test_chezod117 115; `train_tier` =
strict 1,998 / moderate 2,627 / tolerant 965 / unfiltered 317. **Side-chain companion**:
3,458,851 shifts over 12,643 of the 16,851 chains (C6 §3.6 estimated 3,458,852 — one
row's difference).

### 9.4 Superseded numbers in this document

The `[harness]` and `[est]` projections in §1.3 (Set A / Set B, ~15,210 / ~11,153 /
3,878–4,300) were pre-run estimates and are superseded by the table in 9.3. The
physical-state deny-list figures in §1.3, §2 (C1b), §3.1 and §7.3 — 54 / 40 / 7
`[harness]`, and the "−65 / −46 / −14" that reached `docs/filtering.md` — all describe
the **tag-alone** rule that was not shipped; on the regenerated run that rule costs
65 / 46 / 14 and the shipped rule costs 39 / 12 / 4 (9.2). The current per-tier
filter-loss report for every criterion is in `docs/filtering.md`. §6D's documentation
checklist is done for `docs/filtering.md`, `docs/dataset/datasheet.md`,
`docs/dataset/dataset-construction.md` and `docs/dataset/bundle-README.md`; the
manuscript items in §6A–C and `trizod/dataset/deploy_fasta.py:25` are **not**.