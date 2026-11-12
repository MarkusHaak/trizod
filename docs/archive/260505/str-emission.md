# Re-referenced NMR-STAR (`.str`) emission

## What changed

The TriZOD CLI now optionally emits one re-referenced `.str` file per scored entry. Driven by the new `--emit-str <dir>` flag. Off by default.

```bash
uv run trizod \
  --input-dir data/bmrb_entries/ \
  --output-prefix data/release/strict/scores \
  --output-format json \
  --filter-defaults strict \
  --emit-str data/release/strict/str/ \
  --rereference-mode both
```

## File layout per emitted entry

Filename: `bmr<entryID>_<stID>_<entity_assemID>_<entityID>_rereferenced.str` — uniqueness keyed by the same identity tuple TriZOD uses internally, so multi-shift-table entries don't collide.

Each file contains **two saveframes**:

### 1. `assigned_chem_shift_list_1` — backbone shifts

Standard NMR-STAR `Atom_chem_shift` loop with one row per (residue, backbone-atom) pair where the parser found a shift. Tags:

```
ID  Seq_ID  Comp_ID  Atom_ID  Atom_type  Val  Val_err  Ambiguity_code
```

Values are **post-LACS** (the LACS offset has been subtracted). `Val_err` is `.` (NMR-STAR sentinel for "not specified"); `Ambiguity_code` is also `.` (the writer doesn't propagate the original ambiguity codes — that information is only meaningful for full-table re-deposit, which isn't the goal).

### 2. `trizod_rereferencing_info` — re-referencing metadata

Custom saveframe (category `trizod_rereferencing`) with the per-entry record of what was done:

| Tag | Value |
|---|---|
| `Source_BMRB_id` | the original BMRB entry id |
| `Pipeline_version` | `trizod-2026-05-05` |
| `Re_referencing_mode` | `none` / `lacs` / `potenci-only` / `both` |

Plus two loops, one row per backbone atom:

- `LACS_offsets` — `Atom_ID, Offset_ppm`
- `POTENCI_residual_offsets` — `Atom_ID, Offset_ppm`

## Why two saveframes

The first is what most consumers want — clean re-referenced backbone shifts ready to drop into downstream pipelines. The second makes the file self-describing: anyone re-using these shifts can see which corrections were applied and how big they were, without having to consult the original TriZOD JSON output.

## What's NOT emitted

- Side-chain shifts beyond what `get_valid_bbshifts` extracts (HA2/HA3/HB1/HB2/HB3 collapsed to HA/HB by averaging).
- The Step 8 methyl wildcards (`CDx`/`CGx`) — those live in the raw shift table, but `get_valid_bbshifts` doesn't surface side-chain methyls into the emitted file.
- Sample / assembly / experimental-condition saveframes.

This is by design: the `.str` is a *re-referenced backbone-shifts subset*, not a full BMRB re-deposit.

## Code surface

| File | Change |
|---|---|
| `trizod/io/__init__.py` | New (empty). |
| `trizod/io/str_writer.py` | New. Single function `write_rereferenced_str(...)` writing the two saveframes via pynmrstar. |
| `trizod/trizod.py` | New `--emit-str <dir>` flag + per-row writer block in `main()`, lazy-imported. Passes `averaging=not args.no_shift_averaging` to `get_valid_bbshifts` to keep the emitted file consistent with what was scored. |
| `tests/test_str_writer.py` | Round-trip + aux-content unit tests. |
| `tests/test_smoke.py` | End-to-end CLI smoke test on BMRB 6968. |

## Talking points for the slide

- **What:** users now get a self-contained re-referenced backbone-shifts file per entry, ready to consume.
- **Why:** the JSON output gives scores; the `.str` files give the *corrected shifts themselves* — useful for any downstream secondary-shift / disorder / structural-analysis tool.
- **Audit trail:** every file carries the LACS + POTENCI offsets and the pipeline version, so re-referencing decisions are reproducible.

## Commits

- `5863851` — `feat(output): emit re-referenced NMR-STAR (.str) files via --emit-str`
- `4c11eb9` — `fixup(emit-str): unique filename per shift table + honour --no-shift-averaging`
