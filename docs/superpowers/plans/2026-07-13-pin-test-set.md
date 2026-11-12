# Pin the TriZOD Test Set Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the TriZOD test set so `trizod dataset test-set` reproduces the same sequences on every run, changing only under an explicit `--redraw`.

**Architecture:** Commit the current (post-fix) 342 representative sequences as a pinned reference file. `test-set` loads the pin by default and resolves each sequence against the current strict-tier pool (no mmseqs); `--redraw` runs today's seeded recipe and overwrites the pin.

**Tech Stack:** Python ≥3.9, argparse (module) + Typer (CLI), mmseqs2 (only on `--redraw`), pytest, ruff, uv.

## Global Constraints

- Python ≥3.9; `from __future__ import annotations` is already in `testset.py`.
- Lint/format: `uv run ruff check trizod/ tests/` and `uv run ruff format --check trizod/ tests/` must pass.
- Tests: `uv run pytest tests/ -v` must pass.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Branch: `feat/pin-test-set` (already created, spec committed there).
- Show staged files + proposed message before each commit.

## File Structure

- `trizod/dataset/pinned/TriZOD_test_set.fasta` — **new, committed**: the 342 pinned representative sequences (`>entry_id` headers).
- `trizod/dataset/pinned/TriZOD_test_set.provenance.json` — **new, committed**: provenance sidecar.
- `trizod/dataset/paths.py` — **modify**: add `pinned_testset` to `resolve_paths()`.
- `trizod/dataset/testset.py` — **modify**: add `resolve_pinned_testset()` + `_entry_sort_key()` + `_write_pin()`; extract the seeded recipe into `_redraw()`; make `main()` default to pinned load, `--redraw` re-pin.
- `trizod/cli/main.py` — **modify**: add `--redraw` option to the `test-set` command.
- `tests/test_testset_pin.py` — **new**: unit tests for the resolver + `main()` pinned/missing-pin paths (mmseqs-free).

---

### Task 1: `pinned_testset` path + committed pin file (bootstrap)

**Files:**
- Modify: `trizod/dataset/paths.py` (inside `resolve_paths`, the returned `SimpleNamespace`)
- Create: `trizod/dataset/pinned/TriZOD_test_set.fasta` (copied from the current build)
- Create: `trizod/dataset/pinned/TriZOD_test_set.provenance.json`
- Test: `tests/test_testset_pin.py`

**Interfaces:**
- Produces: `resolve_paths(...).pinned_testset -> Path` pointing at `<root>/trizod/dataset/pinned/TriZOD_test_set.fasta`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_testset_pin.py`:

```python
from trizod.dataset.paths import resolve_paths
from trizod.io.fasta import count_fasta


def test_pinned_testset_path_and_file_present():
    paths = resolve_paths()
    assert paths.pinned_testset.name == "TriZOD_test_set.fasta"
    assert paths.pinned_testset.parent.name == "pinned"
    assert paths.pinned_testset.exists(), "committed pinned test set is missing"
    assert count_fasta(paths.pinned_testset) == 342
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_testset_pin.py::test_pinned_testset_path_and_file_present -v`
Expected: FAIL — `AttributeError: ... 'pinned_testset'` (attribute not yet added).

- [ ] **Step 3: Add the path resolver**

In `trizod/dataset/paths.py`, inside the `SimpleNamespace(...)` returned by `resolve_paths`, add (next to the other repo-root inputs like `chezod117`):

```python
        # committed pinned test set (repo-root input, not a work-dir artifact)
        pinned_testset=layout.root
        / "trizod"
        / "dataset"
        / "pinned"
        / "TriZOD_test_set.fasta",
```

- [ ] **Step 4: Create the committed pin file + provenance from the current build**

Run (bootstraps the pin from the validated post-fix build):

```bash
mkdir -p trizod/dataset/pinned
cp data/interim/build/testset/TriZOD_test_set.fasta trizod/dataset/pinned/TriZOD_test_set.fasta
uv run python - <<'PY'
import json, datetime
from trizod.io.fasta import count_fasta
from pathlib import Path
p = Path("trizod/dataset/pinned/TriZOD_test_set.fasta")
prov = {
    "adopted_from": "2026-07-13 post-fix rebuild (F->K #11, details #12, robustfit)",
    "source_build_version": "2026-06",
    "seed": 42,
    "sample_fraction": 0.25,
    "count": count_fasta(p),
    "written": datetime.date.today().isoformat(),
}
Path("trizod/dataset/pinned/TriZOD_test_set.provenance.json").write_text(
    json.dumps(prov, indent=2) + "\n"
)
print("count:", prov["count"])
PY
```

Expected: `count: 342`.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_testset_pin.py::test_pinned_testset_path_and_file_present -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add trizod/dataset/paths.py trizod/dataset/pinned/ tests/test_testset_pin.py
git status --short   # show staged files
git commit -m "feat(dataset): add committed pinned TriZOD test set + path resolver

Bootstraps the pin from the post-fix 2026-07-13 rebuild (342 sequences) and
exposes it via resolve_paths().pinned_testset.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `resolve_pinned_testset` resolver (pure function)

**Files:**
- Modify: `trizod/dataset/testset.py` (add helpers near the top, after the constants)
- Test: `tests/test_testset_pin.py`

**Interfaces:**
- Produces:
  - `_entry_sort_key(entry_id: str) -> tuple` — deterministic ordering key.
  - `resolve_pinned_testset(pinned: dict[str, str], strict: dict[str, str]) -> tuple[dict[str, str], dict]` — returns `(test_recs, info)` where `test_recs` is `{entry_id: sequence}` and `info` has keys `pinned_total`, `resolved`, `dropped` (list of pinned IDs), `substitutions` (list of `[pinned_id, used_id]`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_testset_pin.py`:

```python
from trizod.dataset.testset import resolve_pinned_testset


def test_pinned_exact_match():
    pinned = {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC"}
    strict = {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC", "300_1_1_1": "GGGG"}
    recs, info = resolve_pinned_testset(pinned, strict)
    assert recs == {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC"}
    assert info["dropped"] == []
    assert info["substitutions"] == []
    assert info["resolved"] == 2 and info["pinned_total"] == 2


def test_pinned_entry_id_fallback_lowest_numbered():
    # The pinned representative entry 999 is gone; the sequence now appears
    # under entries 100 and 50 -> use the lowest-numbered (50).
    pinned = {"999_1_1_1": "AAAA"}
    strict = {"100_1_1_1": "AAAA", "50_1_1_1": "AAAA"}
    recs, info = resolve_pinned_testset(pinned, strict)
    assert recs == {"50_1_1_1": "AAAA"}
    assert info["substitutions"] == [["999_1_1_1", "50_1_1_1"]]


def test_pinned_dropped_when_sequence_absent():
    pinned = {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC"}
    strict = {"100_1_1_1": "AAAA"}  # CCCC no longer in the strict pool
    recs, info = resolve_pinned_testset(pinned, strict)
    assert recs == {"100_1_1_1": "AAAA"}
    assert info["dropped"] == ["200_1_1_1"]


def test_pinned_stable_under_representative_reshuffle():
    # Same sequences, but the strict pool gained an extra entry for AAAA.
    # The emitted test set must stay identical to the pinned IDs.
    pinned = {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC"}
    strict_v1 = {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC"}
    strict_v2 = {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC", "900_1_1_1": "AAAA"}
    r1, _ = resolve_pinned_testset(pinned, strict_v1)
    r2, _ = resolve_pinned_testset(pinned, strict_v2)
    assert r1 == r2 == {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_testset_pin.py -k "pinned_exact or fallback or dropped or reshuffle" -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_pinned_testset'`.

- [ ] **Step 3: Implement the resolver**

In `trizod/dataset/testset.py`, after the module constants (`SEED`, `SAMPLE_FRACTION`, `CHEZOD_PREFIX`) and before `chezod1325_records`, add:

```python
def _entry_sort_key(entry_id: str):
    """Deterministic key so the lowest-numbered entry ID sharing a sequence is
    chosen. Splits on '_'; numeric parts sort before non-numeric parts."""
    key = []
    for part in entry_id.split("_"):
        key.append((0, int(part), "") if part.isdigit() else (1, 0, part))
    return tuple(key)


def resolve_pinned_testset(
    pinned: dict[str, str], strict: dict[str, str]
) -> tuple[dict[str, str], dict]:
    """Resolve pinned test sequences against the current strict-tier pool.

    Each pinned sequence maps to its pinned entry ID if that entry is still in
    the strict pool, else to the lowest-numbered current entry sharing the
    identical sequence. A pinned sequence absent from the pool is dropped.
    """
    seq_to_ids: dict[str, list[str]] = {}
    for eid, seq in strict.items():
        seq_to_ids.setdefault(seq, []).append(eid)
    for seq in seq_to_ids:
        seq_to_ids[seq].sort(key=_entry_sort_key)

    test_recs: dict[str, str] = {}
    dropped: list[str] = []
    substitutions: list[list[str]] = []
    for pid, pseq in pinned.items():
        ids = seq_to_ids.get(pseq)
        if not ids:
            dropped.append(pid)
            continue
        chosen = pid if pid in ids else ids[0]
        if chosen != pid:
            substitutions.append([pid, chosen])
        test_recs[chosen] = pseq

    info = {
        "pinned_total": len(pinned),
        "resolved": len(test_recs),
        "dropped": dropped,
        "substitutions": substitutions,
    }
    return test_recs, info
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_testset_pin.py -v`
Expected: PASS (all resolver tests + the Task 1 path test).

- [ ] **Step 5: Commit**

```bash
git add trizod/dataset/testset.py tests/test_testset_pin.py
git status --short
git commit -m "feat(dataset): add resolve_pinned_testset resolver

Resolves pinned test sequences against the current strict pool: prefer the
pinned entry ID, else the lowest-numbered entry with the same sequence, drop
sequences no longer present.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `test-set` default = pinned, `--redraw` re-pins; CLI flag

**Files:**
- Modify: `trizod/dataset/testset.py` (module docstring, extract `_redraw`, add `_write_pin`, rewrite `main`)
- Modify: `trizod/cli/main.py:374-382` (add `--redraw` option)
- Test: `tests/test_testset_pin.py`

**Interfaces:**
- Consumes: `resolve_pinned_testset` (Task 2), `resolve_paths(...).pinned_testset` (Task 1), `read_fasta`/`write_fasta` from `trizod.io.fasta`.
- Produces: `testset.main(argv)` — default emits the pinned test set; `--redraw` runs the seeded recipe and overwrites the pin. `_write_pin(pin_path: Path, test_recs: dict[str, str], extra: dict | None) -> None`. `_redraw(paths, out: Path) -> dict[str, str]` returns the redrawn `test_recs`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_testset_pin.py`:

```python
import pytest

from trizod.dataset import testset
from trizod.io.fasta import read_fasta


def _setup_root_and_wd(tmp_path, pin_records, strict_records):
    root = tmp_path / "root"
    wd = tmp_path / "wd"
    pin_dir = root / "trizod" / "dataset" / "pinned"
    pin_dir.mkdir(parents=True)
    with (pin_dir / "TriZOD_test_set.fasta").open("w") as fh:
        for rid, seq in pin_records.items():
            fh.write(f">{rid}\n{seq}\n")
    strict_dir = wd / "final_dataset" / "strict"
    strict_dir.mkdir(parents=True)
    with (strict_dir / "strict.fasta").open("w") as fh:
        for rid, seq in strict_records.items():
            fh.write(f">{rid} tier=strict\n{seq}\n")
    return root, wd


def test_main_pinned_mode_emits_pinned_set(tmp_path):
    root, wd = _setup_root_and_wd(
        tmp_path,
        pin_records={"100_1_1_1": "AAAA", "200_1_1_1": "CCCC"},
        strict_records={"100_1_1_1": "AAAA", "200_1_1_1": "CCCC", "300_1_1_1": "GG"},
    )
    testset.main(["--work-dir", str(wd), "--root", str(root)])
    out = read_fasta(wd / "testset" / "TriZOD_test_set.fasta")
    assert out == {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC"}


def test_main_missing_pin_errors(tmp_path):
    root = tmp_path / "root"
    wd = tmp_path / "wd"
    (wd / "final_dataset" / "strict").mkdir(parents=True)
    (wd / "final_dataset" / "strict" / "strict.fasta").write_text(">1_1_1_1\nAA\n")
    with pytest.raises(SystemExit):
        testset.main(["--work-dir", str(wd), "--root", str(root)])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_testset_pin.py -k "pinned_mode_emits or missing_pin" -v`
Expected: FAIL — current `main()` ignores the pin and runs mmseqs (errors/None), so assertions fail or it raises the wrong error.

- [ ] **Step 3: Refactor `main` into `_redraw` + add `_write_pin`, and rewrite `main`**

In `trizod/dataset/testset.py`:

(a) Add `import datetime` to the imports.

(b) Rename the existing `main(argv=None)` body: change the function signature line `def main(argv=None) -> None:` to `def _redraw(paths, out: Path) -> dict[str, str]:`, remove its `ap`/`args`/`paths`/`out`/`tmp` setup lines (the first block through `tmp.mkdir(...)`), and instead start the function body with:

```python
def _redraw(paths, out: Path) -> dict[str, str]:
    """Seeded 30/80 -> sample -> 50/80 recipe. Writes the fasta/clu/summary and
    returns {entry_id: sequence} for the redrawn test set."""
    import shutil

    strict_fasta = paths.final_dataset / "strict" / "strict.fasta"
    tmp = out / "_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
```

Keep the rest of the original body unchanged **except** the final `print(...)` lines, and end the function with:

```python
    return test_recs
```

(c) Add the pin-writer helper:

```python
def _write_pin(pin_path, test_recs: dict[str, str], extra: dict | None = None) -> None:
    """Overwrite the committed pin FASTA + provenance sidecar."""
    pin_path.parent.mkdir(parents=True, exist_ok=True)
    write_fasta(test_recs, pin_path)
    prov = {
        "count": len(test_recs),
        "seed": SEED,
        "sample_fraction": SAMPLE_FRACTION,
        "written": datetime.date.today().isoformat(),
    }
    if extra:
        prov.update(extra)
    pin_path.with_suffix(".provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n"
    )
```

(d) Add the new `main`:

```python
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path, default=None,
                    help="dataset build dir (default: <root>/data/interim/build)")
    ap.add_argument("--root", type=Path, default=None,
                    help="repository root (default: auto-detected)")
    ap.add_argument("--redraw", action="store_true",
                    help="Redraw the seeded test set and OVERWRITE the committed "
                         "pin (default: emit the committed pin).")
    args = ap.parse_args(argv)
    paths = resolve_paths(args.work_dir, args.root)
    out = paths.testset
    out.mkdir(parents=True, exist_ok=True)

    if args.redraw:
        test_recs = _redraw(paths, out)
        _write_pin(paths.pinned_testset, test_recs, extra={"mode": "redraw"})
        print(f"Re-pinned {len(test_recs)} sequences -> {paths.pinned_testset}")
        return

    if not paths.pinned_testset.exists():
        raise SystemExit(
            f"Pinned test set not found at {paths.pinned_testset}. "
            f"Run `trizod dataset test-set --redraw` to establish it."
        )

    strict = read_fasta(paths.final_dataset / "strict" / "strict.fasta")
    pinned = read_fasta(paths.pinned_testset)
    test_recs, info = resolve_pinned_testset(pinned, strict)

    write_fasta(test_recs, out / "TriZOD_test_set.fasta")
    summary = {"mode": "pinned", **info}
    (out / "build_test_set_summary.json").write_text(json.dumps(summary, indent=2))

    if info["dropped"]:
        print(f"WARNING: {len(info['dropped'])} pinned sequences are no longer in "
              f"the strict pool and were dropped: {info['dropped']}")
    if info["substitutions"]:
        print(f"NOTE: {len(info['substitutions'])} pinned representatives were "
              f"substituted by the lowest-numbered entry with the same sequence.")
    print(f"TriZOD test set (pinned): {len(test_recs)} sequences -> "
          f"{out / 'TriZOD_test_set.fasta'}")
```

(e) Update the module docstring (lines 2–29): replace the "with a FIXED SEED" recipe description with a note that `test-set` emits the committed pin by default and `--redraw` re-runs the seeded recipe (kept below) and overwrites the pin.

- [ ] **Step 4: Add the `--redraw` CLI flag**

In `trizod/cli/main.py`, replace the `test-set` command (lines 374–382) with:

```python
@dataset_app.command("test-set")
def _dataset_testset(
    work_dir: Optional[str] = typer.Option(None, "--work-dir"),
    root: Optional[str] = typer.Option(None, "--root"),
    redraw: bool = typer.Option(
        False, "--redraw",
        help="Redraw the seeded test set and overwrite the committed pin.",
    ),
):
    """Emit the pinned TriZOD test set (-> testset/); --redraw re-establishes the pin."""
    from trizod.dataset import testset

    argv = _wd_argv(work_dir, root)
    if redraw:
        argv.append("--redraw")
    testset.main(argv)
```

- [ ] **Step 5: Run the new + full test suite**

Run: `uv run pytest tests/test_testset_pin.py -v`
Expected: PASS (pinned-mode emit + missing-pin error + Task 1/2 tests).

Run: `uv run pytest tests/ -q`
Expected: PASS (no regressions).

- [ ] **Step 6: Lint/format**

Run: `uv run ruff check trizod/ tests/ && uv run ruff format --check trizod/ tests/`
Expected: all pass (run `uv run ruff format trizod/ tests/` first if needed).

- [ ] **Step 7: Commit**

```bash
git add trizod/dataset/testset.py trizod/cli/main.py tests/test_testset_pin.py
git status --short
git commit -m "feat(dataset): test-set emits the pinned set by default, --redraw re-pins

Default test-set now loads the committed pin and resolves it against the current
strict pool (no mmseqs). --redraw runs the seeded recipe and overwrites the pin.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Integration verification against the real build

**Files:**
- Modify: `docs/dataset/dataset-construction.md` (if it documents the test-set draw) — note the pin behaviour.
- No new code.

**Interfaces:**
- Consumes: the committed pin (Task 1) + the current `data/interim/build/final_dataset/strict/strict.fasta`.

- [ ] **Step 1: Emit the pinned test set on the real build**

Run: `uv run trizod dataset test-set`
Expected: prints `TriZOD test set (pinned): 342 sequences -> ...`, no WARNING/NOTE (every pinned sequence is in the current strict pool).

- [ ] **Step 2: Verify it reproduces the pinned sequences exactly**

Run:

```bash
uv run python - <<'PY'
from trizod.dataset.paths import resolve_paths
from trizod.io.fasta import read_fasta
p = resolve_paths()
pin = read_fasta(p.pinned_testset)
out = read_fasta(p.testset / "TriZOD_test_set.fasta")
assert set(pin.values()) == set(out.values()), "emitted sequences != pinned"
print(f"OK: {len(out)} sequences, sequence set matches the pin")
PY
```

Expected: `OK: 342 sequences, sequence set matches the pin`.

- [ ] **Step 3: Verify stability — a redraw is NOT triggered by a normal run**

Run: `uv run trizod dataset test-set` again and confirm the output file is byte-identical:

```bash
uv run trizod dataset test-set
git -c core.fileMode=false diff --no-index --stat \
  trizod/dataset/pinned/TriZOD_test_set.fasta \
  data/interim/build/testset/TriZOD_test_set.fasta || true
```

Expected: sequences identical (headers may differ only if a substitution occurred; none expected here).

- [ ] **Step 4: Update docs**

If `docs/dataset/dataset-construction.md` describes the seeded test-set draw, add a short paragraph: the test set is now pinned (committed at `trizod/dataset/pinned/TriZOD_test_set.fasta`); `trizod dataset test-set` emits the pin, and `--redraw` re-establishes it. (Skip if no such section exists.)

- [ ] **Step 5: Commit**

```bash
git add docs/
git status --short
git commit -m "docs(dataset): document the pinned test set + --redraw

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** pinned reference file (Task 1) ✓; default pinned load + resolution rule (Tasks 2–3) ✓; `--redraw` re-pin (Task 3) ✓; missing-pin error (Task 3) ✓; dropped/substitution reporting in summary (Tasks 2–3) ✓; downstream unchanged — no edits to `redundancy.py`/`deploy_fasta.py` ✓; TDD unit tests mmseqs-free (Tasks 2–3) ✓; bootstrap from current build (Task 1) ✓.

**Placeholder scan:** none — every code/command step is complete.

**Type consistency:** `resolve_pinned_testset(pinned, strict) -> (test_recs, info)` used identically in Tasks 2–3; `_redraw(paths, out) -> dict[str,str]` and `_write_pin(pin_path, test_recs, extra)` match their call sites; `pinned_testset` path name consistent across Tasks 1, 3, 4.

**Note on packaging:** `trizod/dataset/pinned/*.fasta` is committed and works for source/`uv run` use. If wheels must ship it, add a `package-data`/`force-include` rule in `pyproject.toml` — out of scope here (source use only).
