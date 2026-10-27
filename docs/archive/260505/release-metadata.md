# Release metadata — Zenodo deposit prep

## What changed

Three small additions to the repo set up the workflow for a versioned dataset deposit on Zenodo (and a parallel GitHub Releases entry for convenience downloads).

| File | Purpose |
|---|---|
| `.zenodo.json` | Zenodo metadata (title, description, creators, keywords, license, related identifiers). Read by the GitHub-Zenodo integration on the first tagged release to populate the deposit page. |
| `CITATION.cff` | Standard `cff-version 1.2.0` citation file. Picked up by GitHub's "Cite this repository" button and by tools like Zotero. |
| `README.md` | New "Releases & re-referenced dataset" section pointing to the `.str` output, the Zenodo files, and the local `data/release/` layout. |

`data/release/` is already gitignored (the existing `/data/` rule covers it). No new entries were added to `.gitignore`.

## Why two metadata files

- **Zenodo** wants `.zenodo.json` for its deposit metadata.
- **CFF** is the open standard for "how do I cite this software/dataset" — independent of Zenodo. Citation managers and journal submission systems read it.

Both reference `version: 0.1.0-pipeline` and the same author block, so updating either is mechanical when we mint a real release.

## What still needs to happen for the actual deposit

This task only ships the *metadata*. The actual deposit happens when:

1. The maintainer authorises the GitHub-Zenodo integration on the repository (one-time, in Zenodo's web UI).
2. The pipeline rerun completes (Task 8).
3. Release artefacts (`data/release/<tier>/scores.json` + `data/release/<tier>/str/*.str`) are bundled into a tarball.
4. A git tag (e.g. `v0.1.0-pipeline`) is pushed.
5. A GitHub release is created with the tarball attached.
6. Zenodo automatically mints a DOI and populates the deposit page from `.zenodo.json`.

None of steps 1-6 happen automatically tonight. They're documented in `README.md` so a future contributor (or future-self) can pick them up.

## Talking points for the slide

- **What:** `.zenodo.json` + `CITATION.cff` + README section ship today; first tagged release publishes the dataset to Zenodo with a DOI.
- **Why:** Zenodo gives a permanent, citable, FAIR-compliant home for the re-referenced dataset; modern NMR community standard (e.g. the 100-Protein NMR Nature Sci Data 2023 dataset uses Zenodo).
- **Status on the slide:** "DOI placeholder — pending first tagged release."

## Commits

- `d9efcc3` — `chore(release): add .zenodo.json, CITATION.cff, README releases section`
