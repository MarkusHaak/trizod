#!/usr/bin/env python3
"""Run the writing plan's grep lint checks (Black checklist) on the TriZOD draft prose.

Usage: python scripts/manuscript/lint_draft.py <manuscript-draft dir>
Comments are stripped before matching. Checks marked "(target 0)" must report 0 hits; the
"(review)" checks list hits for a human read. The '!' hits in S1.tex are TikZ colour syntax.
"""

import pathlib
import re
import sys

D = pathlib.Path(sys.argv[1])
files = (
    [D / "Article.tex", D / "Supplementary.tex"]
    + sorted((D / "sections").glob("*.tex"))
    + [D / "Figures/paper/fig1b_cartoon.tex", D / "Figures/paper/S1.tex"]
)
MAIN = {"front.tex", "results_a.tex", "results_b.tex", "discussion.tex"}


def strip(line):
    if line.lstrip().startswith("%"):
        return ""
    return re.sub(r"(?<!\\)%.*", "", line)


def lines(f):
    for i, l in enumerate(f.read_text().splitlines(), 1):
        s = strip(l)
        if s.strip():
            yield i, s


def check(
    name, pat, flags=re.I, only=None, show=True, expect0=True, exclude_markers=False
):
    hits = []
    for f in files:
        if only and f.name not in only:
            continue
        for i, s in lines(f):
            t = s
            if exclude_markers:
                t = re.sub(r"\\authornote\{.*", "", t)
            for m in re.finditer(pat, t, flags):
                hits.append(
                    (f.name, i, t[max(0, m.start() - 60) : m.end() + 60].strip())
                )
    print(f"\n### {name}: {len(hits)} hit(s){' (target 0)' if expect0 else ''}")
    if show:
        for h in hits[:60]:
            print(f"  {h[0]}:{h[1]}: ...{h[2]}...")
    return hits


check("PR-11 LLM words", r"delve|showcase|pivotal|underscore|intricate")
check("IN-2 'we are interested'", r"we are interested")
check(
    "BW-4 contractions",
    r"\b\w+n't\b|\b(it|that|there|we|they)'(s|re|ll|ve|d)\b",
    flags=0,
)
check("BW-4 exclamation (prose; ignores \\! and math)", r"(?<![\\$])![^=]", flags=0)
check("BW-5 'we' (review)", r"\bwe\b", show=True, expect0=False, only=MAIN)
check("BW-7/8 'never sees' (target 1)", r"never sees", expect0=False)
check("BW-10 ' can ' (review, main text)", r" can ", expect0=False, only=MAIN)
check(
    "BW-12 vague verbs/absolutes (review)",
    r"allows? to|enables?\b|provides?\b|\bunique\b|leakage-free|to that end",
    expect0=False,
)
check(
    "BW-12 hard bans",
    r"leakage-free|to that end|hand-picked|hand-curated|eliminates leakage",
)
check(
    "BW-14 terminology",
    r"CheZOD[- ](117|1174|1325)|\bpreset|P-score|hand-picked|G score\b|Z score\b",
)
check("BW-15 sentence starts with citation", r"^\s*\\cite", flags=0)
check("FG-3 'In Fig. X we show'", r"in (fig|figure)[^.]*we show")
check(
    "EX-3 'outperforms|contradicts|wrong' (review)",
    r"outperform|contradict|\bwrong\b",
    expect0=False,
)
check("EX-4 'demonstrate superiority'", r"demonstrate superiority|superior to")
check("CO-1 'halves'", r"\bhalves\b|half as much|halving")
check("Title/abstract: machine-learning-ready", r"machine-learning-ready")
check(
    "DO-NOT-USE macros",
    r"\\(pctIRLSNonConvergence|nInheritedUnparsableEntries|nInheritedInvalidEntries|nInheritedErroredEntries|nDisProtIDRsClaim|nDisProtProteinsClaim|nBMRBExperimentsClaim)\b",
    flags=0,
)
check(
    "Banned style phrases",
    r"largest shift-derived|order of magnitude more|curation was a stand-in",
    expect0=False,
)
# subsection headings (case)
print("\n### BW-14 headings (sentence case, read once)")
for f in files:
    for i, s in lines(f):
        m = re.search(r"\\(sub)*section\*?\{(.*)\}", s)
        if m and f.name not in ("S1.tex",):
            print(f"  {f.name}:{i}: {m.group(2)[:150]}")
