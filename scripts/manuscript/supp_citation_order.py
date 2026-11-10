#!/usr/bin/env python3
"""Order of first citation of the Supplementary items, for the Nature-style renumbering.

Nature journals number Supplementary Figures, Tables and Notes 1, 2, ... in the order in which
the main text first cites them, without letter prefixes, and write "Supplementary" before every
item. The draft cites them by writing-plan ID in plain text ("Supplementary Fig.~S5",
"Table~T3", "Note~N4"). This script reads the main-text section files in the order in which
``Article.tex`` inputs them, skips figure legends (``sections/fig*.tex``), comments,
``\\authornote{}`` and ``\\needsnumber{}``, and prints

* the plan IDs of each kind in order of first citation, as the ``\\SuppFigOrder`` and
  ``\\SuppTabOrder`` lines of ``Supplementary.tex`` (paste them before setting
  ``\\SuppFinalNumberstrue``) and the note order for ``sections/supp_text.tex``;
* items that the main text never cites (NCS: every item must be cited);
* citations whose item is not preceded by the word "Supplementary" (e.g. the second item of
  "Supplementary Fig.~S3 and Table~T3"), which Nature style writes out in full.

Run (code worktree root):
    python3 scripts/manuscript/supp_citation_order.py --draft <manuscript-draft dir>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

N_ITEMS = {"S": 17, "T": 12, "N": 7}
KIND = {"S": "Figures", "T": "Tables", "N": "Notes"}
ITEM = re.compile(
    r"(?P<sup>Supplementary[~\s]+)?(?P<word>Figs?\.|Figures?|Tables?|Notes?)"
    r"(?P<ids>(?:[~\s]*(?:,|and|--|–)?[~\s]*[STN]\d{1,2}\b)+)"
)


def strip_cmd(t: str, cmd: str) -> str:
    out, i = [], 0
    while True:
        j = t.find("\\" + cmd + "{", i)
        if j < 0:
            out.append(t[i:])
            return "".join(out)
        out.append(t[i:j])
        k, depth = j + len(cmd) + 2, 1
        while depth and k < len(t):
            depth += {"{": 1, "}": -1}.get(t[k], 0)
            k += 1
        i = k


def clean(t: str) -> str:
    t = "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in t.splitlines())
    for c in ("authornote", "needsnumber"):
        t = strip_cmd(t, c)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--draft", type=Path, required=True, help="the manuscript-draft directory"
    )
    a = ap.parse_args()
    art = clean((a.draft / "Article.tex").read_text())
    files = re.findall(r"\\input\{(sections/[^}]+?)(?:\.tex)?\}", art)
    body = [f for f in files if not Path(f).name.startswith("fig")]
    seen = {k: [] for k in N_ITEMS}
    no_sup = []
    for f in body:
        text = clean((a.draft / f"{f}.tex").read_text())
        for m in ITEM.finditer(text):
            ids = re.findall(r"([STN])(\d{1,2})", m.group("ids"))
            for k, n in ids:
                if k + n not in seen[k]:
                    seen[k].append(k + n)
            if not m.group("sup"):
                line = text.count("\n", 0, m.start()) + 1
                no_sup.append(f"{f}.tex:{line}: {m.group(0).strip()!r}")
    print("Input order (legends skipped):", ", ".join(body))
    for k in N_ITEMS:
        order = [x[1:] for x in seen[k]]
        print(f"{KIND[k]}: {', '.join(seen[k])}")
        missing = [
            f"{k}{i}" for i in range(1, N_ITEMS[k] + 1) if f"{k}{i}" not in seen[k]
        ]
        if missing:
            print(f"  never cited in the main text: {', '.join(missing)}")
        if k == "S":
            print(f"  \\newcommand{{\\SuppFigOrder}}{{{','.join(order)}}}")
        elif k == "T":
            print(f"  \\newcommand{{\\SuppTabOrder}}{{{','.join(order)}}}")
    if no_sup:
        print(
            "Citations without the word 'Supplementary' (write it out at the renumbering):"
        )
        for s in no_sup:
            print("  " + s)


if __name__ == "__main__":
    main()
