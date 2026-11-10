#!/usr/bin/env python3
r"""List every draft status marker of the TriZOD manuscript with its location and owner.

Markers (defined in numbers.tex): \pending{}, \provisional{}, \placeholder{}, \ownerdraft{XX}{},
\authornote{}, \needsnumber{}. Comments are ignored; nested markers are listed separately.
Number macros inside a marker are shown with their current value from numbers.tex.

Usage:
    python scripts/manuscript/list_markers.py --draft <manuscript-draft dir> [--md out.md]
Prints a per-type / per-file summary; with --md also writes the full Markdown inventory
(the appendix of DRAFT_README.md).
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

TYPES = [
    "pending",
    "provisional",
    "placeholder",
    "ownerdraft",
    "authornote",
    "needsnumber",
]
ORDER = [
    "sections/front.tex",
    "sections/results_a.tex",
    "sections/results_b.tex",
    "sections/discussion.tex",
    "sections/fig1.tex",
    "sections/fig2.tex",
    "sections/fig3.tex",
    "Figures/paper/fig1b_cartoon.tex",
    "sections/methods_a.tex",
    "sections/methods_b.tex",
    "Article.tex",
    "Supplementary.tex",
    "sections/supp_figs_a.tex",
    "sections/supp_figs_b.tex",
    "sections/supp_text.tex",
    "Figures/paper/S1.tex",
]
PART = {
    "sections/front.tex": "Abstract + Introduction",
    "sections/results_a.tex": "Results R1-R3",
    "sections/results_b.tex": "Results R4-R6",
    "sections/discussion.tex": "Discussion",
    "sections/fig1.tex": "Fig. 1 legend",
    "sections/fig2.tex": "Fig. 2 legend",
    "sections/fig3.tex": "Fig. 3 legend",
    "Figures/paper/fig1b_cartoon.tex": "Fig. 1b artwork",
    "sections/methods_a.tex": "Methods M1-M6",
    "sections/methods_b.tex": "Methods M7-end matter",
    "Article.tex": "Title page",
    "Supplementary.tex": "Supplementary (S6, S8, S9, S11, S13, S14)",
    "sections/supp_figs_a.tex": "Supplementary Figs (S3-S5, S15, S16)",
    "sections/supp_figs_b.tex": "Supplementary Figs (S1, S2, S7, S10, S12, S17)",
    "sections/supp_text.tex": "Supplementary Notes + Tables",
    "Figures/paper/S1.tex": "Supplementary Fig. S1 artwork",
}
INITIALS = r"(?:TS|MH|RA|SK|IP|BR|DW|MHe)"


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in text.splitlines())


def arg_end(t: str, i: int) -> int:
    depth = 0
    for j in range(i, len(t)):
        if t[j] == "{" and t[j - 1] != "\\":
            depth += 1
        elif t[j] == "}" and t[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError("unbalanced braces")


def load_macros(draft: Path) -> dict[str, str]:
    out = {}
    for line in (draft / "numbers.tex").read_text().splitlines():
        m = re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*%", line)
        if m:
            v = (
                m.group(2)
                .replace("{,}", ",")
                .replace("\\ensuremath{-}", "-")
                .replace("\\%", "%")
            )
            v = re.sub(r"\\[A-Za-z]+\s*", "", v).replace("{", "").replace("}", "")
            out[m.group(1)] = v
    return out


def plain(s: str, macros: dict[str, str], limit: int) -> str:
    s = re.sub(
        r"\\([A-Za-z]+)(\{\})?",
        lambda m: (
            f"{m.group(0)} [{macros[m.group(1)]}]"
            if m.group(1) in macros
            else m.group(0)
        ),
        s,
    )
    s = re.sub(r"\s+", " ", s).strip().replace("|", "\\|")
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def owner(kind: str, body: str, first_arg: str | None) -> str:
    if kind == "ownerdraft":
        return first_arg or "?"
    if kind == "pending":
        return "TS (rebuild)"
    if kind == "provisional":
        return "TS (DisProt regeneration)"
    if kind == "needsnumber":
        return "TS"
    if re.search(r"S\.\s?K\.|Sandro|^\s*Structure agreement", body):
        return "SK"
    if re.search(r"R\.\s?A\.|\bReid\b", body):
        return "RA"
    if re.search(r"ablation", body, re.I):
        return "TS"
    m = re.search(
        r"Owner:\s*(" + INITIALS + r"(?:\s*(?:and|/|,)\s*" + INITIALS + r")*)", body
    )
    if m:
        return re.sub(r"\s*(and|,)\s*", "/", m.group(1))
    m = re.match(
        r"\s*(?:\\textbf\{)?(" + INITIALS + r"(?:\s*(?:and|/)\s*" + INITIALS + r")*)\b",
        body,
    )
    if m:
        return re.sub(r"\s*and\s*", "/", m.group(1))
    found = re.findall(
        r"\b("
        + INITIALS
        + r")\b(?= (?:decides|confirms|picks|sets|frames|reviews|due|rewrites))",
        body,
    )
    if found:
        return "/".join(dict.fromkeys(found))
    if re.search(r"\b(all authors|each author)\b", body, re.I):
        return "all authors"
    return "TS/MH"


def scan(draft: Path, macros: dict[str, str], limit: int = 260):
    items = []
    for rel in ORDER:
        f = draft / rel
        if not f.exists():
            continue
        raw = f.read_text()
        t = strip_comments(raw)
        starts = [0] + [m.end() for m in re.finditer("\n", t)]
        for kind in TYPES:
            for m in re.finditer(r"\\" + kind + r"\s*\{", t):
                first = None
                i = m.end() - 1
                e = arg_end(t, i)
                body = t[i + 1 : e - 1]
                if kind == "ownerdraft":
                    first = body
                    j = e
                    while t[j].isspace():
                        j += 1
                    e2 = arg_end(t, j)
                    body = t[j + 1 : e2 - 1]
                if "#4" in body or "#1" in body:  # a macro definition, not an item
                    continue
                line = max(k for k, s in enumerate(starts, 1) if s <= m.start())
                items.append(
                    {
                        "file": rel,
                        "line": line,
                        "kind": kind,
                        "owner": owner(kind, body, first),
                        "text": plain(body, macros, limit),
                    }
                )
        # \SuppPlaceholderFig{n}{title}{legend}{status}: one placeholder figure each (Supplementary.tex)
        for m in re.finditer(r"\\SuppPlaceholderFig\{(\d+)\}\{", t):
            args, j = [], m.end() - 1
            for _ in range(3):
                while t[j].isspace():
                    j += 1
                e = arg_end(t, j)
                args.append(t[j + 1 : e - 1])
                j = e
            title, _legend, status = args
            line = max(k for k, s in enumerate(starts, 1) if s <= m.start())
            items.append(
                {
                    "file": rel,
                    "line": line,
                    "kind": "placeholder",
                    "owner": owner("placeholder", status, None),
                    "text": plain(
                        f"Supplementary Fig. S{m.group(1)} ({title}): {status}",
                        macros,
                        limit,
                    ),
                }
            )
    items.sort(key=lambda d: (ORDER.index(d["file"]), d["line"]))
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--draft", type=Path, required=True, help="the manuscript-draft directory"
    )
    ap.add_argument(
        "--md", type=Path, default=None, help="also write the Markdown inventory here"
    )
    a = ap.parse_args()
    macros = load_macros(a.draft)
    items = scan(a.draft, macros)
    by = Counter(d["kind"] for d in items)
    print("markers by type:", {k: by.get(k, 0) for k in TYPES}, "total", len(items))
    tab = defaultdict(Counter)
    for d in items:
        tab[d["file"]][d["kind"]] += 1
    print(f"{'file':40s} " + " ".join(f"{k[:11]:>11s}" for k in TYPES))
    for rel in ORDER:
        if rel in tab:
            print(f"{rel:40s} " + " ".join(f"{tab[rel].get(k, 0):11d}" for k in TYPES))
    if a.md:
        L = []
        L.append("| Marker | " + " | ".join(f"`\\{k}`" for k in TYPES) + " |")
        L.append("|---|" + "---:|" * len(TYPES))
        for rel in ORDER:
            if rel in tab:
                L.append(
                    f"| `{rel}` ({PART[rel]}) | "
                    + " | ".join(str(tab[rel].get(k, 0)) for k in TYPES)
                    + " |"
                )
        L.append(
            "| **Total** | " + " | ".join(f"**{by.get(k, 0)}**" for k in TYPES) + " |"
        )
        L.append("")
        for kind in [
            "authornote",
            "placeholder",
            "ownerdraft",
            "provisional",
            "needsnumber",
            "pending",
        ]:
            sel = [d for d in items if d["kind"] == kind]
            L.append(f"### `\\{kind}` ({len(sel)})")
            L.append("")
            if kind == "pending":
                L.append(
                    "Grouped by line; each cell lists the wrapped text with the current macro values. "
                    "Owner: TS (moves with the rebuild or a planned re-run)."
                )
                L.append("")
                L.append("| Location | Wrapped text |")
                L.append("|---|---|")
                grp = defaultdict(list)
                for d in sel:
                    grp[(d["file"], d["line"])].append(d["text"])
                for (rel, ln), texts in grp.items():
                    L.append(f"| `{rel}:{ln}` | " + " · ".join(texts) + " |")
            else:
                L.append("| Location | Owner | Text |")
                L.append("|---|---|---|")
                for d in sel:
                    L.append(
                        f"| `{d['file']}:{d['line']}` | {d['owner']} | {d['text']} |"
                    )
            L.append("")
        a.md.write_text("\n".join(L) + "\n")
        print("wrote", a.md)


if __name__ == "__main__":
    main()
