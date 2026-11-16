#!/usr/bin/env python3
"""Order of first citation of the Supplementary items, for the Nature-style renumbering.

Nature journals number Supplementary Figures, Tables and Notes 1, 2, ... in the order in which
the main text first cites them, without letter prefixes. The draft cites every item with the
macros of ``suppref.tex`` and its writing-plan ID: ``\\sfig{5}``, ``\\sfigs{5}{7}``,
``\\stab{2}``, ``\\stabs{2}{3}``, ``\\snote{2}``, ``\\snotes{2}{4}`` and ``\\sref{sfig:S5}``
(older plain-text citations, "Supplementary Fig.~S5", are still read and listed, so they can
be converted). ``Supplementary.tex`` maps the plan IDs to printed numbers through
``\\SuppFigOrder``, ``\\SuppTabOrder`` and ``\\SuppNoteOrder``.

The script reads the main-text section files in the order in which ``Article.tex`` inputs them
(Introduction, Results, Discussion, Methods), skips comments, ``\\authornote{}`` and
``\\needsnumber{}``, and places the display-item legends (``sections/fig*.tex``,
``sections/table*.tex``) according to ``--legends``:

* ``skip`` (default; the documented procedure): legends do not count;
* ``inline``: each legend counts right after the main-text file that first cites its display
  item (``Fig.~\\ref{fig:n}``, ``Table~\\ref{tab:n}``), the closest proxy for "in place";
* ``after``: all legends after the main text.

Items the main text never cites are then ordered by their first citation in the SI
(``Supplementary.tex`` and the ``sections/supp_*.tex`` files, in print order), and items cited
nowhere follow in plan order. The item set is read from the SI itself: figures from
``\\SuppFigFrom{n}`` / ``\\SuppFigHere{n}``, tables from ``\\label{stab:Tn}``, notes from
``\\SuppNote{n}`` (retired plan IDs, such as S19 and T13, are therefore not items).

It prints the three ``\\newcommand`` lines for ``Supplementary.tex``, says whether they differ
from the ones there, and lists items never cited in the main text (NCS: every item must be
cited), citations of IDs that are not items (retired or misspelled) and plain-text citations
that bypass the macros.

Run (code worktree root):
    python3 scripts/manuscript/supp_citation_order.py --draft <manuscript-draft dir> \
        [--legends skip|inline|after]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

KIND = {"S": "Figures", "T": "Tables", "N": "Notes"}
ORDER_CMD = {"S": "SuppFigOrder", "T": "SuppTabOrder", "N": "SuppNoteOrder"}
MACRO_KIND = {"fig": "S", "tab": "T", "note": "N"}
# \sfig{n}, \sfigs{a}{b}, \stab..., \snote...; \sref{sfig:Sn}; legacy plain text
CITE = re.compile(
    r"\\s(?P<mk>fig|tab|note)(?P<pl>s)?\{(?P<a>\d+)\}(?(pl)\{(?P<b>\d+)\})"
    r"|\\sref\{s(?:fig|tab|note):(?P<rk>[STN])(?P<rn>\d+)\}"
    r"|(?P<plain>Supplementary[~\s]+(?:Fig(?:ure)?s?\.?|Tables?|Notes?)"
    r"(?P<ids>(?:[~\s]*(?:,|and|--|–)?[~\s]*[STN]\d{1,2}\b)+))"
)
DISPLAY_REF = re.compile(r"\\ref\{(fig|tab):(\d+)\}")


def strip_cmd(t: str, cmd: str) -> str:
    """``t`` without every ``\\cmd{...}`` (balanced braces)."""
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


def cites(text: str):
    """``(item, is_plain_text, line)`` for every citation in ``text``, in text order."""
    for m in CITE.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        if m.group("mk"):
            k = MACRO_KIND[m.group("mk")]
            yield k + m.group("a"), False, line
            if m.group("b"):
                yield k + m.group("b"), False, line
        elif m.group("rk"):
            yield m.group("rk") + m.group("rn"), False, line
        else:
            for k, n in re.findall(r"([STN])(\d{1,2})", m.group("ids")):
                yield k + n, True, line


def items(draft: Path) -> dict:
    """The Supplementary items that exist, read from the SI sources."""
    files = [
        draft / "Supplementary.tex",
        *sorted((draft / "sections").glob("supp_*.tex")),
    ]
    txt = "\n".join(clean(f.read_text()) for f in files if f.exists())
    found = {
        "S": {int(n) for n in re.findall(r"\\SuppFig(?:From|Here)\{(\d+)\}", txt)},
        "T": {int(n) for n in re.findall(r"\\label\{stab:T(\d+)\}", txt)},
        "N": {int(n) for n in re.findall(r"\\SuppNote\{(\d+)\}", txt)},
    }
    return {k: sorted(v) for k, v in found.items()}


def si_files(draft: Path) -> list[str]:
    """SI sources in print order: Supplementary.tex, then the section files it reads
    (``\\SuppFigFrom{n}{file}``, ``\\input``), then any other ``sections/supp_*.tex``."""
    sup = clean((draft / "Supplementary.tex").read_text())
    order = ["Supplementary.tex"]
    for m in re.finditer(
        r"\\SuppFigFrom\{\d+\}\{([^}]+)\}|\\input\{(sections/[^}]+?)(?:\.tex)?\}", sup
    ):
        f = f"sections/{m.group(1)}.tex" if m.group(1) else f"{m.group(2)}.tex"
        if f not in order and (draft / f).exists():
            order.append(f)
    for f in sorted((draft / "sections").glob("supp_*.tex")):
        r = f"sections/{f.name}"
        if r not in order:
            order.append(r)
    return order


def current_orders(draft: Path) -> dict:
    sup = (draft / "Supplementary.tex").read_text()
    out = {}
    for k, cmd in ORDER_CMD.items():
        m = re.search(r"\\newcommand\{\\" + cmd + r"\}\{([^}]*)\}", sup)
        out[k] = m.group(1).strip() if m else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--draft", type=Path, required=True, help="the manuscript-draft directory")  # fmt: skip
    ap.add_argument("--legends", choices=("skip", "inline", "after"), default="skip",
                    help="where figure and table legends count (default: skip)")  # fmt: skip
    a = ap.parse_args()
    D = a.draft
    art = clean((D / "Article.tex").read_text())
    inputs = [f"{f}.tex" for f in re.findall(r"\\input\{(sections/[^}]+?)(?:\.tex)?\}", art)]  # fmt: skip
    is_legend = lambda f: Path(f).name.startswith(("fig", "table"))  # noqa: E731
    body = [f for f in inputs if not is_legend(f)]
    legends = {}
    for f in inputs:
        m = re.match(r"(fig|table)(\d+)\.tex$", Path(f).name)
        if m:
            legends[("fig" if m.group(1) == "fig" else "tab", m.group(2))] = f
    text = {f: clean((D / f).read_text()) for f in set(inputs) if (D / f).exists()}

    # reading order of the main text
    seq = []
    if a.legends == "inline":
        placed = set()
        for f in body:
            seq.append(f)
            for kind, n in DISPLAY_REF.findall(text.get(f, "")):
                lf = legends.get((kind, n))
                if lf and lf not in placed:
                    seq.append(lf)
                    placed.add(lf)
        seq += [lf for lf in legends.values() if lf not in placed]
    elif a.legends == "after":
        seq = body + list(legends.values())
    else:
        seq = body

    have = items(D)
    known = {f"{k}{n}" for k, ns in have.items() for n in ns}
    first_main = {k: [] for k in KIND}
    where, plain, unknown = {}, [], []
    for f in seq:
        for it, is_plain, line in cites(text.get(f, "")):
            if it not in known:
                unknown.append(f"{f}:{line}: {it}")
                continue
            if it not in first_main[it[0]]:
                first_main[it[0]].append(it)
                where[it] = f
            if is_plain:
                plain.append(f"{f}:{line}: {it}")
    first_si = {k: [] for k in KIND}
    for f in si_files(D):
        for it, is_plain, line in cites(clean((D / f).read_text())):
            if it not in known:
                unknown.append(f"{f}:{line}: {it} (SI)")
                continue
            if it not in first_main[it[0]] and it not in first_si[it[0]]:
                first_si[it[0]].append(it)
                where[it] = f
            if is_plain:
                plain.append(f"{f}:{line}: {it} (SI)")

    cur = current_orders(D)
    print(f"Main text in reading order (legends: {a.legends}): {', '.join(seq)}")
    print("Items in the SI: " + "; ".join(f"{KIND[k]} {len(v)} ({', '.join(f'{k}{n}' for n in v)})" for k, v in have.items()))  # fmt: skip
    for k in KIND:
        nowhere = [
            f"{k}{n}" for n in have[k] if f"{k}{n}" not in first_main[k] + first_si[k]
        ]
        order = first_main[k] + first_si[k] + nowhere
        line = f"\\newcommand{{\\{ORDER_CMD[k]}}}{{{','.join(x[1:] for x in order)}}}"
        same = cur[k] == ",".join(x[1:] for x in order)
        print(
            f"\n{KIND[k]}: first cited in the main text: {', '.join(first_main[k]) or '-'}"
        )
        if first_si[k]:
            print(f"  first cited only in the SI: {', '.join(f'{x} ({Path(where[x]).stem})' for x in first_si[k])}")  # fmt: skip
        if nowhere:
            print(f"  cited nowhere (appended in plan order): {', '.join(nowhere)}")
        print(f"  {line}")
        print(f"  Supplementary.tex now: {cur[k]}  -> {'SAME' if same else 'DIFFERS: update it'}")  # fmt: skip
    if unknown:
        print("\nCitations of IDs that are not SI items (retired or misspelled):")
        for s in unknown:
            print("  " + s)
    if plain:
        print("\nPlain-text citations (convert to \\sfig / \\stab / \\snote):")
        for s in plain:
            print("  " + s)


if __name__ == "__main__":
    main()
