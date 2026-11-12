r"""Word counts per main-text section of the TriZOD draft, with numbers.tex macros expanded.

Usage: python scripts/manuscript/wordcount.py <manuscript-draft dir> <scratch output dir>
Prints JSON per section; writes the expanded section files and counts.json to the output dir.
Requires texcount (TeX Live) on PATH.

Counted: Abstract, Introduction, Results, Discussion, Methods, end matter, and the legend
(the ``\caption{}``) of every display item that ``Article.tex`` inputs -- ``sections/fig<n>.tex``
and ``sections/table<n>.tex`` (Figs. 1-5 and Table 1 in the story-blueprint draft). The JSON
ends with a ``Totals`` entry: the NCS main text (Introduction + Results + Discussion) with and
without headings (texcount and whitespace split), and the summed legend words.

texcount sees \nChainsScored{} as a macro and skips it; the rendered text has a number there.
So each section is pre-processed: macros -> their values, draft markers resolved
(\pending/\provisional/\ownerdraft -> their text; \authornote/\needsnumber/\placeholder -> removed
and counted separately), then texcount runs on the result.
"""

import json
import pathlib
import re
import subprocess
import sys

D = pathlib.Path(sys.argv[1])
OUT = pathlib.Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

macros = {}
for line in (D / "numbers.tex").read_text().splitlines():
    m = re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*(%.*)?$", line)
    if m:
        v = m.group(2)
        v = v.replace("{,}", ",").replace("\\ensuremath{-}", "-")
        v = re.sub(r"\\[A-Za-z]+\s*", "", v).replace("{", "").replace("}", "")
        if "%" in v and "\\%" not in v:
            v = v.replace("%", "\\%")
        macros[m.group(1)] = v


def strip_comments(t):
    return "\n".join(
        re.sub(r"(?<!\\)%.*", "", l) if not l.lstrip().startswith("%TC") else l
        for l in t.splitlines()
    )


def arg_span(t, i):
    """t[i] == '{' ; return index after the matching brace."""
    depth = 0
    for j in range(i, len(t)):
        if t[j] == "{" and t[j - 1] != "\\":
            depth += 1
        elif t[j] == "}" and t[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError("unbalanced")


def process(t, removed):
    # remove markers whose content is not main text
    for cmd in ("authornote", "needsnumber", "placeholder"):
        while True:
            m = re.search(r"\\" + cmd + r"\s*\{", t)
            if not m:
                break
            e = arg_span(t, m.end() - 1)
            removed[cmd].append(t[m.end() : e - 1])
            t = t[: m.start()] + t[e:]
    # \ownerdraft{XX}{text} -> text
    while True:
        m = re.search(r"\\ownerdraft\s*\{[^{}]*\}\s*\{", t)
        if not m:
            break
        e = arg_span(t, m.end() - 1)
        t = t[: m.start()] + t[m.end() : e - 1] + t[e:]
    # \pending{x}, \provisional{x} -> x
    for cmd in ("pending", "provisional"):
        while True:
            m = re.search(r"\\" + cmd + r"\s*\{", t)
            if not m:
                break
            e = arg_span(t, m.end() - 1)
            t = t[: m.start()] + t[m.end() : e - 1] + t[e:]

    # number macros
    def rep(m):
        name = m.group(1)
        return macros[name] if name in macros else m.group(0)

    t = re.sub(r"\\([A-Za-z]+)(\{\})?", rep, t)
    return t


def texcount(path):
    full = subprocess.run(
        ["texcount", "-utf8", str(path)], capture_output=True, text=True
    ).stdout

    def g(k):
        m = re.search(k + r":\s*(\d+)", full)
        return int(m.group(1)) if m else 0

    return {
        "text": g("Words in text"),
        "headers": g("Words in headers"),
        "captions": g("Words outside text \\(captions, etc.\\)"),
        "math_inline": g("Number of math inlines"),
        "math_display": g("Number of math displayed"),
    }


sec = D / "sections"
front = strip_comments((sec / "front.tex").read_text())
abstract = front[
    front.index("\\begin{abstract}") + len("\\begin{abstract}") : front.index(
        "\\end{abstract}"
    )
]
intro = front[front.index("\\end{abstract}") + len("\\end{abstract}") :]
intro = intro.replace("\\ifdefined\\afterabstract\\afterabstract\\fi", "")
parts = {
    "Abstract": abstract,
    "Introduction": intro,
    "Results": strip_comments((sec / "results_a.tex").read_text())
    + "\n"
    + strip_comments((sec / "results_b.tex").read_text()),
    "Discussion": strip_comments((sec / "discussion.tex").read_text()),
    "Methods (M1-M11, statistics, LLM statement)": "",
}
# every display item Article.tex inputs: figures, then tables, by number
_inputs = re.findall(
    r"\\input\{sections/((?:fig|table)\d+)(?:\.tex)?\}",
    strip_comments((D / "Article.tex").read_text()),
)
_disp = sorted(
    dict.fromkeys(_inputs),
    key=lambda n: (n.startswith("table"), int(re.sub(r"\D", "", n))),
)
for _n in _disp:
    _label = f"Fig. {_n[3:]}" if _n.startswith("fig") else f"Table {_n[5:]}"
    parts[f"{_label} legend"] = strip_comments((sec / f"{_n}.tex").read_text())
mb = strip_comments((sec / "methods_b.tex").read_text())
cut = mb.index("\\section*{Data availability}")
parts["Methods (M1-M11, statistics, LLM statement)"] = (
    strip_comments((sec / "methods_a.tex").read_text()) + "\n" + mb[:cut]
)
parts["End matter (availability, acknowledgements, contributions, interests)"] = mb[
    cut:
]


def caption_only(t):
    m = re.search(r"\\caption\s*\{", t)
    e = arg_span(t, m.end() - 1)
    return t[m.end() : e - 1]


res = {}
for name, t in parts.items():
    if name.endswith(" legend"):
        t = caption_only(t)
    removed = {"authornote": [], "needsnumber": [], "placeholder": []}
    pt = process(t, removed)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    f = OUT / f"{slug}.tex"
    f.write_text(pt)
    c = texcount(f)

    # words inside removed markers (rough, whitespace split after stripping TeX)
    def plainwords(s):
        s = re.sub(r"\\[A-Za-z]+", " ", s)
        s = re.sub(r"[{}$~\\]", " ", s)
        return len(s.split())

    c["removed_authornote_words"] = sum(plainwords(x) for x in removed["authornote"])
    c["removed_needsnumber"] = len(removed["needsnumber"])
    c["removed_placeholder_words"] = sum(plainwords(x) for x in removed["placeholder"])
    c["n_authornote"] = len(removed["authornote"])
    c["n_placeholder"] = len(removed["placeholder"])
    # simple whitespace count of the de-TeXed prose as a cross-check
    s = re.sub(r"\\(citep|citet|cite|ref|label|eqref)\s*\{[^}]*\}", " ", pt)
    s = re.sub(r"\\(sub)*section\*?\s*\{([^}]*)\}", r" ", s)
    s = re.sub(r"\$[^$]*\$", " X ", s)
    s = re.sub(r"\\[A-Za-z]+\*?", " ", s)
    s = re.sub(r"[{}~\\]", " ", s)
    c["split_words"] = len([w for w in s.split() if re.search(r"[A-Za-z0-9]", w)])
    res[name] = c
main = ("Introduction", "Results", "Discussion")
leg = [k for k in res if k.endswith(" legend")]
res["Totals"] = {
    "main_text_texcount": sum(res[k]["text"] for k in main),
    "main_text_headers": sum(res[k]["headers"] for k in main),
    "main_text_texcount_with_headings": sum(
        res[k]["text"] + res[k]["headers"] for k in main
    ),
    "main_text_split": sum(res[k]["split_words"] for k in main),
    "legends_counted": leg,
    # texcount puts a \caption{} body under "captions" (words outside text)
    "legends_texcount": sum(res[k]["text"] + res[k]["captions"] for k in leg),
    "legends_split": sum(res[k]["split_words"] for k in leg),
}
print(json.dumps(res, indent=1))
(OUT / "counts.json").write_text(json.dumps(res, indent=1))
