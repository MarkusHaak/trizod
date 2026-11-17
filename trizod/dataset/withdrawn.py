#!/usr/bin/env python3
"""BMRB entries that the archive no longer lists publicly (NMR-STAR audit I-06).

A local BMRB snapshot keeps every entry it ever downloaded, but BMRB withdraws
entries (author request, PDB obsoletion) and puts others on hold. BMRB publishes
no per-entry history and no reason, so the only checkable rule is list
membership: an entry absent from ``/v2/list_entries`` is not part of the public
archive any more.

The committed list ``trizod/dataset/data/withdrawn_entries.tsv`` records those
entries with the status BMRB's instant search gives them and the date it was
checked. ``trizod dataset build`` drops every listed entry before
deduplication (see :mod:`trizod.dataset.build`), so none reaches a tier pool,
a training set or a test set. The build reads only the committed file and never
calls the network; :func:`fetch_unlisted_entries` re-derives the list from the
BMRB API for a future snapshot, and ``main`` writes it back for review.

Refresh (network, ~2 requests per unlisted entry)::

    uv run python -m trizod.dataset.withdrawn --bmrb-dir data/raw/bmrb_entries
"""

from __future__ import annotations

import argparse
import datetime
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WITHDRAWN_TSV = Path(__file__).parent / "data" / "withdrawn_entries.tsv"
BMRB_API = "https://api.bmrb.io/v2"
SOURCE = "BMRB API /v2/list_entries?database=macromolecules; /v2/entry; /v2/instant"
COLUMNS = ("entry_id", "status", "date_checked", "source", "detail")
#: Every status is excluded from the build; the value only says why.
STATUSES = ("withdrawn", "on_hold", "unlisted")

HEADER_COMMENT = """\
# BMRB entries in TriZOD's local snapshot that BMRB no longer lists publicly.
# `trizod dataset build` drops every entry listed here before deduplication
# (NMR-STAR parsing audit 2026-11, issue I-06), so none reaches a tier pool,
# a training set or a test set. Every status listed here is excluded.
# status: withdrawn = /v2/instant labels it "Entry has been withdrawn by the author."
#         on_hold   = /v2/instant labels it "Entry is on hold"
#         unlisted  = absent from /v2/list_entries with no instant-search label
# Refresh (network): python -m trizod.dataset.withdrawn --bmrb-dir data/raw/bmrb_entries
# then review the diff and commit it; the build itself never calls the network.
"""

#: ``get(url) -> (http_status, parsed_json_or_None)``; injectable for tests.
Getter = Callable[[str], tuple[int, Any]]


@dataclass(frozen=True)
class UnlistedEntry:
    entry_id: str
    status: str
    date_checked: str
    source: str
    detail: str = ""


def _id_key(entry_id: str):
    return (0, int(entry_id), "") if entry_id.isdigit() else (1, 0, entry_id)


def load_withdrawn(path: Path | None = None) -> dict[str, UnlistedEntry]:
    """Read the list (default: the committed one) as ``{entry_id: entry}``.

    Raises ``ValueError`` on a malformed header, an unknown status or a
    duplicated entry ID, rather than silently excluding the wrong set.
    """
    path = WITHDRAWN_TSV if path is None else Path(path)
    entries: dict[str, UnlistedEntry] = {}
    header = None
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip() or raw.startswith("#"):
            continue
        fields = raw.split("\t")
        if header is None:
            header = tuple(fields)
            if header != COLUMNS:
                raise ValueError(f"{path}:{lineno}: header {header} != {COLUMNS}")
            continue
        if len(fields) != len(COLUMNS):
            raise ValueError(
                f"{path}:{lineno}: expected {len(COLUMNS)} tab-separated "
                f"fields, got {len(fields)}"
            )
        entry = UnlistedEntry(*(f.strip() for f in fields))
        if entry.status not in STATUSES:
            raise ValueError(
                f"{path}:{lineno}: status {entry.status!r} not in {STATUSES}"
            )
        if entry.entry_id in entries:
            raise ValueError(f"{path}:{lineno}: duplicate entry {entry.entry_id}")
        entries[entry.entry_id] = entry
    if header is None:
        raise ValueError(f"{path}: no header row")
    return entries


def write_withdrawn(entries: Iterable[UnlistedEntry], path: Path) -> None:
    """Write ``entries`` in the canonical format (numeric ID order)."""
    rows = sorted(entries, key=lambda e: _id_key(e.entry_id))
    lines = [HEADER_COMMENT.rstrip("\n"), "\t".join(COLUMNS)]
    lines += ["\t".join(getattr(e, c) for c in COLUMNS) for e in rows]
    Path(path).write_text("\n".join(lines) + "\n")


def local_entry_ids(bmrb_dir: Path) -> set[str]:
    """Entry IDs of a BMRB snapshot (``<bmrb_dir>/bmr<ID>/bmr<ID>_3.str``): the
    entries the scoring pipeline reads, found the way it finds them
    (:func:`trizod.pipeline.find_bmrb_files`), withdrawn ones included."""
    # imported here: trizod.pipeline imports this module when it filters
    from trizod.pipeline import find_bmrb_files

    return set(find_bmrb_files(bmrb_dir, exclude_withdrawn=False))


def http_get_json(url: str, timeout: float = 60.0) -> tuple[int, Any]:
    """GET ``url`` and return ``(status, parsed JSON)``; JSON is None on error."""
    req = urllib.request.Request(url, headers={"Application": "trizod"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, None


def _classify(label: str) -> str:
    low = label.lower()
    if "withdrawn" in low:
        return "withdrawn"
    if "on hold" in low:
        return "on_hold"
    return "unlisted"


def fetch_unlisted_entries(
    local_ids: Iterable[str],
    *,
    get: Getter | None = None,
    api: str = BMRB_API,
    today: str | None = None,
) -> list[UnlistedEntry]:
    """Diff ``local_ids`` against BMRB's public list and classify the rest.

    NETWORK: one ``list_entries`` call, then ``/entry/<id>?tag=Entry.ID`` (HTTP
    status only) and ``/instant?term=<id>`` (status label) per unlisted entry.
    Pass ``get`` to stub the transport. Returns entries in numeric ID order.
    """
    get = http_get_json if get is None else get
    today = today or datetime.date.today().isoformat()
    code, listed = get(f"{api}/list_entries?database=macromolecules")
    if code != 200 or not isinstance(listed, list) or not listed:
        raise RuntimeError(
            f"BMRB list_entries returned HTTP {code} with no usable ID list; "
            "refusing to classify every local entry as unlisted"
        )
    public = {str(x) for x in listed}
    missing = sorted({str(x) for x in local_ids} - public, key=_id_key)

    out = []
    for eid in missing:
        entry_code, _ = get(f"{api}/entry/{eid}?tag=Entry.ID")
        _, hits = get(f"{api}/instant?term={eid}")
        label = next(
            (
                str(h.get("label", ""))
                for h in (hits or [])
                if isinstance(h, dict) and str(h.get("value")) == eid
            ),
            "",
        )
        detail = (
            f"absent from list_entries; /v2/entry HTTP {entry_code}; "
            f"instant: {label or '(no hit)'}"
        )
        out.append(UnlistedEntry(eid, _classify(label), today, SOURCE, detail))
    return out


def main(argv=None) -> None:
    from trizod.paths import layout

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--bmrb-dir",
        type=Path,
        default=None,
        help="local BMRB snapshot (default: <root>/data/raw/bmrb_entries)",
    )
    ap.add_argument("--root", type=Path, default=None, help="repository root")
    ap.add_argument(
        "--out",
        type=Path,
        default=WITHDRAWN_TSV,
        help=f"list to write (default: the committed {WITHDRAWN_TSV.name})",
    )
    args = ap.parse_args(argv)
    bmrb_dir = args.bmrb_dir or layout(args.root).raw_bmrb
    local = local_entry_ids(bmrb_dir)
    if not local:
        raise SystemExit(f"no bmr<ID>/bmr<ID>_3.str files under {bmrb_dir}")
    print(f"{len(local)} local entries in {bmrb_dir}; querying {BMRB_API} ...")

    fresh = fetch_unlisted_entries(local)
    old = load_withdrawn(args.out) if args.out.exists() else {}
    new = {e.entry_id: e for e in fresh}
    for eid in sorted(new.keys() - old.keys(), key=_id_key):
        print(f"  + {eid}\t{new[eid].status}\t{new[eid].detail}")
    for eid in sorted(old.keys() - new.keys(), key=_id_key):
        print(f"  - {eid}\t(now listed, or no longer in the local snapshot)")
    write_withdrawn(fresh, args.out)
    print(f"Wrote {len(fresh)} unlisted entries -> {args.out}; review the diff.")


if __name__ == "__main__":
    main()
