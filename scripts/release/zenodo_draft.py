"""Create or update the Zenodo draft of a TriZOD release through the Zenodo REST API.

It never publishes: publishing stays a deliberate click on the draft page.

The token is read from the macOS keychain (service ``zenodo-token``), else from
$ZENODO_TOKEN, and is never printed. Create it at zenodo.org > Applications >
Personal access tokens with the scopes deposit:write and deposit:actions, then store it:

    security add-generic-password -a "$USER" -s zenodo-token -w

(the command prompts for the token, so it never lands in the shell history).

Usage:
    # new version of the latest published record, metadata from zenodo.json, upload all files
    uv run --with requests python scripts/release/zenodo_draft.py new-version \\
        --record 21381698 --files data/release/zenodo-upload-v0.4.0 --dry-run
    # fill an existing draft: metadata from zenodo.json, then all files (or --metadata-only)
    uv run --with requests python scripts/release/zenodo_draft.py fill \\
        --draft <deposition id> --files data/release/zenodo-upload-v0.4.0 [--metadata-only]
    # replace some files in an existing draft (e.g. after patching the reserved DOI)
    uv run --with requests python scripts/release/zenodo_draft.py replace-files \\
        --draft <deposition id> data/release/zenodo-upload-v0.4.0/CITATION.cff ...
    # show a draft's state (metadata summary, files, reserved DOI)
    uv run --with requests python scripts/release/zenodo_draft.py show --draft <deposition id>
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

API = "https://zenodo.org/api"

# Keys of zenodo.json the legacy deposit API accepts in "metadata".
METADATA_KEYS = (
    "title",
    "description",
    "version",
    "creators",
    "keywords",
    "license",
    "upload_type",
    "related_identifiers",
    "notes",
    "communities",
    "contributors",
    "language",
)


def _token() -> str:
    # The keychain wins: a ZENODO_TOKEN left in the shell profile for another
    # project may lack the rights on this record (it answered 403).
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-s", "zenodo-token", "-w"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    tok = os.environ.get("ZENODO_TOKEN")
    if tok:
        return tok.strip()
    sys.exit(
        "No Zenodo token: store one with "
        '`security add-generic-password -a "$USER" -s zenodo-token -w` '
        "or set ZENODO_TOKEN."
    )


def _session():
    import requests

    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {_token()}"
    return s


def _check(r, what: str):
    if not r.ok:
        sys.exit(f"{what} failed: HTTP {r.status_code} {r.text[:500]}")
    return r.json() if r.content else {}


def _call(send, what: str, tries: int = 7):
    """send() -> response; retried on connection errors and HTTP 5xx (Zenodo overload)."""
    import requests

    err = ""
    for i in range(tries):
        try:
            r = send()
        except requests.RequestException as e:
            err = f"{type(e).__name__}: {e}"[:200]
        else:
            if r.status_code < 500:
                return _check(r, what)
            err = f"HTTP {r.status_code}"
        if i < tries - 1:
            wait = min(120, 10 * 2**i)
            print(f"  {what}: {err}; retry in {wait} s", flush=True)
            time.sleep(wait)
    sys.exit(f"{what} failed after {tries} tries: {err}")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_metadata(files_dir: Path, tag_url: str | None) -> dict:
    z = json.loads((files_dir / "zenodo.json").read_text())
    md = {k: z[k] for k in METADATA_KEYS if k in z}
    md["publication_date"] = datetime.date.today().isoformat()
    md["prereserve_doi"] = True
    if tag_url:
        rel = md.setdefault("related_identifiers", [])
        if not any(r.get("identifier") == tag_url for r in rel):
            rel.append(
                {
                    "identifier": tag_url,
                    "relation": "isSupplementTo",
                    "scheme": "url",
                    "resource_type": "software",
                }
            )
    return md


def show(s, draft: int) -> dict:
    d = _call(lambda: s.get(f"{API}/deposit/depositions/{draft}"), "read draft")
    md = d.get("metadata", {})
    pre = md.get("prereserve_doi", {})
    print(f"draft {d['id']}: state {d.get('state')}, submitted {d.get('submitted')}")
    print(f"  title:   {md.get('title')}")
    print(f"  version: {md.get('version')}  licence: {md.get('license')}")
    print(f"  creators: {[c.get('name') for c in md.get('creators', [])]}")
    print(f"  reserved DOI: {pre.get('doi') if isinstance(pre, dict) else pre}")
    print(
        f"  files ({len(d.get('files', []))}): {[f['filename'] for f in d.get('files', [])]}"
    )
    print(f"  edit page: https://zenodo.org/uploads/{d['id']}")
    return d


def upload(s, draft: dict, paths: list[Path], dry: bool) -> None:
    bucket = draft["links"]["bucket"]
    for p in paths:
        size = p.stat().st_size
        print(f"  upload {p.name} ({size / 1e6:.1f} MB)", flush=True)
        if dry:
            continue

        def send(p=p):
            with p.open("rb") as fh:
                return s.put(f"{bucket}/{p.name}", data=fh)

        got = _call(send, f"upload {p.name}")
        want = _md5(p)
        if got.get("checksum", "").split(":")[-1] != want:
            sys.exit(
                f"checksum mismatch for {p.name}: {got.get('checksum')} vs md5:{want}"
            )


def new_version(s, record: int, files_dir: Path, tag_url: str | None, dry: bool):
    paths = sorted(p for p in files_dir.iterdir() if p.is_file())
    md = load_metadata(files_dir, tag_url)
    print(f"new version of record {record}: {len(paths)} files from {files_dir}")
    print(
        f"  metadata: {md['title']!r}, version {md.get('version')}, "
        f"{len(md.get('creators', []))} creators, licence {md.get('license')}"
    )
    if dry:
        upload(s, {"links": {"bucket": "<bucket>"}}, paths, dry=True)
        print("dry run: nothing sent")
        return
    # The records API creates the version (the legacy newversion action answers 403
    # for a token without deposit:actions); the legacy deposit API fills it.
    r = _call(lambda: s.post(f"{API}/records/{record}/versions"), "new version")
    fill(s, int(r["id"]), files_dir, tag_url, upload_files=True)


def fill(s, draft_id: int, files_dir: Path, tag_url: str | None, upload_files: bool):
    """Set the draft's metadata from zenodo.json; with upload_files, replace all its files."""
    md = load_metadata(files_dir, tag_url)
    draft = _call(lambda: s.get(f"{API}/deposit/depositions/{draft_id}"), "read draft")
    if draft.get("submitted"):
        sys.exit("that deposition is published; files can only change in a new version")
    if isinstance(draft["metadata"].get("prereserve_doi"), dict):
        md.pop("prereserve_doi")  # already reserved; keep that DOI
    draft = _call(
        lambda: s.put(draft["links"]["self"], json={"metadata": md}), "set metadata"
    )
    if upload_files:
        for f in draft.get("files", []):
            _call(lambda f=f: s.delete(f["links"]["self"]), f"delete {f['filename']}")
        upload(s, draft, sorted(p for p in files_dir.iterdir() if p.is_file()), False)
    show(s, draft["id"])
    print("Draft saved, NOT published. Review it on the edit page, then publish there.")


def replace_files(s, draft_id: int, paths: list[Path], dry: bool):
    d = _call(lambda: s.get(f"{API}/deposit/depositions/{draft_id}"), "read draft")
    if d.get("submitted"):
        sys.exit("that deposition is published; files can only change in a new version")
    names = {p.name for p in paths}
    for f in d.get("files", []):
        if f["filename"] in names:
            print(f"  delete {f['filename']}")
            if not dry:
                _call(
                    lambda f=f: s.delete(f["links"]["self"]), f"delete {f['filename']}"
                )
    upload(s, d, paths, dry)
    if not dry:
        show(s, draft_id)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("new-version")
    a.add_argument("--record", type=int, required=True)
    a.add_argument("--files", type=Path, required=True)
    a.add_argument("--tag-url", default=None)
    a.add_argument("--dry-run", action="store_true")
    b = sub.add_parser("replace-files")
    b.add_argument("--draft", type=int, required=True)
    b.add_argument("paths", type=Path, nargs="+")
    b.add_argument("--dry-run", action="store_true")
    c = sub.add_parser("show")
    c.add_argument("--draft", type=int, required=True)
    f = sub.add_parser("fill")
    f.add_argument("--draft", type=int, required=True)
    f.add_argument("--files", type=Path, required=True)
    f.add_argument("--tag-url", default=None)
    f.add_argument("--metadata-only", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "new-version" and args.dry_run:
        new_version(None, args.record, args.files, args.tag_url, dry=True)
        return
    s = _session()
    if args.cmd == "new-version":
        new_version(s, args.record, args.files, args.tag_url, dry=False)
    elif args.cmd == "fill":
        fill(s, args.draft, args.files, args.tag_url, not args.metadata_only)
    elif args.cmd == "replace-files":
        replace_files(s, args.draft, args.paths, args.dry_run)
    else:
        show(s, args.draft)


if __name__ == "__main__":
    main()
