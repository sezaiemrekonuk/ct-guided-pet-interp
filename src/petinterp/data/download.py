"""Fetch a public Zenodo record (DEEP-PSMA) onto Drive, resumably.

A splittable stage in the rule-8 sense: one archive is one unit of work, an archive already on
Drive is skipped, and every Drive write is atomic (tmp + os.replace). Each archive is first
streamed to fast local scratch with HTTP Range resume, md5-checked against Zenodo's checksum, and
only then copied to Drive — so a disconnect costs at most the archive in flight, and a truncated
download can never land under its final name.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import requests

CHUNK = 8 << 20  # 8 MiB


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(url: str, part: Path, size: int) -> None:
    """Stream url into part, resuming from whatever part already holds."""
    have = part.stat().st_size if part.exists() else 0
    if have >= size:
        return
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        if have and r.status_code != 206:  # server ignored Range: start over
            have = 0
        with open(part, "ab" if have else "wb") as f:
            for block in r.iter_content(CHUNK):
                f.write(block)


def download_zenodo(record_id: str, dst_dir: str | Path, scratch_dir: str | Path = "/content/dl") -> list[Path]:
    """Download every file of Zenodo record_id into dst_dir; returns the final paths."""
    dst_dir, scratch_dir = Path(dst_dir), Path(scratch_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    record = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=60)
    record.raise_for_status()
    out = []
    for f in record.json()["files"]:
        name, size, url = f["key"], f["size"], f["links"]["self"]
        final = dst_dir / name
        out.append(final)
        if final.exists():
            print(f"skip   {name} (already on Drive)")
            continue

        part = scratch_dir / f"{name}.part"
        print(f"fetch  {name} ({size / 1e9:.2f} GB)")
        _fetch(url, part, size)
        algo, want = f["checksum"].split(":", 1)
        if algo != "md5" or _md5(part) != want:
            part.unlink()  # corrupt or unexpected: never let it reach Drive
            raise RuntimeError(f"{name}: checksum mismatch, deleted local copy; re-run to retry")

        tmp = final.with_name(name + ".tmp")
        shutil.copyfile(part, tmp)
        os.replace(tmp, final)
        part.unlink()
        print(f"done   {name}")
    return out
