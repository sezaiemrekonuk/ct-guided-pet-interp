"""start(): the single entry point every pipeline notebook calls (design spec D6).

Mounts Drive on Colab, resolves the data root, loads and hashes the config, records git SHA and
working-tree cleanliness, and opens runs/<exp_id>/ with config.yaml, provenance.json and an
append-only log.txt.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
MYDRIVE = Path("/content/drive/MyDrive")


def find_data_root() -> Path:
    """paths.local.yaml if present, else $PETINTERP_ROOT, else the one PETInterp/ under MyDrive.

    The auto-search exists because paths.local.yaml is gitignored, so a fresh Colab clone has
    none, and the shared folder sits at a different depth in each person's Drive.
    """
    local = REPO / "paths.local.yaml"
    if local.exists():
        return Path(yaml.safe_load(local.read_text(encoding="utf-8"))["data_root"]).expanduser()
    if os.environ.get("PETINTERP_ROOT"):
        return Path(os.environ["PETINTERP_ROOT"])
    hits = [p for pat in ("PETInterp", "*/PETInterp", "*/*/PETInterp", "*/*/*/PETInterp")
            for p in MYDRIVE.glob(pat) if p.is_dir()]
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one PETInterp/ under {MYDRIVE}, found {hits}; "
                           "set os.environ['PETINTERP_ROOT'] to pick one")
    return hits[0]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_json(path: Path, obj) -> None:
    """Atomic JSON write (rule 8)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class Run:
    cfg: dict
    cfg_hash: str
    root: Path
    dir: Path

    def path(self, rel: str) -> Path:
        """A config-relative path resolved against the data root."""
        return self.root / rel

    def log(self, msg: str) -> None:
        line = f"{dt.datetime.now().isoformat(timespec='seconds')}  {msg}"
        print(line)
        with open(self.dir / "log.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")


def start(config: str) -> Run:
    try:
        from google.colab import drive
        if not MYDRIVE.exists():
            drive.mount("/content/drive")
    except ImportError:
        pass  # not on Colab

    cfg_path = REPO / config
    text = cfg_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    cfg_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    root = find_data_root()
    run_dir = root / "runs" / cfg["exp_id"]
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.yaml").write_text(text, encoding="utf-8")
    dirty = _git("status", "--porcelain") != ""
    prov = {
        "exp_id": cfg["exp_id"], "config": config, "config_hash": cfg_hash,
        "git_sha": _git("rev-parse", "HEAD"), "dirty": dirty,
        "started": dt.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(), "gpu": _gpu(),
    }
    write_json(run_dir / "provenance.json", prov)
    run = Run(cfg, cfg_hash, root, run_dir)
    run.log(f"start {config} sha={prov['git_sha'][:8]} dirty={dirty} root={root}")
    if dirty:
        run.log("WARNING: dirty working tree -- this run cannot go into the paper (rule 1)")
    return run


def archive_notebook(dest: Path, ipynb: Path | None = None) -> None:
    """The executed notebook, outputs included -> dest/notebook.html (rule 11).

    Without ipynb it asks the open Colab browser tab for the running notebook. A run launched
    through the Colab CLI has no tab. There the notebook is executed headless with
    $PETINTERP_DEFER_ARCHIVE set to a file, so this call only records dest in that file, and
    archive_pending() writes the archives from the executed notebook once it has finished.
    There is no silent fallback: with no tab and no deferral it raises.

    Takes a directory, not a Run: re-starting an old run just to archive it would overwrite its
    provenance.json with the current commit.
    """
    import nbformat
    from nbconvert import HTMLExporter

    pending = os.environ.get("PETINTERP_DEFER_ARCHIVE")
    if ipynb is None and pending:
        with open(pending, "a", encoding="utf-8") as f:
            f.write(f"{dest}\n")
        print(f"archive of {dest} deferred to {pending}")
        return
    if ipynb is not None:
        text = Path(ipynb).read_text(encoding="utf-8")
    else:
        from google.colab import _message
        reply = _message.blocking_request("get_ipynb", timeout_sec=120)
        if reply is None:
            raise RuntimeError("no Colab browser tab answered get_ipynb; a run without one (the "
                               "Colab CLI) sets PETINTERP_DEFER_ARCHIVE or passes ipynb=")
        text = json.dumps(reply["ipynb"])
    # reads(), not from_dict(): Colab sends sources as lists of lines, which nbconvert rejects
    html, _ = HTMLExporter().from_notebook_node(nbformat.reads(text, as_version=4))
    tmp = Path(dest) / "notebook.html.tmp"
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, Path(dest) / "notebook.html")


def archive_pending(pending: Path, ipynb: Path) -> list[Path]:
    """Archive every run directory deferred to `pending` from the executed notebook `ipynb`."""
    dests = list(dict.fromkeys(Path(p) for p in Path(pending).read_text(encoding="utf-8").split("\n") if p))
    for d in dests:
        archive_notebook(d, ipynb=ipynb)
    return dests


def _gpu() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except ImportError:
        return "none"
