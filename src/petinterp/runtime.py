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


def _gpu() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except ImportError:
        return "none"
