"""Block-wise 2.5D samples for the learned models: one thick slice in, its k thin slices out.

Thick slice i averages thin slices i*k .. i*k+k-1, so the unit of prediction is that block of k.
The network input stacks, on one 2D in-plane grid:
  PET   thick slices i-r .. i+r                    2r+1 channels, log-SUV
  CT    thin slices  i*k-c .. i*k+k-1+c            k+2c channels, HU -> [-1, 1]  (if use_ct)
  base  cubic_z prediction for the k target slices k channels, log-SUV         (always last)
and the network predicts a residual on base. Indices past either end of the volume are clamped,
so edge slices repeat.

Training reads per-patient .npy files from local scratch through mmap: Drive's FUSE mount is far
too slow for random slice access, and the training set does not fit in Colab RAM as float32.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from petinterp.data import volumes as V
from petinterp.data.degrade import crop
from petinterp.models.classical import cubic_z

SCRATCH = Path(os.environ.get("PETINTERP_SCRATCH", tempfile.gettempdir())) / "petinterp"
CACHED = {"hr": np.float16, "base": np.float16, "lr": np.float16, "ct": np.float16, "ttb": np.uint8}


def pet_fwd(suv: np.ndarray, suv_ref: float) -> np.ndarray:
    """log(1+SUV)/log(1+suv_ref): absolute, never per-patient (CLAUDE.md domain guardrails)."""
    return (np.log1p(np.clip(suv, 0, None)) / np.log1p(suv_ref)).astype(np.float32)


def pet_inv(t: np.ndarray, suv_ref: float) -> np.ndarray:
    return np.expm1(t * np.log1p(suv_ref))


def ct_fwd(hu: np.ndarray, hu_range) -> np.ndarray:
    lo, hi = hu_range
    return (2 * (np.clip(hu, lo, hi) - lo) / (hi - lo) - 1).astype(np.float32)


def volumes(lr_suv: np.ndarray, ct_hu: np.ndarray, k: int, cfg: dict) -> dict:
    """Model-space arrays of one patient: lr [D/k,H,W]; base and ct [D,H,W]."""
    d = cfg["data"]
    return {"lr": pet_fwd(lr_suv, d["suv_ref"]),
            "base": pet_fwd(cubic_z(lr_suv, k), d["suv_ref"]),
            "ct": ct_fwd(ct_hu, d["ct_hu"])}


def n_channels(cfg: dict) -> int:
    m, k = cfg["model"], cfg["k"]
    return 2 * m["pet_context"] + 1 + (k + 2 * m["ct_context"] if m["use_ct"] else 0) + k


def block_input(v: dict, i: int, k: int, m: dict, yx=(slice(None), slice(None))) -> np.ndarray:
    """[n_channels, H, W] input for thick slice i, cropped in-plane to yx."""
    n = len(v["lr"])
    parts = [v["lr"][(np.clip(np.arange(i - m["pet_context"], i + m["pet_context"] + 1), 0, n - 1), *yx)]]
    if m["use_ct"]:
        z = np.arange(i * k - m["ct_context"], (i + 1) * k + m["ct_context"])
        parts.append(v["ct"][(np.clip(z, 0, n * k - 1), *yx)])
    parts.append(v["base"][(slice(i * k, (i + 1) * k), *yx)])
    return np.concatenate(parts).astype(np.float32)


def _pad(a: np.ndarray, p: int, fill: float) -> np.ndarray:
    """Centre-pad in-plane to at least p x p (the smallest cohort grid is 128 x 128)."""
    ph, pw = max(p - a.shape[1], 0), max(p - a.shape[2], 0)
    return np.pad(a, ((0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)), constant_values=fill)


def cache(run, cases: list[str]) -> list[Path]:
    """Write model-space .npy files per patient to local scratch; returns their directories.

    A splittable stage (rule 8): a finished patient ('done' marker, written last) is skipped and
    every file is written atomically. The directory is keyed by everything that shapes its
    contents, so a config change can never read a stale cache.
    """
    cfg, k = run.cfg, run.cfg["k"]
    key = json.dumps([cfg["processed"], cfg["degraded"], k, cfg["data"], cfg["train"]["patch"]])
    root = SCRATCH / f"cache_{hashlib.sha256(key.encode()).hexdigest()[:12]}"
    dirs = []
    for n, case in enumerate(cases, 1):
        d = root / case
        dirs.append(d)
        if (d / "done").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        proc = run.path(cfg["processed"]) / case
        lr = V.array(V.read(run.path(cfg["degraded"]) / f"k{k}" / f"{case}.nii.gz"))
        arrs = volumes(lr, crop(V.array(V.read(proc / "ct.nii.gz")), k), k, cfg)
        arrs["hr"] = pet_fwd(crop(V.array(V.read(proc / "pet.nii.gz")), k), cfg["data"]["suv_ref"])
        arrs["ttb"] = crop(V.array(V.read(proc / "ttb.nii.gz")), k)
        for name, a in arrs.items():
            tmp = d / f"{name}.tmp.npy"
            np.save(tmp, _pad(a, cfg["train"]["patch"], -1 if name == "ct" else 0).astype(CACHED[name]))
            os.replace(tmp, d / f"{name}.npy")
        (d / "done").write_text("")
        run.log(f"cache [{n}/{len(cases)}] {case}")
    return dirs

