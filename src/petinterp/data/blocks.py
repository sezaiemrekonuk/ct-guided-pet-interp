"""Block-wise 2.5D samples for the learned models: one thick slice in, its k thin slices out.

Thick slice i averages thin slices i*k .. i*k+k-1, so the unit of prediction is that block of k.
The network input stacks, on one 2D in-plane grid:
  PET   thick slices i-r .. i+r                    2r+1 channels, log-SUV
  CT    thin slices  i*k-c .. i*k+k-1+c            k+2c channels, HU -> [-1, 1]  (if use_ct)
        or, with model.ct_slab = S, S thin slices centred on the block (the CT-FM branch)
        or, with model.ctfm.ct_scale = s as well, those S slices of the native-resolution CT
        (data/ct_native, s x s CT pixels per PET pixel, CT-FM's HU window -> [-1, 1]), packed
        space-to-depth: S*s*s channels on the PET grid, unpacked again by the model
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


def volumes(lr_suv: np.ndarray, ct_hu: np.ndarray, k: int, cfg: dict, ctn_hu: np.ndarray | None = None) -> dict:
    """Model-space arrays of one patient: lr [D/k,H,W]; base and ct [D,H,W]; ctn [D,sH,sW]."""
    d = cfg["data"]
    v = {"lr": pet_fwd(lr_suv, d["suv_ref"]),
         "base": pet_fwd(cubic_z(lr_suv, k), d["suv_ref"]),
         "ct": ct_fwd(ct_hu, d["ct_hu"])}
    if ctn_hu is not None:  # native-resolution CT [D, sH, sW], in CT-FM's own window
        v["ctn"] = ct_fwd(ctn_hu, cfg["model"]["ctfm"]["hu"])
    return v


def ct_scale(m: dict) -> int | None:
    """CT pixels per PET pixel in-plane for the native-CT branch; None = CT on the PET grid."""
    return m.get("ctfm", {}).get("ct_scale")


def read_ct_native(run, case: str, k: int) -> np.ndarray | None:
    """Native-resolution CT in HU [D, sH, sW] of one patient, if the config names a ct_native version."""
    if "ct_native" not in run.cfg:
        return None
    return crop(V.array(V.read(run.path(run.cfg["ct_native"]) / f"{case}.nii.gz")), k)


def ct_z(i: int, k: int, m: dict) -> np.ndarray:
    """Thin CT slice indices for thick slice i (unclamped): the block +- ct_context, or ct_slab."""
    if "ct_slab" in m:
        return i * k + (k - m["ct_slab"]) // 2 + np.arange(m["ct_slab"])
    return np.arange(i * k - m["ct_context"], (i + 1) * k + m["ct_context"])


def n_channels(cfg: dict) -> int:
    m, k = cfg["model"], cfg["k"]
    return 2 * m["pet_context"] + 1 + (len(ct_z(0, k, m)) * (ct_scale(m) or 1) ** 2 if m["use_ct"] else 0) + k


def block_input(v: dict, i: int, k: int, m: dict) -> np.ndarray:
    """[n_channels, H, W] input for thick slice i; crop and flip v beforehand (train.Blocks)."""
    n = len(v["lr"])
    parts = [v["lr"][np.clip(np.arange(i - m["pet_context"], i + m["pet_context"] + 1), 0, n - 1)]]
    if m["use_ct"]:
        z, s = np.clip(ct_z(i, k, m), 0, n * k - 1), ct_scale(m)
        if s:  # [S, sH, sW] -> [S*s*s, H, W], the channel order torch's pixel_shuffle undoes
            c = v["ctn"][z]
            h, w = c.shape[1] // s, c.shape[2] // s
            parts.append(c.reshape(len(z), h, s, w, s).transpose(0, 2, 4, 1, 3).reshape(-1, h, w))
        else:
            parts.append(v["ct"][z])
    parts.append(v["base"][i * k:(i + 1) * k])
    return np.concatenate(parts).astype(np.float32)


def _pad(a: np.ndarray, p: int, fill: float, s: int = 1) -> np.ndarray:
    """Centre-pad in-plane to at least p x p PET pixels (the smallest cohort grid is 128 x 128);
    an array with s x s pixels per PET pixel gets s times the pad, so it stays aligned."""
    ph, pw = max(p - a.shape[1] // s, 0), max(p - a.shape[2] // s, 0)
    return np.pad(a, ((0, 0), (s * (ph // 2), s * (ph - ph // 2)), (s * (pw // 2), s * (pw - pw // 2))),
                  constant_values=fill)


def cache(run, cases: list[str]) -> list[Path]:
    """Write model-space .npy files per patient to local scratch; returns their directories.

    A splittable stage (rule 8): a finished patient ('done' marker, written last) is skipped and
    every file is written atomically. The directory is keyed by everything that shapes its
    contents, so a config change can never read a stale cache.
    """
    cfg, k = run.cfg, run.cfg["k"]
    key = [cfg["processed"], cfg["degraded"], k, cfg["data"], cfg["train"]["patch"]]
    if "ct_native" in cfg:  # appended only here, so the p4/p5 caches keep their keys
        key += [cfg["ct_native"], cfg["model"]["ctfm"]["hu"]]
    key = json.dumps(key)
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
        # ponytail: whole-volume float32 in RAM; a 512-grid patient at ct_scale 4 needs ~25 GB
        # (fine on Colab's A100 high-RAM runtime), write ctn in z chunks if a smaller runtime OOMs
        arrs = volumes(lr, crop(V.array(V.read(proc / "ct.nii.gz")), k), k, cfg, read_ct_native(run, case, k))
        arrs["hr"] = pet_fwd(crop(V.array(V.read(proc / "pet.nii.gz")), k), cfg["data"]["suv_ref"])
        arrs["ttb"] = crop(V.array(V.read(proc / "ttb.nii.gz")), k)
        s = ct_scale(cfg["model"]) or 1
        if "ctn" in arrs and arrs["ctn"].shape != (len(arrs["hr"]), s * arrs["hr"].shape[1], s * arrs["hr"].shape[2]):
            raise RuntimeError(f"{case}: ct_native {arrs['ctn'].shape} is not ct_scale={s} x the PET grid "
                               f"{arrs['hr'].shape}; the config and the data version disagree")
        for name, a in arrs.items():
            tmp = d / f"{name}.tmp.npy"
            np.save(tmp, _pad(a, cfg["train"]["patch"], -1 if name.startswith("ct") else 0,
                              s if name == "ctn" else 1).astype(CACHED.get(name, np.float16)))
            os.replace(tmp, d / f"{name}.npy")
        (d / "done").write_text("")
        run.log(f"cache [{n}/{len(cases)}] {case}")
    return dirs

