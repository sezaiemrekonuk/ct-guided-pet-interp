"""Slice-averaging degradation and the measurement-consistency projection that inverts it.

A thick slice is the mean of k thin slices, as the scanner integrates counts -- never
decimation. The thin volume is cropped to a multiple of k at the high-index end; that cropped
volume is the ground truth every model is scored against.
"""

from __future__ import annotations

import json

import numpy as np
import SimpleITK as sitk

from petinterp.data import volumes as V
from petinterp.runtime import Run, write_json


def crop(hr: np.ndarray, k: int) -> np.ndarray:
    return hr[: hr.shape[0] // k * k]


def avg_k(hr: np.ndarray, k: int) -> np.ndarray:
    """[D, H, W] -> [D/k, H, W]; D must be a multiple of k."""
    return hr.reshape(hr.shape[0] // k, k, *hr.shape[1:]).mean(axis=1)


def consistency(pred: np.ndarray, measured: np.ndarray, k: int) -> np.ndarray:
    """pred <- pred + upsample(measured - avg_k(pred)): every block of k now averages to the
    measured thick slice exactly (CLAUDE.md problem spec, applied to every model)."""
    return pred + np.repeat(measured - avg_k(pred, k), k, axis=0)


def thick_image(lr: np.ndarray, hr_img: sitk.Image, k: int) -> sitk.Image:
    """Geometry of the thick volume: k-times z spacing, first centre at thin index (k-1)/2."""
    img = sitk.GetImageFromArray(lr.astype(np.float32))
    sx, sy, sz = hr_img.GetSpacing()
    img.SetSpacing((sx, sy, sz * k))
    img.SetDirection(hr_img.GetDirection())
    img.SetOrigin(hr_img.TransformContinuousIndexToPhysicalPoint((0, 0, (k - 1) / 2)))
    return img


def degrade(run: Run) -> None:
    cfg = run.cfg
    src, out = run.path(cfg["input"]), run.path(cfg["output"])
    split = json.loads((src / cfg["split_file"]).read_text(encoding="utf-8"))
    cases = sorted(split["train"] + split["val"] + split["test"])
    out.mkdir(parents=True, exist_ok=True)
    m = out / "manifest.json"
    if m.exists() and json.loads(m.read_text(encoding="utf-8"))["config_hash"] != run.cfg_hash:
        raise RuntimeError(f"{out} is frozen under another config; bump the version")

    for k in cfg["k"]:
        (out / f"k{k}").mkdir(exist_ok=True)
        for case in cases:
            dst = out / f"k{k}" / f"{case}.nii.gz"
            if dst.exists():
                continue
            hr_img = V.read(src / case / "pet.nii.gz")
            V.write(thick_image(avg_k(crop(V.array(hr_img), k), k), hr_img, k), dst)
        run.log(f"k={k}: {len(cases)} cases")
    write_json(m, {"config_hash": run.cfg_hash, "config": cfg, "k": cfg["k"],
                   "n_patients": len(cases), "source": cfg["input"]})
