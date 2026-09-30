"""DEEP-PSMA archives -> data/processed/<version>/<case>/, then the frozen patient-level split.

Per case (one tracer): PET in SUV as shipped, CT resampled onto the PET grid (never the
reverse), TTB lesion mask, TotalSegmentator labels on the PET grid, a CT body mask, and qc.json.
qc.json is written last, so its existence marks a finished case and the stage resumes per
patient. Cases are read straight out of the zips, so nothing is ever unzipped onto Drive.
"""

from __future__ import annotations

import datetime as dt
import json
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from petinterp.data import volumes as V
from petinterp.runtime import REPO, Run, write_json

FILES = ("PET.nii.gz", "CT.nii.gz", "TTB.nii.gz", "totseg_24.nii.gz")


def _cases(archives: Path, tracer: str) -> dict[str, tuple[Path, str]]:
    """case id -> (zip path, member prefix) for every case that ships this tracer."""
    out = {}
    for zp in sorted(archives.glob("*.zip")):
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                parts = n.split("/")
                if len(parts) == 3 and parts[1] == tracer and parts[2] == "PET.nii.gz":
                    out[parts[0]] = (zp, f"{parts[0]}/{tracer}/")
    return out


def _onto(img: sitk.Image, ref: sitk.Image, interp, default: float) -> sitk.Image:
    return sitk.Resample(img, ref, sitk.Transform(), interp, default, img.GetPixelID())


def body_mask(ct_hu: np.ndarray, threshold: float) -> np.ndarray:
    """Largest connected component above threshold HU, holes filled slice by slice."""
    m = ct_hu > threshold
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    m = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    return np.stack([ndimage.binary_fill_holes(s) for s in m])


def process_case(case: str, zp: Path, prefix: str, cfg: dict, out: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(zp) as z:
        img = {}
        for f in FILES:
            z.extract(prefix + f, tmp)
            img[f] = V.read(Path(tmp) / prefix / f)

    pet = sitk.Cast(img["PET.nii.gz"], sitk.sitkFloat32)
    suv = V.array(pet)
    qc = {"case": case, "size_xyz": pet.GetSize(), "spacing_mm": pet.GetSpacing(),
          "direction": [round(d, 3) for d in pet.GetDirection()], "reasons": []}

    # QC gate (CLAUDE.md domain guardrails)
    if not np.allclose(pet.GetDirection(), img["CT.nii.gz"].GetDirection(), atol=1e-3):
        qc["reasons"].append("PET/CT direction mismatch")
    if not np.isfinite(suv).all():
        qc["reasons"].append("non-finite SUV")
    if suv.min() < 0:
        qc["reasons"].append(f"negative SUV (min {suv.min():.3g})")
    if suv.shape[0] < cfg["qc"]["min_slices"]:
        qc["reasons"].append(f"{suv.shape[0]} slices < {cfg['qc']['min_slices']}")
    qc["included"] = not qc["reasons"]
    if not qc["included"]:
        return qc

    ct = _onto(img["CT.nii.gz"], pet, sitk.sitkLinear, -1000.0)
    ct_hu = V.array(ct).astype(np.int16)
    ttb = V.array(_onto(img["TTB.nii.gz"], pet, sitk.sitkNearestNeighbor, 0)) > 0
    organs = V.array(_onto(img["totseg_24.nii.gz"], pet, sitk.sitkNearestNeighbor, 0)).astype(np.uint8)
    body = body_mask(ct_hu, cfg["body_mask_hu"])

    voxel_ml = float(np.prod(pet.GetSpacing())) / 1000.0
    means = {name: float(suv[organs == lid].mean()) if (organs == lid).any() else None
             for name, lid in cfg["labels"].items()}
    lo, hi = cfg["qc"]["liver_suvmean_range"]
    liver = means.get("liver")
    qc.update({
        "n_slices": int(suv.shape[0]),
        "ttb_ml": float(ttb.sum() * voxel_ml),
        "suv_max": float(suv.max()),
        "organ_suvmean": means,
        # Flags, not exclusions: the liver range itself is under review (log 2026-08-26).
        "flag_liver_out_of_range": liver is None or not lo <= liver <= hi,
        "flag_psma_pattern": liver is None or not all(
            means.get(k) is not None and means[k] > liver
            for k in ("kidney_left", "kidney_right", "urinary_bladder")),
    })

    d = out / case
    d.mkdir(parents=True, exist_ok=True)
    V.write(pet, d / "pet.nii.gz")
    V.write(V.like(ct_hu, pet), d / "ct.nii.gz")
    V.write(V.like(ttb.astype(np.uint8), pet), d / "ttb.nii.gz")
    V.write(V.like(organs, pet), d / "organs.nii.gz")
    V.write(V.like(body.astype(np.uint8), pet), d / "body.nii.gz")
    return qc


def _freeze_guard(out: Path, run: Run) -> None:
    """A version directory belongs to one config forever (rule 7)."""
    m = out / "manifest.json"
    if m.exists():
        prev = json.loads(m.read_text(encoding="utf-8"))
        if prev["config_hash"] != run.cfg_hash:
            raise RuntimeError(f"{out} was written by config hash {prev['config_hash']}, this config "
                               f"is {run.cfg_hash}. Frozen: bump the version instead of overwriting.")
    else:
        write_json(m, {"status": "in_progress", "config_hash": run.cfg_hash, "config": run.cfg})


def preprocess(run: Run) -> list[dict]:
    cfg = run.cfg
    out = run.path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    _freeze_guard(out, run)

    cases = _cases(run.path(cfg["source"]["archives"]), cfg["source"]["tracer"])
    run.log(f"{len(cases)} {cfg['source']['tracer']} cases in archives")
    qcs = []
    for i, (case, (zp, prefix)) in enumerate(sorted(cases.items()), 1):
        qc_path = out / case / "qc.json"
        if qc_path.exists():
            qcs.append(json.loads(qc_path.read_text(encoding="utf-8")))
            continue
        qc = process_case(case, zp, prefix, cfg, out)
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(qc_path, qc)
        qcs.append(qc)
        flags = [k for k in qc if k.startswith("flag_") and qc[k]]
        run.log(f"[{i}/{len(cases)}] {case} included={qc['included']} {qc['reasons'] or ''} "
                f"liver={(qc.get('organ_suvmean') or {}).get('liver')} {flags or ''}")

    # Orientation consistency across the cohort: the model works in array index space, so a
    # case whose axes are flipped relative to the rest would be silently mirrored.
    ref = Counter(tuple(q["direction"]) for q in qcs if q["included"]).most_common(1)[0][0]
    for q in qcs:
        if q["included"] and tuple(q["direction"]) != ref:
            q["included"] = False
            q["reasons"].append(f"direction {q['direction']} differs from cohort {list(ref)}")

    excluded = {q["case"]: q["reasons"] for q in qcs if not q["included"]}
    write_json(out / "manifest.json", {
        "status": "complete", "config_hash": run.cfg_hash, "config": run.cfg,
        "date": dt.date.today().isoformat(),
        "n_patients": sum(q["included"] for q in qcs), "excluded": excluded,
        "flagged_liver": [q["case"] for q in qcs if q.get("flag_liver_out_of_range")],
        "flagged_psma_pattern": [q["case"] for q in qcs if q.get("flag_psma_pattern")],
    })
    write_json(run.dir / "qc_summary.json", qcs)
    run.log(f"done: {sum(q['included'] for q in qcs)} included, {len(excluded)} excluded {excluded}")
    return qcs


def make_split(run: Run, qcs: list[dict]) -> dict:
    """Patient-level 70/10/20, stratified by TTB-volume tertile, frozen once written.

    Every DEEP-PSMA case has disease, so "stratified by lesion presence" is vacuous here;
    tertiles of lesion volume keep the burden distribution balanced across the three sets.
    """
    s = run.cfg["split"]
    inc = sorted((q for q in qcs if q["included"]), key=lambda q: q["case"])
    vols = np.array([q["ttb_ml"] for q in inc])
    edges = np.quantile(vols, np.linspace(0, 1, s["strata"] + 1)[1:-1])
    stratum = np.searchsorted(edges, vols, side="right")
    rng = np.random.default_rng(s["seed"])
    split = {"train": [], "val": [], "test": []}
    for k in range(s["strata"]):
        ids = [q["case"] for q, st in zip(inc, stratum) if st == k]
        rng.shuffle(ids)
        n_tr, n_va = round(len(ids) * s["fractions"][0]), round(len(ids) * s["fractions"][1])
        split["train"] += ids[:n_tr]
        split["val"] += ids[n_tr:n_tr + n_va]
        split["test"] += ids[n_tr + n_va:]
    split = {k: sorted(v) for k, v in split.items()}
    doc = {"seed": s["seed"], "source": run.cfg["output"], "config_hash": run.cfg_hash,
           "stratified_by": f"TTB volume, {s['strata']} quantile strata", **split}

    # Immutable: an existing split file (committed or on Drive) is verified, never overwritten.
    for path in (REPO / s["file"], run.path(run.cfg["output"]) / Path(s["file"]).name):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            prev = json.loads(path.read_text(encoding="utf-8"))
            if {k: prev[k] for k in split} != split:
                raise RuntimeError(f"{path} exists with a different split; write split_v2 instead")
        else:
            write_json(path, doc)
    run.log(f"split: {', '.join(f'{k}={len(v)}' for k, v in split.items())}")
    return doc
