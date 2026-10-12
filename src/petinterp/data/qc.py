"""SUV conversion check on DEEP-PSMA: is a low PSMA liver SUVmean the patient, or the conversion?

40/100 PSMA livers fall outside the Ga-68 range of 4-8 (p1_preprocess_deep_psma), so the liver
cannot be trusted as the calibration reference here (log 2026-08-26). Model per scan:
log(organ SUV) = log(shared factor) + log(organ-specific uptake). A dose or decay error is one
shared factor; so is anything else that moves every organ of a scan together (tumour sink on the
blood pool, uptake time). Each tracer of each case is measured identically, straight from the zips:

1. Paired reference tracer (FDG). Same patient, separate scan: a liver inside its range rules out
   patient-level errors (weight units), not the PSMA scan's own dose or decay correction.
2. Slope of log(reference organ) on log(liver) across all patients: the fraction of liver
   variance shared with that organ. Near 0, the liver varies on its own and calibration cannot
   explain it. Near 1 means shared by the organs, not necessarily by the conversion, so it is
   also reported with log tumour volume regressed out, which removes a sink acting on all organs
   insofar as it is linear in log TTB.
   A bias common to every scan has no spread across patients and is invisible here; only the
   paired FDG and the absolute ranges see it.
3. For the below-range patients, each reference organ's drop against the in-range patients
   estimates their shared-factor offset (its own uptake is independent of how the liver was
   selected); set it against the liver's drop.
4. Tumour sink: liver and reference organs against TTB volume.

Organ means exclude TTB voxels; liver and aorta also get a core eroded by `core_erosion_mm`
against partial volume. One JSON per patient under runs/<exp_id>/per_patient/, recomputed when a
measurement setting or ROW_SCHEMA changes. A case that cannot be read is excluded and logged, and
retried on the next run rather than cached.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage, stats

from petinterp.data import volumes as V
from petinterp.data.preprocess import _cases, _onto
from petinterp.runtime import Run, write_json

ROW_SCHEMA = 2                        # bump when organ_stats changes what a row holds
MEASUREMENT_KEYS = ("source", "test_tracer", "reference_tracer", "labels", "core_erosion_mm")


def _mean(suv: np.ndarray, mask: np.ndarray) -> float | None:
    return float(suv[mask].mean()) if mask.any() else None


def organ_stats(zp: Path, prefix: str, cfg: dict) -> dict:
    """One tracer of one case: lesion-free organ SUVmeans, eroded cores, tumour volume."""
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(zp) as z:
        for f in ("PET.nii.gz", "totseg_24.nii.gz", "TTB.nii.gz"):
            z.extract(prefix + f, tmp)
        d = Path(tmp) / prefix
        pet = V.read(d / "PET.nii.gz")
        organs = V.array(_onto(V.read(d / "totseg_24.nii.gz"), pet, sitk.sitkNearestNeighbor, 0))
        ttb = V.array(_onto(V.read(d / "TTB.nii.gz"), pet, sitk.sitkNearestNeighbor, 0)) > 0
    suv = V.array(pet)
    spacing_zyx = pet.GetSpacing()[::-1]
    masks = {name: np.isin(organs, ids) & ~ttb for name, ids in cfg["labels"].items()}
    return {
        "organ_suvmean": {name: _mean(suv, m) for name, m in masks.items()},
        "liver_with_lesions": _mean(suv, np.isin(organs, cfg["labels"]["liver"])),
        "core_suvmean": {name: _mean(suv, ndimage.distance_transform_edt(masks[name], sampling=spacing_zyx) > mm)
                         for name, mm in cfg["core_erosion_mm"].items()},
        "ttb_ml": float(ttb.sum() * np.prod(spacing_zyx) / 1000.0),
    }


def measurement_key(cfg: dict) -> str:
    """Hash of what decides a per-patient row; summary-only settings stay out of it."""
    return hashlib.sha256(json.dumps({"schema": ROW_SCHEMA, **{k: cfg[k] for k in MEASUREMENT_KEYS}},
                                     sort_keys=True).encode()).hexdigest()[:16]


def _values(rows: list[dict], tracer: str, field: str, name: str | None = None) -> np.ndarray:
    """One measurement across patients, None -> NaN."""
    v = [r[tracer][field] if name is None else r[tracer][field][name] for r in rows]
    return np.array([np.nan if x is None else x for x in v], dtype=float)


def _finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def _quantiles(x: np.ndarray, qs: list[int]) -> dict:
    a = _finite(x)
    return {"n": int(a.size), **{f"q{q}": float(np.percentile(a, q)) if a.size else None for q in qs}}


def _median_ratio(a: np.ndarray, b: np.ndarray) -> float | None:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = _finite(a / b)
    return float(np.median(r)) if r.size else None


def _spearman(x: np.ndarray, y: np.ndarray) -> dict:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return {"rho": None, "p": None, "n": int(ok.sum())}
    r, p = stats.spearmanr(x[ok], y[ok])
    return {"rho": float(r), "p": float(p), "n": int(ok.sum())}


def _boot(stat, groups: list[np.ndarray], cfg: dict) -> dict:
    """Point estimate and percentile 95% CI, resampling patients within each group."""
    n = [int(g.shape[0]) for g in groups]
    if min(n) < cfg["min_group"]:
        return {"value": None, "ci95": None, "n": n}
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    draws = [stat(*(g[rng.integers(0, g.shape[0], g.shape[0])] for g in groups))
             for _ in range(cfg["bootstrap"]["n"])]
    return {"value": float(stat(*groups)), "n": n,
            "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def _slope(m: np.ndarray) -> float:
    """OLS slope of column 1 on column 0."""
    return float(np.polyfit(m[:, 0], m[:, 1], 1)[0])


def _slope_given(m: np.ndarray) -> float:
    """Slope of column 1 on column 0 with column 2 regressed out of both."""
    z = np.column_stack([np.ones(len(m)), m[:, 2]])
    res = m[:, :2] - z @ np.linalg.lstsq(z, m[:, :2], rcond=None)[0]
    return _slope(res)


def _drop(below: np.ndarray, inside: np.ndarray) -> float:
    """Fractional change of the median, below-range patients against in-range ones (log input)."""
    return float(np.exp(np.median(below) - np.median(inside)) - 1)


def _rows_where(m: np.ndarray, sel: np.ndarray) -> np.ndarray:
    m = m[sel]
    return m[np.isfinite(m).all(axis=1)]


def summarise(rows: list[dict], cfg: dict) -> dict:
    """Every number the decision rests on, from the per-patient rows alone."""
    excluded = {r["case"]: r["excluded"] for r in rows if "excluded" in r}
    rows = [r for r in rows if "excluded" not in r]
    t, ref = cfg["test_tracer"], cfg["reference_tracer"]
    ranges, qs = cfg["liver_suvmean_range"], cfg["quantiles"]

    status = {}                                       # per tracer: below / in / above / missing
    for tr in (t, ref):
        liv, (lo, hi) = _values(rows, tr, "organ_suvmean", "liver"), ranges[tr]
        status[tr] = np.where(np.isnan(liv), "missing", np.where(liv < lo, "below", np.where(liv > hi, "above", "in")))
    groups = ("below", "in", "above", "missing")
    crosstab = {f"{t}_{a}__{ref}_{b}": int(((status[t] == a) & (status[ref] == b)).sum())
                for a in groups for b in groups}

    with np.errstate(divide="ignore", invalid="ignore"):
        log = {k: np.log(_values(rows, t, "organ_suvmean", k)) for k in cfg["labels"]}
    ttb = _values(rows, t, "ttb_ml")
    log_ttb = np.log1p(ttb)
    below, inside = status[t] == "below", status[t] == "in"
    everyone = np.ones(len(rows), bool)
    scale = {}
    for k in cfg["reference_organs"]:
        m = np.column_stack([log["liver"], log[k], log_ttb])
        b, i = _rows_where(m, below), _rows_where(m, inside)
        scale[k] = {
            "slope_on_liver": _boot(_slope, [_rows_where(m, everyone)[:, :2]], cfg),
            "slope_on_liver_given_ttb": _boot(_slope_given, [_rows_where(m, everyone)], cfg),
            "drop_below_vs_in": _boot(lambda bb, ii: _drop(bb[:, 1], ii[:, 1]), [b, i], cfg),
            "liver_drop_same_patients": _boot(lambda bb, ii: _drop(bb[:, 0], ii[:, 0]), [b, i], cfg),
            "vs_ttb_ml": _spearman(log[k], ttb),
        }

    liver_t = _values(rows, t, "organ_suvmean", "liver")
    core = _values(rows, t, "core_suvmean", "liver")
    bin_suv = cfg["histogram_bin_suv"]
    edges = (np.arange(0, np.nanmax(liver_t) + 2 * bin_suv, bin_suv) if np.isfinite(liver_t).any()
             else np.array([]))
    return {
        "measurement_key": measurement_key(cfg),
        "n_cases": len(rows), "excluded": excluded, "liver_range": ranges, "crosstab": crosstab,
        "liver_suvmean": {tr: _quantiles(_values(rows, tr, "organ_suvmean", "liver"), qs) for tr in (t, ref)},
        f"{ref}_liver_in_{t}_below_range": _quantiles(_values(rows, ref, "organ_suvmean", "liver")[below], qs),
        f"{t}_vs_{ref}_liver": _spearman(liver_t, _values(rows, ref, "organ_suvmean", "liver")),
        "scale": scale,
        "tumour_sink": {"liver_vs_ttb_ml": _spearman(liver_t, ttb)},
        "partial_volume": {"whole_vs_core_liver": _spearman(liver_t, core),
                           "median_core_over_whole": _median_ratio(core, liver_t),
                           "n_core_missing": int(np.isnan(core).sum()),
                           "n_core_below_range": int((core < ranges[t][0]).sum()),
                           "median_with_lesions_over_lesion_free":
                               _median_ratio(_values(rows, t, "liver_with_lesions"), liver_t)},
        "liver_histogram": {"edges": edges.tolist(),
                            "counts": np.histogram(_finite(liver_t), bins=edges)[0].tolist() if edges.size else []},
    }


def _json_safe(x):
    """NaN/inf -> None, so summary.json stays strict JSON."""
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    return None if isinstance(x, float) and not np.isfinite(x) else x


def suv_check(run: Run) -> dict:
    cfg = run.cfg
    tracers = (cfg["test_tracer"], cfg["reference_tracer"])
    archives = run.path(cfg["source"]["archives"])
    by_tracer = {t: _cases(archives, t) for t in tracers}
    cases = sorted(set.intersection(*(set(c) for c in by_tracer.values())))
    run.log(f"{len(cases)} cases ship both {tracers}")

    key = measurement_key(cfg)
    out = run.dir / "per_patient"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, case in enumerate(cases, 1):
        path = out / f"{case}.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("key") == key:
                rows.append(row)
                continue
        try:
            row = {"case": case, "key": key, **{t: organ_stats(*by_tracer[t][case], cfg) for t in tracers}}
        except (KeyError, RuntimeError, OSError, zipfile.BadZipFile) as e:   # missing member, bad file, Drive
            rows.append({"case": case, "excluded": f"{type(e).__name__}: {e}"})
            run.log(f"[{i}/{len(cases)}] {case} excluded: {rows[-1]['excluded']}")
            continue
        write_json(path, _json_safe(row))
        rows.append(row)
        run.log(f"[{i}/{len(cases)}] {case} " + "  ".join(
            f"{t} liver={row[t]['organ_suvmean']['liver']}" for t in tracers))

    summary = _json_safe(summarise(rows, cfg))
    write_json(run.dir / "summary.json", summary)
    run.log(f"done: {len(summary['excluded'])} excluded, crosstab {summary['crosstab']}")
    return summary
