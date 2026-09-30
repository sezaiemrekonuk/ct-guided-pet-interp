"""The only metric implementations in the repository (rule 6). All inputs are in SUV.

Global metrics are computed inside the body mask only -- air inflates full-FOV numbers.
PSNR and SSIM use data_range = the patient's ground-truth SUVmax inside the body.
NRMSE = ||pred - gt|| / ||gt|| over body voxels.
Lesions are the 26-connected components of the TTB mask; each gets SUVmax/SUVmean bias,
volume, and distance to the bladder mask.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

SMALL_ML = 1.0
BLADDER_MM = 20.0


def psnr(pred, gt, mask, data_range):
    mse = np.mean((pred[mask] - gt[mask]) ** 2)
    return float(10 * np.log10(data_range ** 2 / mse)) if mse > 0 else float("inf")


def ssim(pred, gt, mask, data_range, sigma=1.5):
    """3D Gaussian-window SSIM (Wang et al. 2004), averaged over the mask."""
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    f = lambda x: ndimage.gaussian_filter(x, sigma)  # noqa: E731
    p, g = pred.astype(np.float64), gt.astype(np.float64)
    mp, mg = f(p), f(g)
    vp, vg, cov = f(p * p) - mp ** 2, f(g * g) - mg ** 2, f(p * g) - mp * mg
    s = ((2 * mp * mg + c1) * (2 * cov + c2)) / ((mp ** 2 + mg ** 2 + c1) * (vp + vg + c2))
    return float(s[mask].mean())


def nrmse(pred, gt, mask):
    return float(np.linalg.norm(pred[mask] - gt[mask]) / np.linalg.norm(gt[mask]))


def lesions(pred, gt, ttb, bladder, spacing_zyx):
    lab, n = ndimage.label(ttb, structure=np.ones((3, 3, 3)))
    voxel_ml = float(np.prod(spacing_zyx)) / 1000.0
    dist = (ndimage.distance_transform_edt(~bladder, sampling=spacing_zyx)
            if bladder is not None and bladder.any() else None)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        m = lab[sl] == i
        g, p = gt[sl][m], pred[sl][m]
        out.append({
            "volume_ml": float(m.sum() * voxel_ml),
            "suvmax_gt": float(g.max()), "suvmax_pred": float(p.max()),
            "suvmax_bias": float((p.max() - g.max()) / g.max()),
            "suvmean_bias": float((p.mean() - g.mean()) / g.mean()),
            "bladder_mm": float(dist[sl][m].min()) if dist is not None else None,
        })
    return out


def evaluate_case(pred, gt, body, ttb, bladder, spacing_zyx) -> dict:
    body = body.astype(bool)
    dr = float(gt[body].max())
    return {
        "psnr": psnr(pred, gt, body, dr),
        "ssim": ssim(pred, gt, body, dr),
        "nrmse": nrmse(pred, gt, body),
        "lesions": lesions(pred, gt, ttb.astype(bool),
                           None if bladder is None else bladder.astype(bool), spacing_zyx),
    }


def _stats(x):
    x = np.asarray([v for v in x if v is not None], dtype=float)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "median": float(np.median(x))}


def aggregate(per_patient: dict[str, dict]) -> dict:
    """Patient-level means for global metrics; lesion-level stats per subgroup."""
    les = [l for r in per_patient.values() for l in r["lesions"]]
    groups = {
        "all": les,
        "small_lt_1ml": [l for l in les if l["volume_ml"] < SMALL_ML],
        "bladder_adjacent_lt_2cm": [l for l in les if l["bladder_mm"] is not None
                                    and l["bladder_mm"] < BLADDER_MM],
    }
    return {
        "n_patients": len(per_patient),
        "global": {m: _stats(r[m] for r in per_patient.values()) for m in ("psnr", "ssim", "nrmse")},
        "lesions": {name: {"suvmax_bias": _stats(l["suvmax_bias"] for l in g),
                           "suvmax_abs_bias": _stats(abs(l["suvmax_bias"]) for l in g),
                           "suvmean_bias": _stats(l["suvmean_bias"] for l in g)}
                    for name, g in groups.items()},
    }
