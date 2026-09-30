"""evaluate(run): predict every test case, score it raw and after measurement consistency.

One JSON per patient under runs/<exp_id>/per_patient/ (skipped when present, so the stage
resumes per patient), then metrics.json aggregating both variants.
"""

from __future__ import annotations

import json

import numpy as np

from petinterp.data import volumes as V
from petinterp.data.degrade import consistency, crop
from petinterp.evaluation.metrics import aggregate, evaluate_case
from petinterp.models.classical import MODELS
from petinterp.runtime import Run, write_json

VARIANTS = ("raw", "consistency")


def predict_case(run: Run, case: str) -> dict:
    cfg, k = run.cfg, run.cfg["k"]
    proc = run.path(cfg["processed"]) / case
    hr_img = V.read(proc / "pet.nii.gz")
    gt = crop(V.array(hr_img), k)
    lr = V.array(V.read(run.path(cfg["degraded"]) / f"k{k}" / f"{case}.nii.gz"))
    raw = MODELS[cfg["model"]["name"]](lr, k)
    preds = {"raw": raw, "consistency": consistency(raw, lr, k)}

    body = crop(V.array(V.read(proc / "body.nii.gz")), k)
    ttb = crop(V.array(V.read(proc / "ttb.nii.gz")), k)
    bladder = crop(V.array(V.read(proc / "organs.nii.gz")), k) == cfg["bladder_label"]
    spacing_zyx = hr_img.GetSpacing()[::-1]
    return {v: evaluate_case(p, gt, body, ttb, bladder, spacing_zyx) for v, p in preds.items()}


def evaluate(run: Run) -> dict:
    cfg = run.cfg
    split = json.loads((run.path(cfg["processed"]) / cfg["split_file"]).read_text(encoding="utf-8"))
    cases = split[cfg.get("subset", "test")]
    pp_dir = run.dir / "per_patient"
    pp_dir.mkdir(exist_ok=True)

    per = {}
    for i, case in enumerate(cases, 1):
        f = pp_dir / f"{case}.json"
        if not f.exists():
            write_json(f, predict_case(run, case))
        per[case] = json.loads(f.read_text(encoding="utf-8"))
        r = per[case]["consistency"]
        run.log(f"[{i}/{len(cases)}] {case} psnr={r['psnr']:.2f} ssim={r['ssim']:.4f} "
                f"nrmse={r['nrmse']:.4f} lesions={len(r['lesions'])}")

    metrics = {
        "exp_id": cfg["exp_id"], "model": cfg["model"]["name"], "k": cfg["k"],
        "subset": cfg.get("subset", "test"), "config_hash": run.cfg_hash,
        **{v: aggregate({c: per[c][v] for c in cases}) for v in VARIANTS},
        "per_patient": {c: {v: {m: per[c][v][m] for m in ("psnr", "ssim", "nrmse")} for v in VARIANTS}
                        for c in cases},
    }
    write_json(run.dir / "metrics.json", metrics)
    for v in VARIANTS:
        g, les = metrics[v]["global"], metrics[v]["lesions"]
        run.log(f"{v:>11}: PSNR {g['psnr']['mean']:.2f}  SSIM {g['ssim']['mean']:.4f}  "
                f"NRMSE {g['nrmse']['mean']:.4f}  SUVmax bias {les['all']['suvmax_bias'].get('mean', np.nan):+.3f} "
                f"(small {les['small_lt_1ml']['suvmax_bias'].get('mean', np.nan):+.3f}, "
                f"bladder {les['bladder_adjacent_lt_2cm']['suvmax_bias'].get('mean', np.nan):+.3f})")
    return metrics
