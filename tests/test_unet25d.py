"""Smoke test of the learned-model path on a tiny synthetic cohort, CPU only:
cache -> untrained net == cubic baseline -> train -> resume -> evaluate.

    pip install -e .[train] pytest && pytest tests
"""

import json

import numpy as np
import pytest
import SimpleITK as sitk
import torch
import yaml

import petinterp.data.blocks as B
import petinterp.runtime as rt
from petinterp.data import volumes as V
from petinterp.data.degrade import avg_k, crop, thick_image
from petinterp.evaluation.run import evaluate
from petinterp.models.unet25d import Residual
from petinterp.train import train

CASES = {"train": ["a", "b"], "val": ["c"], "test": ["d"]}


def _cohort(root, rng, k):
    proc = root / "data/processed/v1"
    for case in sum(CASES.values(), []):
        z, y, x = np.mgrid[:21, :170, :180]
        body = ((y - 85) ** 2 / 70 ** 2 + (x - 90) ** 2 / 80 ** 2) < 1
        pet = body * (1 + np.sin(z / 3.0) + rng.random(body.shape)).astype(np.float32)
        ttb = (abs(z - 10) < 3) & ((y - 85) ** 2 + (x - 90) ** 2 < 36)
        pet[ttb] += 20
        ref = sitk.GetImageFromArray(pet.astype(np.float32))
        ref.SetSpacing((4.0, 4.0, 3.0))
        (proc / case).mkdir(parents=True)
        V.write(ref, proc / case / "pet.nii.gz")
        for name, arr in {"ct": np.where(body, 40, -1000).astype(np.int16), "ttb": ttb,
                          "body": body, "organs": np.zeros_like(body)}.items():
            V.write(V.like(arr.astype(np.int16 if name == "ct" else np.uint8), ref), proc / case / f"{name}.nii.gz")
        (root / f"data/degraded/v1/k{k}").mkdir(parents=True, exist_ok=True)
        V.write(thick_image(avg_k(crop(pet, k), k), ref, k), root / f"data/degraded/v1/k{k}/{case}.nii.gz")
    (proc / "split_v1.json").write_text(json.dumps(CASES))


@pytest.mark.parametrize("k", [2, 3, 4])
def test_unet25d_end_to_end(tmp_path, monkeypatch, k):
    _cohort(tmp_path, np.random.default_rng(0), k)
    monkeypatch.setattr(rt, "find_data_root", lambda: tmp_path)
    monkeypatch.setattr(B, "SCRATCH", tmp_path / "scratch")
    cfg = yaml.safe_load((rt.REPO / "configs/p4_unet_k2.yaml").read_text(encoding="utf-8"))
    cfg["exp_id"], cfg["k"] = "smoke", k
    cfg["model"]["features"] = [4, 4, 8, 8, 16, 4]
    cfg["train"].update(patch=176, batch=2, epochs=2, steps_per_epoch=2, workers=0)
    cfg["eval"]["batch"] = 4
    (tmp_path / "smoke.yaml").write_text(yaml.safe_dump(cfg))
    run = rt.start(str(tmp_path / "smoke.yaml"))

    # Untrained (zero last layer) network reproduces the cubic baseline block by block.
    v = {n: np.load(d / f"{n}.npy") for d in B.cache(run, ["d"]) for n in B.CACHED}
    x = np.stack([B.block_input(v, i, k, cfg["model"]) for i in range(len(v["lr"]))])
    with torch.no_grad():
        out = Residual(cfg)(torch.from_numpy(x)).numpy()
    assert x.shape[1] == 5 + (k + 2) + k and v["hr"].shape == (21 // k * k, 176, 180)
    np.testing.assert_array_equal(out.reshape(v["base"].shape), v["base"].astype(np.float32))

    hist = train(run)
    assert len(hist) == 2 and all(np.isfinite(h["val_loss"]) for h in hist)
    assert (run.dir / "checkpoints/best.pt").exists()
    assert train(run) == hist  # resumes at epoch 2 of 2: nothing left to do

    m = evaluate(run)
    assert m["consistency"]["n_patients"] == 1
    assert np.isfinite(m["consistency"]["global"]["psnr"]["mean"])
