"""Smoke test of the learned-model path on a tiny synthetic cohort, CPU only:
cache -> untrained net == cubic baseline -> train -> resume -> evaluate. The CT-FM variant runs
with a small randomly initialised encoder in place of the 77.8M-parameter download; it
has CT-FM's five levels and z depths 16/8/4/2/1 on a 16-slice slab.

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
import petinterp.models.unet25d as U
from monai.networks.nets.segresnet_ds import SegResEncoder
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


@pytest.mark.parametrize("config", ["p4_unet_k2", "p5_unet_ctfm_k2", "p5_unet_ctfmz_k2"])
@pytest.mark.parametrize("k", [2, 3, 4])
def test_unet25d_end_to_end(tmp_path, monkeypatch, k, config):
    _cohort(tmp_path, np.random.default_rng(0), k)
    monkeypatch.setattr(rt, "find_data_root", lambda: tmp_path)
    monkeypatch.setattr(B, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(U, "ctfm_encoder", lambda c: SegResEncoder(init_filters=2, blocks_down=(1, 1, 1, 1, 1)))
    cfg = yaml.safe_load((rt.REPO / f"configs/{config}.yaml").read_text(encoding="utf-8"))
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
        out = U.build(cfg).eval()(torch.from_numpy(x)).numpy()
    assert x.shape[1] == 5 + (16 if "ctfm" in config else k + 2) + k and v["hr"].shape == (21 // k * k, 176, 180)
    np.testing.assert_array_equal(out.reshape(v["base"].shape), v["base"].astype(np.float32))

    hist = train(run)
    assert len(hist) == 2 and all(np.isfinite(h["val_loss"]) for h in hist)
    assert (run.dir / "checkpoints/best.pt").exists()
    assert train(run) == hist  # resumes at epoch 2 of 2: nothing left to do

    m = evaluate(run)
    assert m["consistency"]["n_patients"] == 1
    assert np.isfinite(m["consistency"]["global"]["psnr"]["mean"])


ROWS = {2: [[6, 7, 8, 9], [3, 4, 5], [2], [1], [0]],          # block = slab slices 7..8
        3: [[6, 7, 8, 9, 10], [3, 4, 5], [2, 3], [1], [0]],   # 7..9
        4: [[5, 6, 7, 8, 9, 10], [3, 4, 5], [1, 2, 3], [1], [0]]}  # 6..9


@pytest.mark.parametrize("k", [2, 3, 4])
def test_ctfm_block_rows(monkeypatch, k):
    """z: block keeps the CT-FM rows on the block +- 1 thin slice; z-mean keeps one per level."""
    monkeypatch.setattr(U, "ctfm_encoder", lambda c: SegResEncoder(init_filters=2, blocks_down=(1, 1, 1, 1, 1)))
    cfgs = {}
    for name in ("p5_unet_ctfm", "p5_unet_ctfmz"):
        cfgs[name] = cfg = yaml.safe_load((rt.REPO / f"configs/{name}_k{k}.yaml").read_text(encoding="utf-8"))
        cfg["model"]["features"] = [4, 4, 8, 8, 16, 4]
    mean, block = U.build(cfgs["p5_unet_ctfm"]), U.build(cfgs["p5_unet_ctfmz"])
    assert mean.rows is None and block.rows == ROWS[k]
    chans = [2, 4, 8, 16, 32]                                   # the small encoder's widths
    assert [p.in_channels for p in mean.proj] == chans
    assert [p.in_channels for p in block.proj] == [c * len(r) for c, r in zip(chans, ROWS[k])]

    x = torch.randn(2, U.n_channels(cfgs["p5_unet_ctfmz"]), 32, 32)
    for p in block.proj.parameters():                           # untrained: exactly the base
        assert not p.any()
    torch.testing.assert_close(block.eval()(x), x[:, -k:])
