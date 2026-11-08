"""Smoke test of the learned-model path on a tiny synthetic cohort, CPU only:
cache -> untrained net == cubic baseline -> train -> resume -> evaluate. The CT-FM variant runs
with a small randomly initialised encoder in place of the 77.8M-parameter download; it
has CT-FM's five levels and z depths 16/8/4/2/1 on a 16-slice slab. The native-CT variant
(p5_unet_ctfm*_native) gets its data/ct_native/v1 from the real ct_native stage, run on a
synthetic archive whose CT is a linear ramp in HU, so pixel alignment can be checked exactly.

    pip install -e .[train] pytest && pytest tests
"""

import json
import zipfile

import numpy as np
import pytest
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import yaml

import petinterp.data.blocks as B
import petinterp.runtime as rt
from petinterp.data import volumes as V
from petinterp.data.degrade import avg_k, crop, thick_image
from petinterp.data.preprocess import _onto, ct_native
from petinterp.evaluation.run import evaluate
import petinterp.models.unet25d as U
from monai.networks.nets.segresnet_ds import SegResEncoder
from petinterp.train import Blocks, train

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


def _ramp(p):
    """HU linear in physical position (mm): linear interpolation reproduces it exactly, and a shift
    of one native CT pixel (1 mm) moves it by 6 to 8 HU."""
    return 8 * (p[0] - 360) + 6 * (p[1] - 340) + 5 * (p[2] - 30)


def _native(root):
    """Archive with a ramp CT at 2 mm per case, the processed CT re-made from it on the PET grid as
    preprocess does, then the real ct_native stage -> data/ct_native/v1."""
    proc, raw = root / "data/processed/v1", root / "data/raw/deep_psma"
    raw.mkdir(parents=True)
    z, y, x = np.mgrid[:34, :346, :366] * 2.0 - 4  # covers the PET's 0..720 x 0..680 x 0..60 mm
    ct = sitk.GetImageFromArray(_ramp((x, y, z)).astype(np.int16))
    ct.SetSpacing((2.0, 2.0, 2.0))
    ct.SetOrigin((-4.0, -4.0, -4.0))
    V.write(ct, root / "ct.nii.gz")
    with zipfile.ZipFile(raw / "cases.zip", "w") as zf:
        for case in sum(CASES.values(), []):
            zf.write(root / "ct.nii.gz", f"{case}/PSMA/CT.nii.gz")
            zf.write(proc / case / "pet.nii.gz", f"{case}/PSMA/PET.nii.gz")
            V.write(_onto(ct, V.read(proc / case / "pet.nii.gz"), sitk.sitkLinear, -1000.0), proc / case / "ct.nii.gz")
    return ct_native(rt.start("configs/p5_ct_native_v1.yaml"))


def _smoke(tmp_path, config, k):
    cfg = yaml.safe_load((rt.REPO / f"configs/{config}.yaml").read_text(encoding="utf-8"))
    cfg["exp_id"], cfg["k"] = f"smoke_{config}", k
    cfg["model"]["features"] = [4, 4, 8, 8, 16, 4]
    cfg["train"].update(patch=176, batch=2, epochs=2, steps_per_epoch=2, workers=0)
    cfg["eval"]["batch"] = 4
    (tmp_path / f"{config}.yaml").write_text(yaml.safe_dump(cfg))
    return rt.start(str(tmp_path / f"{config}.yaml"))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "find_data_root", lambda: tmp_path)
    monkeypatch.setattr(B, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(U, "ctfm_encoder", lambda c: SegResEncoder(init_filters=2, blocks_down=(1, 1, 1, 1, 1)))
    return tmp_path


@pytest.mark.parametrize("config", ["p4_unet_k2", "p5_unet_ctfm_k2", "p5_unet_ctfmz_k2",
                                    "p5_unet_ctfm_native_k2", "p5_unet_ctfmz_native_k2"])
@pytest.mark.parametrize("k", [2, 3, 4])
def test_unet25d_end_to_end(env, k, config):
    tmp_path = env
    _cohort(tmp_path, np.random.default_rng(0), k)
    if "native" in config:
        _native(tmp_path)
    run = _smoke(tmp_path, config, k)
    cfg = run.cfg

    # Untrained (zero last layer) network reproduces the cubic baseline block by block.
    v = {f.stem: np.load(f) for d in B.cache(run, ["d"]) for f in d.glob("*.npy")}
    x = np.stack([B.block_input(v, i, k, cfg["model"]) for i in range(len(v["lr"]))])
    with torch.no_grad():
        out = U.build(cfg).eval()(torch.from_numpy(x)).numpy()
    n_ct = 16 * 16 if "native" in config else 16 if "ctfm" in config else k + 2
    assert x.shape[1] == 5 + n_ct + k == U.n_channels(cfg) and v["hr"].shape == (21 // k * k, 176, 180)
    if "native" in config:  # 170 rows padded to 176 PET pixels -> 704 CT pixels, 3 * 4 each side
        assert v["ctn"].shape == (21 // k * k, 704, 720)
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


def test_ct_native_alignment(env):
    """PET pixel (y, x) is exactly native CT pixels 4y..4y+3, 4x..4x+3: on disk, and in what CT-FM
    sees after the random crop, L-R flip, padding and space-to-depth packing of training. With a
    linear ramp, the native CT averaged over each PET pixel must equal the CT on the PET grid."""
    tmp_path, k = env, 2
    _cohort(tmp_path, np.random.default_rng(0), k)
    doc = _native(tmp_path)
    assert doc["n_patients"] == 4 and doc["factor"] == 4 and doc["mae_hu_vs_processed_in_body"]["max"] < 1.5

    pet = V.read(tmp_path / "data/processed/v1/a/pet.nii.gz")
    img = V.read(tmp_path / "data/ct_native/v1/a.nii.gz")
    assert img.GetSize() == (720, 680, 21) and img.GetSpacing() == (1.0, 1.0, 3.0)
    for ix, iy in ((0, 0), (3, 2), (717, 679)):  # fine pixel 4x+i sits at PET index x + (i + 0.5)/4 - 0.5
        ijk = (ix // 4 + (ix % 4 + 0.5) / 4 - 0.5, iy // 4 + (iy % 4 + 0.5) / 4 - 0.5, 5)
        np.testing.assert_allclose(img.TransformIndexToPhysicalPoint((ix, iy, 5)),
                                   pet.TransformContinuousIndexToPhysicalPoint(ijk))
    native = V.array(img).astype(np.float64)
    coarse = V.array(V.read(tmp_path / "data/processed/v1/a/ct.nii.gz")).astype(np.float64)
    inner = np.abs(coarse) < 900                                # away from CT-FM's clip at -1024
    back = native.reshape(21, 170, 4, 180, 4).mean(axis=(2, 4))
    np.testing.assert_allclose(back[inner], coarse[inner], atol=1.5)

    seen = {}
    for config in ("p5_unet_ctfm_native_k2", "p5_unet_ctfm_k2"):
        run = _smoke(tmp_path, config, k)
        ds = Blocks(B.cache(run, ["a", "b"]), run.cfg, train=True)
        model = U.build(run.cfg).eval()
        model.ctfm.register_forward_pre_hook(lambda mod, inp, c=config: seen.setdefault(c, []).append(inp[0]))
        torch.manual_seed(3)                                     # the same crops and flips for both
        x = torch.stack([ds[int(n)][0] for n in torch.randint(len(ds), (6,))])
        with torch.no_grad():
            model(x)
    fine = torch.cat(seen["p5_unet_ctfm_native_k2"])            # [6, 1, 16, 704, 704], one per call
    grid = torch.cat(seen["p5_unet_ctfm_k2"])                   # [6, 1, 16, 176, 176]
    assert len(seen["p5_unet_ctfm_native_k2"]) == 6 and fine.shape[-2:] == (704, 704)
    hu = [F.avg_pool3d(fine, (1, 4, 4)) * 3072 - 1024, grid * 3072 - 1024]  # CT-FM input -> HU
    inner = hu[1].abs() < 900
    assert inner.float().mean() > 0.2
    torch.testing.assert_close(hu[0][inner], hu[1][inner], atol=1.5, rtol=0)
