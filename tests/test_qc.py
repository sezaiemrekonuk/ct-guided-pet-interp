"""The SUV check must tell a calibration error from biology on noisy synthetic DEEP-PSMA-style zips.

    pip install -e . pytest && pytest tests/test_qc.py
"""

import json
import shutil
import zipfile

import numpy as np
import pytest
import SimpleITK as sitk
import yaml

import petinterp.runtime as rt
from petinterp.data.qc import suv_check

CONFIG = "configs/p1_qc_suv_deep_psma.yaml"
SHAPE = (30, 48, 48)                                    # z, y, x at 2 x 4 x 4 mm: 32 mm^3 voxels
ORGANS = {5: np.s_[2:22, 2:16, 2:16], 1: np.s_[2:12, 2:10, 22:30], 2: np.s_[14:22, 20:28, 2:8],
          3: np.s_[14:22, 20:28, 10:16], 52: np.s_[2:28, 32:40, 22:30]}
BASE = {5: 4.8, 1: 6.0, 2: 23.0, 3: 23.0, 52: 1.5}       # PSMA organ SUVs before shared factor and biology
FDG = {5: 2.3, 1: 2.0, 2: 2.0, 3: 2.0, 52: 1.6}
LESION, HALO = np.s_[10:12, 8:10, 8:10], np.s_[9:13, 7:11, 7:11]   # liver metastasis + its spill-out


def _img(arr, dtype):
    img = sitk.GetImageFromArray(arr.astype(dtype))
    img.SetSpacing((4.0, 4.0, 2.0))
    return img


def _scan(organ_suv, tumour_width, drop_label=None):
    pet = np.full(SHAPE, 0.5, np.float32)
    seg = np.zeros(SHAPE, np.uint8)
    for lid, sl in ORGANS.items():
        if lid != drop_label:
            seg[sl] = lid
        pet[sl] = organ_suv[lid]
    pet[HALO] = 12.0                                    # inside the liver mask, outside TTB
    ttb = np.zeros(SHAPE, np.uint8)
    ttb[24:30, 2:12, 34:34 + tumour_width] = 1          # nodal disease, outside every organ: 60*w voxels
    ttb[LESION] = 1                                     # + 8 voxels
    pet[ttb > 0] = 20.0
    return {"PET.nii.gz": _img(pet, np.float32), "totseg_24.nii.gz": _img(seg, np.uint8),
            "TTB.nii.gz": _img(ttb, np.uint8)}


def _archive(root, psma_cases, skip=(), name="deep_psma"):
    """psma_cases: one (organ SUVs, tumour width, dropped label) per patient; FDG is always normal."""
    src, dst = root / "src", root / "data/raw" / name
    shutil.rmtree(src, ignore_errors=True)
    shutil.rmtree(dst, ignore_errors=True)
    for i, (organ, width, drop) in enumerate(psma_cases):
        for tracer, imgs in (("PSMA", _scan(organ, width, drop)), ("FDG", _scan(FDG, 4))):
            d = src / f"train_{i:04d}" / tracer
            d.mkdir(parents=True)
            for f, img in imgs.items():
                if (i, tracer, f) not in skip:
                    sitk.WriteImage(img, str(d / f))
    dst.mkdir(parents=True)
    with zipfile.ZipFile(dst / "cases.zip", "w") as z:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(src).as_posix())


def _cohort(scenario, n=60, seed=0):
    """Per patient: shared factor x organ biology, tumour width independent of the shared factor.
    calibration: big shared spread, small organ biology.
    biology: tiny shared spread; liver (and half as much spleen) falls with tumour volume.
    global_sink: tumour volume lowers every organ alike, on top of the usual organ noise."""
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n):
        width = int(rng.integers(1, 13))
        noise = {k: rng.normal(0, 0.08) for k in BASE}
        if scenario == "calibration":
            shared, bio = rng.normal(0, 0.35), noise
        elif scenario == "biology":
            shared, sink = rng.normal(0, 0.03), -0.08 * (width - 6) + rng.normal(0, 0.15)
            bio = {**noise, 5: sink, 1: sink / 2}
        else:                                           # linear in log TTB, the form the check adjusts for
            ttb_ml = (60 * width + 8) * 32 / 1000
            shared, bio = -0.4 * (np.log1p(ttb_ml) - 2.5), noise
        bio[3] = bio[2]
        cases.append(({k: BASE[k] * np.exp(shared + bio[k]) for k in BASE}, width, None))
    return cases


def _run(tmp_path, monkeypatch, **overrides):
    monkeypatch.setattr(rt, "find_data_root", lambda: tmp_path)
    if not overrides:
        return rt.start(CONFIG)
    cfg = {**yaml.safe_load((rt.REPO / CONFIG).read_text(encoding="utf-8")), **overrides}
    (tmp_path / "variant.yaml").write_text(yaml.safe_dump(cfg))
    return rt.start(str(tmp_path / "variant.yaml"))


def _ok(est):
    """A bootstrap estimate with a real interval around it."""
    lo, hi = est["ci95"]
    return lo < est["value"] < hi and hi - lo > 0


def test_calibration_error_is_shared_by_every_organ(tmp_path, monkeypatch):
    _archive(tmp_path, _cohort("calibration"))
    aorta = suv_check(_run(tmp_path, monkeypatch))["scale"]["aorta"]
    assert _ok(aorta["slope_on_liver"]) and aorta["slope_on_liver"]["ci95"][0] > 0.7
    assert aorta["slope_on_liver_given_ttb"]["value"] > 0.7      # tumour volume cannot explain it
    assert aorta["drop_below_vs_in"]["value"] < -0.15            # below-range scans are low everywhere


def test_biology_lowers_liver_alone_and_tracks_tumour(tmp_path, monkeypatch):
    cases = _cohort("biology")
    _archive(tmp_path, cases)
    run = _run(tmp_path, monkeypatch)
    s = suv_check(run)
    aorta = s["scale"]["aorta"]
    assert _ok(aorta["slope_on_liver"]) and aorta["slope_on_liver"]["ci95"][1] < 0.3
    assert abs(aorta["drop_below_vs_in"]["value"]) < 0.1
    assert aorta["liver_drop_same_patients"]["value"] < -0.2
    assert s["tumour_sink"]["liver_vs_ttb_ml"]["rho"] < -0.5

    row = json.loads((run.dir / "per_patient/train_0000.json").read_text())
    assert row["PSMA"]["ttb_ml"] == pytest.approx((60 * cases[0][1] + 8) * 32 / 1000)
    # only the eroded core is free of the metastasis spill-out; the lesion-free mean still sees it
    assert row["PSMA"]["core_suvmean"]["liver"] == pytest.approx(cases[0][0][5], rel=1e-5)
    assert row["PSMA"]["organ_suvmean"]["liver"] > row["PSMA"]["core_suvmean"]["liver"]
    assert row["PSMA"]["liver_with_lesions"] > row["PSMA"]["organ_suvmean"]["liver"]


def test_sink_on_every_organ_is_not_mistaken_for_calibration(tmp_path, monkeypatch):
    _archive(tmp_path, _cohort("global_sink"))
    aorta = suv_check(_run(tmp_path, monkeypatch))["scale"]["aorta"]
    assert aorta["slope_on_liver"]["value"] > 0.7                 # shared, as calibration would be
    given = aorta["slope_on_liver_given_ttb"]                     # but tumour volume accounts for it
    assert abs(given["value"]) < 0.3 and given["ci95"][0] < 0 < given["ci95"][1]
    assert aorta["vs_ttb_ml"]["rho"] < -0.8


def test_groups_exclusions_and_small_cohorts(tmp_path, monkeypatch):
    normal = {**BASE, 5: 5.5}
    cases = [(normal, 3, None), (normal, 5, None), ({**BASE, 5: 9.0}, 7, None), (normal, 9, 5), (normal, 4, None)]
    _archive(tmp_path, cases, skip={(4, "PSMA", "TTB.nii.gz")})
    s = suv_check(_run(tmp_path, monkeypatch))
    json.dumps(s, allow_nan=False)                                # strict JSON, no NaN tokens
    assert list(s["excluded"]) == ["train_0004"] and s["n_cases"] == 4
    assert {k: v for k, v in s["crosstab"].items() if v} == {
        "PSMA_in__FDG_in": 2, "PSMA_above__FDG_in": 1, "PSMA_missing__FDG_in": 1}
    assert s["scale"]["aorta"]["slope_on_liver"]["value"] is None   # below min_group
    assert s["FDG_liver_in_PSMA_below_range"]["n"] == 0

    _archive(tmp_path, cases)                                     # the missing file turns up
    s = suv_check(_run(tmp_path, monkeypatch))
    assert s["excluded"] == {} and s["n_cases"] == 5              # an exclusion is retried, not cached


@pytest.mark.parametrize("overrides, recomputed", [
    ({"labels": {"liver": [5], "spleen": [1, 99], "kidneys": [2, 3], "aorta": [52]}}, True),
    ({"test_tracer": "FDG", "reference_tracer": "PSMA"}, True),
    ({"source": {"archives": "data/raw/copy"}}, True),
    ({"core_erosion_mm": {"liver": 4, "aorta": 4}}, True),
    ({"liver_suvmean_range": {"PSMA": [3.0, 9.0], "FDG": [1.0, 3.5]}}, False),
    ({"reference_organs": ["aorta"], "histogram_bin_suv": 1.0, "min_group": 2}, False),
])
def test_cache_tracks_measurement_settings_only(tmp_path, monkeypatch, overrides, recomputed):
    cases = [({**BASE, 5: 5.5}, 3, None)] * 2
    _archive(tmp_path, cases)
    _archive(tmp_path, cases, name="copy")
    run = _run(tmp_path, monkeypatch)
    suv_check(run)
    path = run.dir / "per_patient/train_0000.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "marker": True}))
    suv_check(_run(tmp_path, monkeypatch, **overrides))
    assert ("marker" not in json.loads(path.read_text())) is recomputed
