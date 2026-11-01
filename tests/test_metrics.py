"""Lesion aggregation by patient, and the lesion tests in make_tables, on synthetic data.

    pip install -e . pytest && pytest tests/test_metrics.py
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from petinterp.evaluation import metrics as M

REPO = Path(__file__).resolve().parents[1]


def _lesion(bias, vol=2.0, bladder=50.0, mean_bias=0.0):
    return {"volume_ml": vol, "suvmax_gt": 10.0, "suvmax_pred": 10.0 * (1 + bias),
            "suvmax_bias": bias, "suvmean_bias": mean_bias, "bladder_mm": bladder}


def test_patient_lesions_medians_and_empty_strata():
    les = [_lesion(-0.1), _lesion(-0.3, vol=0.5), _lesion(0.2, bladder=10.0)]
    s = M.patient_lesions(les)
    assert s["all"]["n"] == 3 and s["all"]["suvmax_bias"] == pytest.approx(-0.1)
    assert s["all"]["suvmax_abs_bias"] == pytest.approx(0.2)
    assert s["small_lt_1ml"] == {"n": 1, "suvmax_bias": -0.3, "suvmax_abs_bias": 0.3, "suvmean_bias": 0.0}
    assert s["bladder_adjacent_lt_2cm"]["suvmax_bias"] == pytest.approx(0.2)
    none = M.patient_lesions([_lesion(-0.1)])["small_lt_1ml"]
    assert none["n"] == 0 and none["suvmax_bias"] is None


def test_aggregate_pooled_unchanged_and_by_patient():
    # patient a has many lesions, b one: pooling is carried by a, the patient view is not
    per = {"a": {"psnr": 40.0, "ssim": 0.9, "nrmse": 0.2, "lesions": [_lesion(-0.1)] * 9},
           "b": {"psnr": 42.0, "ssim": 0.95, "nrmse": 0.1, "lesions": [_lesion(-0.5)]}}
    agg = M.aggregate(per)
    assert agg["global"]["psnr"]["mean"] == pytest.approx(41.0)
    pooled = agg["lesions"]["all"]["suvmax_bias"]
    assert pooled["n"] == 10 and pooled["mean"] == pytest.approx(-0.14)      # (9*-0.1 - 0.5) / 10
    by_p = agg["lesions_by_patient"]["all"]["suvmax_bias"]
    assert by_p["n"] == 2 and by_p["mean"] == pytest.approx(-0.3)            # (-0.1 - 0.5) / 2
    assert agg["lesions_by_patient"]["small_lt_1ml"]["suvmax_bias"] == {"n": 0}


def test_evaluate_case_lesions_and_patient_summary():
    gt = np.ones((20, 20, 20), np.float32)
    ttb = np.zeros_like(gt, bool)
    ttb[2:6, 2:6, 2:6] = True                    # 64 voxels x 32 uL = 2.05 mL
    ttb[14:16, 14:16, 14:16] = True              # 8 voxels = 0.26 mL, small
    gt[ttb] = 10.0
    pred = gt.copy()
    pred[2:6, 2:6, 2:6] = 8.0                    # big lesion -20 %
    pred[14:16, 14:16, 14:16] = 5.0              # small lesion -50 %
    body = np.ones_like(gt, bool)
    r = M.evaluate_case(pred, gt, body, ttb, None, (2.0, 4.0, 4.0))
    s = M.patient_lesions(r["lesions"])
    assert s["all"]["n"] == 2 and s["all"]["suvmax_bias"] == pytest.approx(-0.35)
    assert s["small_lt_1ml"]["suvmax_bias"] == pytest.approx(-0.5)
    assert s["bladder_adjacent_lt_2cm"]["n"] == 0  # no bladder mask -> no distance


def test_wilcoxon_skips_patients_without_the_stratum():
    a = {f"p{i}": {"x": -0.1 - 0.01 * i} for i in range(8)} | {"q": {"x": None}}
    b = {f"p{i}": {"x": -0.2 - 0.01 * i} for i in range(8)} | {"q": {"x": -0.3}}
    w = M.wilcoxon_paired(a, b, "x", n_tests=3)
    assert w["n"] == 8 and w["n_a_greater"] == 8
    assert w["p_bonferroni"] == pytest.approx(min(1.0, 3 * w["p"]))


def _make_tables(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("make_tables", REPO / "scripts" / "make_tables.py")
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)
    monkeypatch.setattr(mt, "REPO", tmp_path)
    return mt


def _write_run(root, exp_id, shift, with_lesions):
    rng = np.random.default_rng(len(exp_id))
    pp = {}
    for i in range(10):
        les = [_lesion(-0.2 + shift + 0.01 * rng.standard_normal(), vol=0.5 if j == 0 else 3.0,
                       bladder=10.0 if j == 1 else 60.0) for j in range(3)]
        rec = {"psnr": 40.0 + shift * 10 + 0.1 * i, "ssim": 0.99, "nrmse": 0.2 - shift}
        if with_lesions:
            rec["lesions"] = M.patient_lesions(les)
        pp[f"p{i}"] = {"raw": rec, "consistency": rec}
    d = root / "results" / exp_id
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"per_patient": pp}), encoding="utf-8")


def test_stats_adds_lesion_family_without_touching_global_rows(tmp_path, monkeypatch):
    mt = _make_tables(tmp_path, monkeypatch)
    for e, shift in (("a1", 0.1), ("a2", 0.1), ("b1", 0.0)):
        _write_run(tmp_path / "with", e, shift, True)
        _write_run(tmp_path / "without", e, shift, False)

    monkeypatch.setattr(mt, "REPO", tmp_path / "without")
    old = mt.stats(["a1", "a2"], ["b1"])
    monkeypatch.setattr(mt, "REPO", tmp_path / "with")
    new = mt.stats(["a1", "a2"], ["b1"])

    assert len(old) == 2 * 3 and len(new) == 2 * 6
    glob_new = [r for r in new if "patient median" not in r["metric"]]
    assert glob_new == old                       # the global rows and their p-values are unchanged
    les = [r for r in new if r["variant"] == "consistency" and "patient median" in r["metric"]]
    assert [r["metric"] for r in les] == ["SUVmax bias (%), all lesions, patient median",
                                          "SUVmax bias (%), <1 mL, patient median",
                                          "SUVmax bias (%), bladder <2 cm, patient median"]
    assert all(r["a better"] == "10/10" for r in les)   # a's bias -0.1 is closer to zero than b's -0.2
