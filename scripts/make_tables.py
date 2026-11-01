"""results/<exp_id>/metrics.json -> paper/tables/<name>.csv and .tex (booktabs). CLAUDE.md rule 2.

    python scripts/make_tables.py main_k2 p3_cubic_k2 p4_unet_noct_k2 p4_unet_k2
    python scripts/make_tables.py stats_k2 p4_unet_noct_k2 p4_unet_noct_k2_s2 vs p4_unet_k2 p4_unet_k2_s2

main: one row per (exp_id, variant): global PSNR/SSIM/NRMSE inside the body mask (patient mean ±
SD) and lesion SUVmax bias (mean ± SD, in %) for all lesions, small (<1 mL) and bladder-adjacent
(<2 cm). Rows follow the order the exp_ids are given in.

stats (exp_ids split by "vs"): arm a against arm b. Each patient's metric is averaged over the
seeds of its arm, then compared patient-paired with Wilcoxon signed-rank, Bonferroni over
PSNR/SSIM/NRMSE within a variant. One row per (variant, metric). When the metrics.json files
carry per-patient lesion summaries, the lesion SUVmax bias is tested the same way on each
patient's median over its lesions, for all / small / bladder-adjacent lesions, as a separate
Bonferroni family of three. A patient with no lesion in a stratum drops out of that test, and
"a better" counts patients where arm a's bias is closer to zero.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from petinterp.evaluation.metrics import wilcoxon_paired  # noqa: E402  rule 6: tests live there

LOWER_IS_BETTER = {"nrmse"}
GLOBAL = (("psnr", "PSNR (dB)", 2, 1), ("ssim", "SSIM", 4, 1), ("nrmse", "NRMSE", 4, 1))
LESION = (("all", "SUVmax bias, all (%)"), ("small_lt_1ml", "small <1 mL (%)"),
          ("bladder_adjacent_lt_2cm", "bladder <2 cm (%)"))
LESION_TEST = {"all": "SUVmax bias (%), all lesions", "small_lt_1ml": "SUVmax bias (%), <1 mL",
               "bladder_adjacent_lt_2cm": "SUVmax bias (%), bladder <2 cm"}


def _pm(s: dict, digits: int, scale: float) -> str:
    if not s.get("n"):
        return "--"
    return f"{s['mean'] * scale:.{digits}f} ± {s['sd'] * scale:.{digits}f}"


def rows(exp_ids: list[str]) -> list[dict]:
    out = []
    for e in exp_ids:
        m = json.loads((REPO / "results" / e / "metrics.json").read_text(encoding="utf-8"))
        for v in ("raw", "consistency"):
            r = {"exp_id": e, "k": m["k"], "variant": v, "patients": m[v]["n_patients"]}
            r.update({label: _pm(m[v]["global"][key], d, s) for key, label, d, s in GLOBAL})
            r.update({label: _pm(m[v]["lesions"][key]["suvmax_bias"], 1, 100) for key, label in LESION})
            r["lesions"] = m[v]["lesions"]["all"]["suvmax_bias"]["n"]
            out.append(r)
    return out


def _per_patient(exp_ids: list[str]) -> list[dict]:
    return [json.loads((REPO / "results" / e / "metrics.json").read_text(encoding="utf-8"))["per_patient"]
            for e in exp_ids]


def _seed_mean(exp_ids: list[str], variant: str) -> dict[str, dict]:
    pp = _per_patient(exp_ids)
    return {c: {key: statistics.mean(p[c][variant][key] for p in pp) for key, *_ in GLOBAL} for c in pp[0]}


def _has_lesions(exp_ids: list[str]) -> bool:
    return all("lesions" in r["raw"] for p in _per_patient(exp_ids) for r in p.values())


def _seed_mean_lesions(exp_ids: list[str], variant: str) -> dict[str, dict]:
    """case -> {stratum: seed mean of the patient's median SUVmax bias, None if no such lesion}."""
    pp = _per_patient(exp_ids)
    out = {}
    for c in pp[0]:
        out[c] = {}
        for key, _ in LESION:
            vals = [p[c][variant]["lesions"][key]["suvmax_bias"] for p in pp]
            out[c][key] = None if None in vals else statistics.mean(vals)
    return out


def _lesion_rows(a: list[str], b: list[str], v: str) -> list[dict]:
    la, lb = _seed_mean_lesions(a, v), _seed_mean_lesions(b, v)
    out = []
    for key, _ in LESION:
        w = wilcoxon_paired(la, lb, key, n_tests=len(LESION))
        cases = [c for c in la if la[c][key] is not None and lb[c][key] is not None]
        out.append({"a": a[0], "b": b[0], "seeds": f"{len(a)}/{len(b)}", "variant": v,
                    "metric": f"{LESION_TEST[key]}, patient median",
                    "mean a": f"{100 * statistics.mean(la[c][key] for c in cases):.1f}",
                    "mean b": f"{100 * statistics.mean(lb[c][key] for c in cases):.1f}",
                    "median a-b": f"{100 * w['median_diff']:.2f}",
                    "a better": f"{sum(abs(la[c][key]) < abs(lb[c][key]) for c in cases)}/{w['n']}",
                    "p (Bonferroni)": f"{w['p_bonferroni']:.2g}"})
    return out


def stats(a: list[str], b: list[str]) -> list[dict]:
    out = []
    lesions = _has_lesions(a) and _has_lesions(b)
    if not lesions:
        print("per-patient lesion summaries missing in some metrics.json: lesion tests skipped "
              "(re-run the evaluate notebook to rebuild them)")
    for v in ("raw", "consistency"):
        ma, mb = _seed_mean(a, v), _seed_mean(b, v)
        for key, label, d, _ in GLOBAL:
            w = wilcoxon_paired(ma, mb, key, n_tests=len(GLOBAL))
            out.append({"a": a[0], "b": b[0], "seeds": f"{len(a)}/{len(b)}", "variant": v, "metric": label,
                        "mean a": f"{statistics.mean(r[key] for r in ma.values()):.{d}f}",
                        "mean b": f"{statistics.mean(r[key] for r in mb.values()):.{d}f}",
                        "median a-b": f"{w['median_diff']:.{d + 2}f}",
                        "a better": f"{w['n'] - w['n_a_greater'] if key in LOWER_IS_BETTER else w['n_a_greater']}/{w['n']}",
                        "p (Bonferroni)": f"{w['p_bonferroni']:.2g}"})
        if lesions:
            out += _lesion_rows(a, b, v)
    return out


def write(name: str, table: list[dict]) -> None:
    d = REPO / "paper" / "tables"
    d.mkdir(parents=True, exist_ok=True)
    cols = list(table[0])
    with open(d / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(table)
    tex = lambda s: str(s).replace("_", r"\_").replace("±", r"$\pm$").replace("%", r"\%").replace("<", r"$<$")  # noqa: E731
    lines = [r"\begin{tabular}{" + "l" * 3 + "r" * (len(cols) - 3) + "}", r"\toprule",
             " & ".join(tex(c) for c in cols) + r" \\", r"\midrule",
             *(" & ".join(tex(r[c]) for c in cols) + r" \\" for r in table), r"\bottomrule", r"\end{tabular}"]
    (d / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    args = sys.argv[2:]
    if "vs" in args:
        i = args.index("vs")
        write(sys.argv[1], stats(args[:i], args[i + 1:]))
    else:
        write(sys.argv[1], rows(args))
