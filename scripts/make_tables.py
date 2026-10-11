"""results/<exp_id>/metrics.json -> paper/tables/<name>.csv and .tex (booktabs). CLAUDE.md rule 2.

    python scripts/make_tables.py main_k2 p3_cubic_k2 p4_unet_noct_k2 p4_unet_k2

One row per (exp_id, variant): global PSNR/SSIM/NRMSE inside the body mask (patient mean ± SD)
and lesion SUVmax bias (mean ± SD, in %) for all lesions, small (<1 mL) and bladder-adjacent
(<2 cm). Rows follow the order the exp_ids are given in.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GLOBAL = (("psnr", "PSNR (dB)", 2, 1), ("ssim", "SSIM", 4, 1), ("nrmse", "NRMSE", 4, 1))
LESION = (("all", "SUVmax bias, all (%)"), ("small_lt_1ml", "small <1 mL (%)"),
          ("bladder_adjacent_lt_2cm", "bladder <2 cm (%)"))


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
    write(sys.argv[1], rows(sys.argv[2:]))
