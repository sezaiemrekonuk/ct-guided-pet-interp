"""Paper figures -> paper/figures/<name>.pdf at 300 dpi (CLAUDE.md rule 3). Reads results/ only.

    python scripts/make_figures.py curves learning_curves_k2 p4_unet_k2 p4_unet_k2_s2 ...

curves: validation loss per epoch, one thin line per run, coloured by arm (CT vs no CT, from the
run's config), so seed spread and the gap between arms read off one axis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ARM = {True: ("PET + CT", "#2a78d6"), False: ("PET only", "#eb6834")}  # validated categorical pair
INK, MUTED = "#1f1f1e", "#8a8984"
HEADROOM = 1.15  # axis ceiling over the highest first-epoch val loss; 0.0030 at k=2, scales with k


def _config(exp_id: str) -> dict:
    return next(yaml.safe_load(p.read_text(encoding="utf-8")) for p in (REPO / "configs").glob("p*.yaml")
                if yaml.safe_load(p.read_text(encoding="utf-8"))["exp_id"] == exp_id)


def curves(name: str, exp_ids: list[str]) -> Path:
    fig, ax = plt.subplots(figsize=(3.5, 2.6))  # one journal column
    seen = set()
    hist = {e: json.loads((REPO / "results" / e / "train_history.json").read_text(encoding="utf-8"))
            for e in exp_ids}
    ymax = HEADROOM * max(h[0]["val_loss"] for h in hist.values())
    for e, h in hist.items():
        label, color = ARM[_config(e)["model"]["use_ct"]]
        x, y = [r["epoch"] for r in h], [r["val_loss"] for r in h]
        ax.plot(x, [min(v, ymax) for v in y], color=color, lw=1.2, alpha=0.85,
                label=None if label in seen else label)
        seen.add(label)
        for xi, yi in zip(x, y):  # one-epoch spikes are clipped, never hidden: their value is printed
            if yi > ymax:
                ax.annotate(f"{yi:.4f}", (xi, ymax), xytext=(3, -2), textcoords="offset points",
                            fontsize=6, color=INK, va="top")
    ax.set_ylim(top=ymax)
    ax.set_xlabel("epoch", color=INK)
    ax.set_ylabel("validation loss", color=INK)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    out = REPO / "paper" / "figures" / f"{name}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=150)  # preview only; the PDF is the paper asset
    plt.close(fig)
    return out


if __name__ == "__main__":
    print({"curves": curves}[sys.argv[1]](sys.argv[2], sys.argv[3:]))
