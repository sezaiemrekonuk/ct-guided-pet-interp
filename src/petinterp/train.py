"""train(run): fit the model the config names on the train split, resumable across sessions.

Rule 8: every epoch writes checkpoints/last.pt (model, optimizer, scheduler, scaler, epoch, RNG
state, config hash) atomically to the run directory on Drive, plus best.pt whenever the
validation loss improves. Re-running the same cell resumes from last.pt; max_session_hours stops
the loop cleanly before the session budget runs out. An epoch is a fixed number of random blocks.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from petinterp.data.blocks import CACHED, block_input, cache
from petinterp.losses.msssim_l1 import msssim_l1
from petinterp.models.unet25d import DEVICE, Residual
from petinterp.runtime import Run, write_json


class Blocks(Dataset):
    """(input, target, L1 weight) per thick slice. Train: random in-plane crop + random L-R flip.
    Otherwise: centre crop of every stride-th block. Never augmented along z."""

    def __init__(self, dirs, cfg: dict, train: bool, stride: int = 1):
        self.cfg, self.k, self.train = cfg, cfg["k"], train
        self.vols = [{n: np.load(d / f"{n}.npy", mmap_mode="r") for n in CACHED} for d in dirs]
        self.index = [(j, i) for j, v in enumerate(self.vols) for i in range(0, len(v["lr"]), stride)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        j, i = self.index[n]
        v, k, p = self.vols[j], self.k, self.cfg["train"]["patch"]
        h, w = v["hr"].shape[1:]
        if self.train:
            y0, x0 = int(torch.randint(h - p + 1, ())), int(torch.randint(w - p + 1, ()))
        else:
            y0, x0 = (h - p) // 2, (w - p) // 2
        zyx = (slice(i * k, (i + 1) * k), slice(y0, y0 + p), slice(x0, x0 + p))
        x = block_input(v, i, k, self.cfg["model"], zyx[1:])
        t = v["hr"][zyx].astype(np.float32)
        wt = 1 + (self.cfg["loss"]["lesion_weight"] - 1) * v["ttb"][zyx].astype(np.float32)
        if self.train and torch.rand(()) < 0.5:  # L-R is the last array axis
            x, t, wt = x[..., ::-1], t[..., ::-1], wt[..., ::-1]
        return tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in (x, t, wt))


def _save(obj, path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def _validate(model, ds: Blocks, cfg: dict) -> float:
    model.eval()
    tot = 0.0
    for x, y, w in DataLoader(ds, batch_size=cfg["train"]["batch"], num_workers=cfg["train"]["workers"]):
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            pred = model(x.to(DEVICE))
        tot += msssim_l1(pred, y.to(DEVICE), w.to(DEVICE), cfg).item() * len(x)
    return tot / len(ds)


def train(run: Run) -> list[dict]:
    cfg, t = run.cfg, run.cfg["train"]
    if DEVICE.type != "cuda":
        run.log("WARNING: no GPU -- Runtime > Change runtime type > GPU, unless this is a smoke test")
    torch.backends.cudnn.benchmark = True
    split = json.loads((run.path(cfg["processed"]) / cfg["split_file"]).read_text(encoding="utf-8"))
    tr = Blocks(cache(run, split["train"]), cfg, train=True)
    va = Blocks(cache(run, split["val"]), cfg, train=False, stride=t["val_stride"])

    torch.manual_seed(t["seed"])
    model = Residual(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=DEVICE.type == "cuda")
    gen = torch.Generator().manual_seed(t["seed"])
    run.log(f"train {len(tr)} blocks / {len(split['train'])} patients, val {len(va)} blocks; "
            f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params on {DEVICE}")

    ck = run.dir / "checkpoints"
    ck.mkdir(exist_ok=True)
    epoch0, best, history = 0, float("inf"), []
    if (ck / "last.pt").exists():
        s = torch.load(ck / "last.pt", map_location=DEVICE, weights_only=False)
        if s["cfg_hash"] != run.cfg_hash:
            raise RuntimeError(f"checkpoints come from config {s['cfg_hash']}, this is {run.cfg_hash}; "
                               "a changed config is a new exp_id")
        for obj, key in ((model, "model"), (opt, "opt"), (sched, "sched"), (scaler, "scaler")):
            obj.load_state_dict(s[key])
        gen.set_state(s["rng"]["loader"].cpu())  # map_location moved them; RNG states must be CPU
        torch.set_rng_state(s["rng"]["torch"].cpu())
        epoch0, best, history = s["epoch"] + 1, s["best"], s["history"]
        run.log(f"resume at epoch {epoch0}/{t['epochs']} (best val {best:.4f})")
    else:
        run.log(f"epoch -1 (untrained = cubic baseline): val {_validate(model, va, cfg):.4f}")

    loader = DataLoader(tr, batch_size=t["batch"], num_workers=t["workers"], generator=gen,
                        pin_memory=DEVICE.type == "cuda", drop_last=True,
                        sampler=RandomSampler(tr, replacement=True, generator=gen,
                                              num_samples=t["steps_per_epoch"] * t["batch"]))
    t0 = time.time()
    for epoch in range(epoch0, t["epochs"]):
        e0 = time.time()
        model.train()
        tot = torch.zeros((), device=DEVICE)
        for x, y, w in loader:
            x, y, w = (a.to(DEVICE, non_blocking=True) for a in (x, y, w))
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                pred = model(x)
            loss = msssim_l1(pred, y, w, cfg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.detach()
        sched.step()
        val = _validate(model, va, cfg)
        history.append({"epoch": epoch, "train_loss": tot.item() / t["steps_per_epoch"], "val_loss": val,
                        "lr": opt.param_groups[0]["lr"], "sec": round(time.time() - e0, 1)})
        if val < best:
            best = val
            _save({"model": model.state_dict(), "epoch": epoch, "val_loss": val, "cfg_hash": run.cfg_hash},
                  ck / "best.pt")
        _save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
               "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "history": history,
               "cfg_hash": run.cfg_hash, "rng": {"loader": gen.get_state(), "torch": torch.get_rng_state()}},
              ck / "last.pt")
        write_json(run.dir / "train_history.json", history)
        h = history[-1]
        run.log(f"epoch {epoch}/{t['epochs'] - 1} train {h['train_loss']:.4f} val {val:.4f}"
                f"{' *best' if val == best else ''} ({h['sec']:.0f}s)")
        if time.time() - t0 + h["sec"] > t["max_session_hours"] * 3600 and epoch + 1 < t["epochs"]:
            run.log("session budget reached -- stopped cleanly; re-run this cell to resume")
            break
    return history
