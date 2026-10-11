"""2.5D residual U-Net (model ladder rung 1): MONAI BasicUNet on block inputs (data/blocks.py).

Output = cubic_z base + learned residual. The last conv starts at zero, so the untrained network
is exactly the cubic baseline and training can only move away from it where that helps.
use_ct: false gives the no-CT arm of the main ablation with everything else identical.
"""

from __future__ import annotations

import numpy as np
import torch
from monai.networks.nets import BasicUNet
from torch import nn

from petinterp.data.blocks import block_input, n_channels, pet_inv, volumes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Residual(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.k = cfg["k"]
        self.net = BasicUNet(spatial_dims=2, in_channels=n_channels(cfg), out_channels=self.k,
                             features=tuple(cfg["model"]["features"]), norm=cfg["model"]["norm"])
        nn.init.zeros_(self.net.final_conv.weight)
        nn.init.zeros_(self.net.final_conv.bias)

    def forward(self, x):
        return x[:, -self.k:] + self.net(x)  # base channels are always last


_loaded: dict = {}


def load(run) -> Residual:
    """The evaluation checkpoint of this run, refused if it was trained under another config."""
    path = run.dir / "checkpoints" / f"{run.cfg['eval']['checkpoint']}.pt"
    if path not in _loaded:
        s = torch.load(path, map_location=DEVICE, weights_only=False)
        if s["cfg_hash"] != run.cfg_hash:
            raise RuntimeError(f"{path} was trained under config {s['cfg_hash']}, not {run.cfg_hash}")
        model = Residual(run.cfg).to(DEVICE)
        model.load_state_dict(s["model"])
        _loaded[path] = model.eval()
        run.log(f"loaded {path.name} (epoch {s['epoch']})")
    return _loaded[path]


@torch.no_grad()
def predict(run, lr: np.ndarray, ct_hu: np.ndarray) -> np.ndarray:
    """Thick SUV [D/k,H,W] + CT HU on the thin grid [D,H,W] -> thin SUV [D,H,W], full slices."""
    cfg, k, bs = run.cfg, run.cfg["k"], run.cfg["eval"]["batch"]
    model, v = load(run), volumes(lr, ct_hu, k, cfg)
    out = v["base"].copy()
    for s in range(0, len(lr), bs):
        ii = range(s, min(s + bs, len(lr)))
        x = torch.from_numpy(np.stack([block_input(v, i, k, cfg["model"]) for i in ii])).to(DEVICE)
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            y = model(x)
        out[s * k:(s + len(ii)) * k] = y.float().cpu().numpy().reshape(-1, *out.shape[1:])
    return np.clip(pet_inv(out, cfg["data"]["suv_ref"]), 0, None).astype(np.float32)
