"""2.5D residual U-Net (model ladder rung 1): MONAI BasicUNet on block inputs (data/blocks.py).

Output = cubic_z base + learned residual. The last conv starts at zero, so the untrained network
is exactly the cubic baseline and training can only move away from it where that helps.
use_ct: false gives the no-CT arm of the main ablation with everything else identical.
name: unet25d_ctfm swaps the raw CT channels for features of the frozen CT-FM encoder,
z-averaged (ctfm.z absent or "mean") or kept per block slice (ctfm.z: block), and fed either the
CT on the PET grid or, with ctfm.ct_scale, the native-resolution CT of data/ct_native.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import BasicUNet
from torch import nn

from petinterp.data.blocks import block_input, ct_scale, ct_z, n_channels, pet_inv, volumes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Residual(nn.Module):
    def __init__(self, cfg: dict, in_channels: int | None = None):
        super().__init__()
        self.k = cfg["k"]
        self.net = BasicUNet(spatial_dims=2, in_channels=in_channels or n_channels(cfg), out_channels=self.k,
                             features=tuple(cfg["model"]["features"]), norm=cfg["model"]["norm"])
        nn.init.zeros_(self.net.final_conv.weight)
        nn.init.zeros_(self.net.final_conv.bias)

    def forward(self, x):
        return x[:, -self.k:] + self.net(x)  # base channels are always last


def ctfm_encoder(c: dict) -> nn.Module:
    """CT-FM's 3D SegResEncoder (Pai et al. 2025) with its pretrained weights, at the pinned revision.

    The Hugging Face checkpoint is a plain MONAI SegResEncoder state dict, so no lighter_zoo.
    """
    from huggingface_hub import hf_hub_download
    from monai.networks.nets.segresnet_ds import SegResEncoder
    from safetensors.torch import load_file

    def get(f):
        return hf_hub_download(c["repo"], f, revision=c["revision"])

    enc = SegResEncoder(**json.loads(Path(get("config.json")).read_text(encoding="utf-8")))
    enc.load_state_dict(load_file(get("model.safetensors")))
    return enc


def ctfm_rows(depths: list[int], slab: int, lo: int, hi: int) -> list[list[int]]:
    """Per CT-FM level, the z rows nearest to slab slices lo..hi (inclusive).

    Level l has depths[l] rows over the slab; each stride-2 conv centres its output row j on
    input row 2j, so row j of a level with depth D sits at slab slice j * slab / D.
    """
    out = []
    for d in depths:
        s = slab / d
        out.append(sorted({min(d - 1, int(np.floor(z / s + 0.5))) for z in range(lo, hi + 1)}))
    return out


class CTFMResidual(Residual):
    """Residual U-Net whose CT branch is the frozen CT-FM encoder instead of raw CT channels.

    The ct_slab thin CT slices around the block go through CT-FM as one 3D volume. Each of its
    feature levels is reduced to 2D, pooled to the in-plane size of the matching BasicUNet
    encoder level and added there through a 1x1 conv. Those convs start at zero, so the untrained
    network is still exactly the cubic baseline. HU above data.ct_hu's upper bound is already
    clipped in the cache, so CT-FM never sees bone above 1000 HU (it was trained up to 2048).

    ctfm.z picks the reduction. "mean" averages each level over the whole slab, so all k target
    slices get the same CT summary. "block" keeps the rows that sit on the block's thin slices
    +- ctfm.z_context and stacks them as channels, so where in z an anatomical edge falls
    survives into the U-Net, as it does for the raw CT channels of p4_unet_k*. The deep levels
    have one or two rows over the slab and differ little between the two.

    ctfm.ct_scale = s feeds CT-FM the native-resolution CT instead (data/ct_native: s x s CT
    pixels per PET pixel, clipped only at CT-FM's own window), which arrives packed as S*s*s
    channels on the PET grid and is unpacked here. Every CT-FM level is then s times the U-Net
    level in-plane, so the pooling averages exactly the CT pixels under each PET pixel. CT-FM
    runs one sample at a time there: its first level alone is 0.6 GB per 192-pixel patch.
    """

    def __init__(self, cfg: dict):
        m, c, k = cfg["model"], cfg["model"]["ctfm"], cfg["k"]
        scale = ct_scale(m)
        super().__init__(cfg, in_channels=n_channels(cfg) - m["ct_slab"] * (scale or 1) ** 2)
        self.pet, self.slab, self.scale = 2 * m["pet_context"] + 1, m["ct_slab"], scale
        self.hu, self.window = cfg["data"]["ct_hu"], c["hu"]
        self.ctfm = ctfm_encoder(c).requires_grad_(False)
        with torch.no_grad():
            shapes = [f.shape for f in self.ctfm.eval()(torch.zeros(1, 1, self.slab, 32, 32))]
        if c.get("z", "mean") == "block":  # absent in the p5_unet_ctfm_k* configs: z-mean
            lo = -int(ct_z(0, k, m)[0]) - c["z_context"]  # block start in the slab, minus context
            hi = lo + k - 1 + 2 * c["z_context"]
            if lo < 0 or hi >= self.slab:
                raise ValueError(f"ct_slab {self.slab} does not cover the block +- z_context at k={k}")
            self.rows = ctfm_rows([s[2] for s in shapes], self.slab, lo, hi)
        else:
            self.rows = None
        chans = [s[1] * (len(self.rows[i]) if self.rows else 1) for i, s in enumerate(shapes)]
        self.proj = nn.ModuleList(nn.Conv2d(ci, co, 1) for ci, co in zip(chans, m["features"]))
        for p in self.proj.parameters():
            nn.init.zeros_(p)

    def _ct(self, f: torch.Tensor, level: int, hw) -> torch.Tensor:
        """One CT-FM level [N,C,D,h,w] -> [N,C',H,W] on the U-Net level's in-plane grid."""
        if self.rows is None:
            return F.adaptive_avg_pool3d(f, (1, *hw))[:, :, 0]
        f = f[:, :, self.rows[level]].flatten(1, 2)  # rows as channels, channel-major
        return F.adaptive_avg_pool2d(f, hw)

    def train(self, mode: bool = True):
        super().train(mode)
        self.ctfm.eval()  # frozen: BatchNorm keeps CT-FM's own statistics
        return self

    def forward(self, x):
        p, (lo, hi), (a, b) = self.pet, self.hu, self.window
        if self.scale:  # packed native CT, already in CT-FM's window as [-1, 1]
            s = self.slab * self.scale ** 2
            ct, chunk = (F.pixel_shuffle(x[:, p:p + s], self.scale) + 1) / 2, 1
        else:
            s = self.slab
            hu = (x[:, p:p + s] + 1) / 2 * (hi - lo) + lo  # back from [-1, 1] to HU, then CT-FM's window
            ct, chunk = (hu.clamp(a, b) - a) / (b - a), len(x)
        hws = [tuple(x.shape[-2:])]  # in-plane size of each U-Net level: BasicUNet max-pools by 2
        for _ in range(4):
            hws.append((hws[-1][0] // 2, hws[-1][1] // 2))
        with torch.no_grad():  # pooled per chunk, so CT-FM's full-size levels never pile up
            parts = [[self._ct(f, i, hw) for i, (f, hw) in enumerate(zip(self.ctfm(c), hws))]
                     for c in ct.unsqueeze(1).split(chunk)]
        feats = [torch.cat(level) for level in zip(*parts)]
        n, h, skips = self.net, torch.cat([x[:, :p], x[:, p + s:]], 1), []
        for layer, f, proj in zip((n.conv_0, n.down_1, n.down_2, n.down_3, n.down_4), feats, self.proj):
            h = layer(h)
            h = h + proj(f.to(h.dtype))
            skips.append(h)
        u = n.upcat_4(skips[4], skips[3])
        for up, skip in zip((n.upcat_3, n.upcat_2, n.upcat_1), skips[2::-1]):
            u = up(u, skip)
        return x[:, -self.k:] + n.final_conv(u)


MODELS = {"unet25d": Residual, "unet25d_ctfm": CTFMResidual}


def build(cfg: dict) -> Residual:
    return MODELS[cfg["model"]["name"]](cfg)


_loaded: dict = {}


def load(run) -> Residual:
    """The evaluation checkpoint of this run, refused if it was trained under another config."""
    path = run.dir / "checkpoints" / f"{run.cfg['eval']['checkpoint']}.pt"
    if path not in _loaded:
        s = torch.load(path, map_location=DEVICE, weights_only=False)
        if s["cfg_hash"] != run.cfg_hash:
            raise RuntimeError(f"{path} was trained under config {s['cfg_hash']}, not {run.cfg_hash}")
        model = build(run.cfg).to(DEVICE)
        model.load_state_dict(s["model"])
        _loaded[path] = model.eval()
        run.log(f"loaded {path.name} (epoch {s['epoch']})")
    return _loaded[path]


@torch.no_grad()
def predict(run, lr: np.ndarray, ct_hu: np.ndarray, ctn_hu: np.ndarray | None = None) -> np.ndarray:
    """Thick SUV [D/k,H,W] + CT HU on the thin grid [D,H,W] (+ native CT [D,sH,sW]) -> thin SUV
    [D,H,W], full slices."""
    cfg, k, bs = run.cfg, run.cfg["k"], run.cfg["eval"]["batch"]
    model, v = load(run), volumes(lr, ct_hu, k, cfg, ctn_hu)
    out = v["base"].copy()
    for s in range(0, len(lr), bs):
        ii = range(s, min(s + bs, len(lr)))
        x = torch.from_numpy(np.stack([block_input(v, i, k, cfg["model"]) for i in ii])).to(DEVICE)
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            y = model(x)
        out[s * k:(s + len(ii)) * k] = y.float().cpu().numpy().reshape(-1, *out.shape[1:])
    return np.clip(pet_inv(out, cfg["data"]["suv_ref"]), 0, None).astype(np.float32)
