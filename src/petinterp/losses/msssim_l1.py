"""Default loss (CLAUDE.md): w_msssim * (1 - MS-SSIM) + w_l1 * L1, with L1 weighted up on lesions.

MS-SSIM never alone -- pure SSIM-family training is unstable. Inputs are in model (log-SUV) space.
"""

from __future__ import annotations

from pytorch_msssim import ms_ssim


def msssim_l1(pred, target, weight, cfg: dict):
    c = cfg["loss"]
    pred, target = pred.float(), target.float()
    l1 = (weight * (pred - target).abs()).mean()
    return c["w_msssim"] * (1 - ms_ssim(pred, target, data_range=c["data_range"])) + c["w_l1"] * l1
