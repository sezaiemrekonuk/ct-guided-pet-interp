"""Classical z-interpolation baselines. No training, no CT."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def cubic_z(lr: np.ndarray, k: int) -> np.ndarray:
    """Cubic spline through the thick-slice centres, sampled at every thin slice.

    Thick slice i averages thin slices i*k .. i*k+k-1, so it sits at thin index i*k + (k-1)/2.
    In-plane the grid is unchanged, so this is the tricubic baseline of the model ladder.
    The (k-1)/2 thin slices beyond the outermost centres are extrapolated; SUV is clipped at 0.
    """
    x = np.arange(lr.shape[0]) * k + (k - 1) / 2
    pred = CubicSpline(x, lr, axis=0)(np.arange(lr.shape[0] * k))
    return np.clip(pred, 0, None).astype(np.float32)


MODELS = {"cubic_z": cubic_z}
