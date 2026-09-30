"""Volume I/O shared by every stage: atomic NIfTI writes and array <-> image round trips."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def read(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def array(img: sitk.Image) -> np.ndarray:
    """numpy copy, indexed [z, y, x]."""
    return sitk.GetArrayFromImage(img)


def like(arr: np.ndarray, ref: sitk.Image) -> sitk.Image:
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(ref)
    return img


def write(img: sitk.Image, path: Path) -> None:
    """Atomic write (rule 8): a killed runtime never leaves a truncated file under the real name."""
    tmp = path.with_name(path.name.replace(".nii.gz", ".tmp.nii.gz"))
    sitk.WriteImage(img, str(tmp), useCompression=True)
    os.replace(tmp, path)
