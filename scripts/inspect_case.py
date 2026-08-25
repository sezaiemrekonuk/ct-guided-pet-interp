"""Inspect one DEEP-PSMA case: print PET/CT geometry and dump orthogonal PNG views.

This is a look-at-the-data utility, not an experiment run, so it takes CLI arguments
instead of a config YAML (CLAUDE.md rule 1 governs experiment runs, which write
results/<exp_id>/metrics.json -- this script writes no metrics).

Usage:
    python scripts/inspect_case.py data/raw/train_0014 --tracer PSMA
    python scripts/inspect_case.py --smoke        # synthetic data, no real case needed
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI: render straight to file

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

# Display windows. PET is shown in absolute SUV and CT in absolute HU -- never a
# per-volume min/max stretch, which would make two patients incomparable by eye and
# is the same mistake the SUV guardrail forbids for the data itself.
PET_WINDOW = (0.0, 5.0)        # SUV
CT_WINDOW = (-1000.0, 1000.0)  # HU: air .. dense bone

VIEWS = ("axial", "coronal", "sagittal")


def load(path: Path) -> sitk.Image:
    """Read a NIfTI file into a SimpleITK Image.

    sitk.ReadImage picks the reader from the file extension (.nii.gz here) and returns
    an Image, which is a voxel buffer PLUS its geometry: origin, spacing and direction
    in physical (mm) space. That geometry is the whole point of the class -- a bare
    numpy array would forget that PET voxels are ~4 mm wide and CT voxels ~1 mm, and
    every resampling step later depends on knowing that.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    return sitk.ReadImage(str(path))


def describe(img: sitk.Image, label: str) -> dict:
    """Print and return the basic facts about a volume."""
    # GetSize() is in index order (x, y, z) -- the ITK convention.
    size = img.GetSize()
    # Physical size of one voxel in mm, same (x, y, z) order. The third number is the
    # through-plane spacing this whole project is about.
    spacing = img.GetSpacing()
    # World coordinate (mm) of voxel [0,0,0], and the 3x3 rotation (row-major, flattened)
    # mapping index axes to patient axes. PET and CT must agree on both to be aligned.
    origin = img.GetOrigin()
    direction = img.GetDirection()
    # The stored pixel type, e.g. "32-bit float". Kept separate from numpy's dtype
    # because ITK also tracks things numpy has no name for (e.g. vector pixels).
    pixel_type = img.GetPixelIDTypeAsString()

    # GetArrayViewFromImage hands back a numpy VIEW with no copy -- cheap for a 500 MB
    # volume. Note the axes are REVERSED relative to GetSize(): the array is indexed
    # [z, y, x]. Mixing the two orders up is the classic first medical-imaging bug.
    arr = sitk.GetArrayViewFromImage(img)

    info = {
        "label": label,
        "size_xyz": size,
        "array_shape_zyx": arr.shape,
        "spacing_mm_xyz": spacing,
        "origin_mm": origin,
        "direction": direction,
        "pixel_type": pixel_type,
        "numpy_dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "p99": float(np.percentile(arr, 99)),
    }

    print(f"\n=== {label} ===")
    print(f"  size (x,y,z)        : {size}")
    print(f"  numpy shape (z,y,x) : {arr.shape}")
    print(f"  spacing mm (x,y,z)  : ({spacing[0]:.4f}, {spacing[1]:.4f}, {spacing[2]:.4f})")
    print(f"  through-plane (z)   : {spacing[2]:.4f} mm  <- the axis we interpolate")
    print(f"  extent mm (x,y,z)   : "
          f"({size[0] * spacing[0]:.1f}, {size[1] * spacing[1]:.1f}, {size[2] * spacing[2]:.1f})")
    print(f"  origin mm           : ({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})")
    print(f"  direction           : {tuple(round(d, 3) for d in direction)}")
    print(f"  pixel type / dtype  : {pixel_type} / {arr.dtype}")
    print(f"  value range         : [{info['min']:.4f}, {info['max']:.4f}]  "
          f"mean {info['mean']:.4f}  p99 {info['p99']:.4f}")
    return info


def _slice_and_aspect(arr: np.ndarray, spacing_xyz, view: str):
    """Cut the mid-slice for one view and give the aspect ratio that keeps it undistorted.

    arr is [z, y, x]; spacing_xyz is (sx, sy, sz) in mm.

    imshow draws every pixel as a square, so an anisotropic volume (4 mm voxels in z,
    1 mm in x) comes out squashed unless we tell it aspect = mm-per-row / mm-per-column.
    """
    sx, sy, sz = spacing_xyz
    nz, ny, nx = arr.shape
    if view == "axial":       # fix z: rows are y, columns are x
        return arr[nz // 2, :, :], sy / sx
    if view == "coronal":     # fix y: rows are z, columns are x
        return arr[:, ny // 2, :], sz / sx
    if view == "sagittal":    # fix x: rows are z, columns are y
        return arr[:, :, nx // 2], sz / sy
    raise ValueError(view)


def save_views(img: sitk.Image, label: str, out_dir: Path, window, cmap: str) -> list[Path]:
    """Write mid-slice axial / coronal / sagittal PNGs for one volume."""
    arr = sitk.GetArrayFromImage(img)  # real copy here: we index and flip it
    spacing = img.GetSpacing()
    vmin, vmax = window
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for view in VIEWS:
        sl, aspect = _slice_and_aspect(arr, spacing, view)
        # keep the figure box the same shape as the physical slice
        rows, cols = sl.shape
        fig, ax = plt.subplots(figsize=(4, 4 * aspect * rows / cols))
        # origin="lower" puts array row 0 at the bottom, so +z (superior) points up in
        # the coronal/sagittal views instead of the patient hanging upside down.
        ax.imshow(sl, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect, origin="lower")
        ax.set_title(f"{label} {view}  [{vmin:g}, {vmax:g}]", fontsize=9)
        ax.axis("off")
        path = out_dir / f"{label}_{view}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        written.append(path)
        print(f"  wrote {path}")
    return written


def inspect(case_dir: Path, tracer: str, out_dir: Path) -> dict:
    tracer_dir = case_dir / tracer
    pet = load(tracer_dir / "PET.nii.gz")
    ct = load(tracer_dir / "CT.nii.gz")

    prefix = f"{case_dir.name}_{tracer}"
    info = {"PET": describe(pet, f"{prefix}_PET"), "CT": describe(ct, f"{prefix}_CT")}

    if pet.GetSize() != ct.GetSize() or pet.GetSpacing() != ct.GetSpacing():
        print("\n  NOTE: PET and CT are on different grids -> resample before pairing them.")

    print("\n--- writing views ---")
    save_views(pet, f"{prefix}_PET", out_dir, PET_WINDOW, "inferno")
    save_views(ct, f"{prefix}_CT", out_dir, CT_WINDOW, "gray")
    return info


def smoke() -> None:
    """Run the whole path on a synthetic 32x32x16 volume -- no real case needed."""
    # numpy is [z, y, x], so 32 x 32 x 16 in (x, y, z) is shape (16, 32, 32).
    rng = np.random.default_rng(0)
    arr = rng.random((16, 32, 32)).astype(np.float32) * 10.0
    arr[6:10, 12:20, 12:20] = 42.0  # a blob, so the views are visibly not blank

    # GetImageFromArray is the inverse of GetArrayFromImage: it also reverses the axes,
    # so this (16, 32, 32) array becomes an Image of size (32, 32, 16).
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((2.0, 2.0, 5.0))  # anisotropic on purpose: exercises the aspect math

    assert img.GetSize() == (32, 32, 16), img.GetSize()
    assert sitk.GetArrayViewFromImage(img).shape == (16, 32, 32)

    info = describe(img, "SMOKE")
    assert info["size_xyz"] == (32, 32, 16)
    assert info["spacing_mm_xyz"] == (2.0, 2.0, 5.0)
    assert abs(info["max"] - 42.0) < 1e-5

    ax, asp = _slice_and_aspect(arr, (2.0, 2.0, 5.0), "axial")
    assert ax.shape == (32, 32) and abs(asp - 1.0) < 1e-9
    cor, asp = _slice_and_aspect(arr, (2.0, 2.0, 5.0), "coronal")
    assert cor.shape == (16, 32) and abs(asp - 2.5) < 1e-9
    sag, asp = _slice_and_aspect(arr, (2.0, 2.0, 5.0), "sagittal")
    assert sag.shape == (16, 32) and abs(asp - 2.5) < 1e-9

    with tempfile.TemporaryDirectory() as tmp:
        written = save_views(img, "SMOKE", Path(tmp), (0.0, 42.0), "inferno")
        assert len(written) == 3
        for p in written:
            assert p.stat().st_size > 1000, p  # a blank/failed PNG is tiny

    print("\nsmoke test OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case_dir", nargs="?", type=Path,
                    help="case directory, e.g. data/raw/train_0014")
    ap.add_argument("--tracer", default="PSMA", choices=["PSMA", "FDG"])
    ap.add_argument("--out", type=Path, default=Path("outputs/inspect"))
    ap.add_argument("--smoke", action="store_true", help="run on synthetic data and exit")
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return
    if args.case_dir is None:
        ap.error("case_dir is required unless --smoke is given")
    inspect(args.case_dir, args.tracer, args.out)


if __name__ == "__main__":
    main()
