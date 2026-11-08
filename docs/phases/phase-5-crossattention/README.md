# Phase 5 — Cross-Attention

## Goal

Make the CT pay for itself. In phase 4 the raw-CT U-Net lost to the no-CT U-Net at k = 2 and 3, so
before cross-attention we tested whether a better CT representation, from a pretrained foundation
model, changes that.

## What we did (commit range, configs used)

- CT-FM (Pai et al. 2025) as the frozen CT branch of the 2.5D U-Net: `p5_unet_ctfm_k*` (z-mean) and
  `p5_unet_ctfmz_k*` (z-resolved), trained at sha 80fbbb3 (log 2026-11-01, 2026-11-08).
- Native-resolution CT ablation, not yet run: `p5_ct_native_v1` builds `data/ct_native/v1` (CT at
  4 × 4 pixels per PET pixel, about 1 mm, clipped at CT-FM's window [−1024, 2048]), and
  `p5_unet_ctfm_native_k*` / `p5_unet_ctfmz_native_k*` feed it to CT-FM (log 2026-11-08 evening).

## Evidence (notebooks, figures, metric tables)

`paper/tables/main_p5_k*`, `paper/tables/stats_ctfm{,z}_vs_{noct,ct}_k*`.

## Decisions & rationale

- CT-FM replaces the CT branch rather than the backbone, so p4_unet_k* and p4_unet_noct_k* stay
  the direct comparisons (log 2026-11-01).
- CT-FM beats the raw-CT U-Net at every k but only ties the no-CT U-Net (k = 4 slightly ahead, one
  seed), so no extra seeds until the native-resolution ablation has run (log 2026-11-08).
- Native CT on an integer 4 × 4 split of the PET pixel, on the PET's thin z planes, so every
  CT-FM level pools exactly onto a U-Net level (log 2026-11-08 evening).

## Handed over to next phase
