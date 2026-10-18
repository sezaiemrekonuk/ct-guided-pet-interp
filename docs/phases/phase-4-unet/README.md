# Phase 4 — U-Net

*Status: in progress. k = 2 is done with three seeds per arm; k = 3 and 4 have one seed per arm,
and seeds 2 and 3 at both factors are set up but not started.*

## Goal

Rung 1 of the model ladder: a 2.5D residual U-Net that restores thin PET slices from thick ones,
trained with and without the co-registered CT. It answers two questions. Does a learned model
beat the cubic baseline, especially on lesion SUVmax? And does CT help when it is simply stacked
onto the input channels (early fusion)? This is the reference point for the cross-attention
model in phase 5.

## What we did (commit range, configs used)

- `e6bd683`: model, training loop and configs. `src/petinterp/data/blocks.py` builds the
  block-wise inputs. `src/petinterp/models/unet25d.py` is the residual MONAI BasicUNet (7.8M
  parameters, BatchNorm, last layer zero-initialised so the untrained net equals `cubic_z`).
  `src/petinterp/losses/msssim_l1.py` is 0.84·MS-SSIM + 0.16·L1, with L1 weighted 5× on TTB
  voxels. `src/petinterp/train.py` checkpoints every epoch and resumes.
- Formulation: one thick slice in, its k thin slices out. The input is 5 thick PET slices, the
  k + 2 thin CT slices around the block (CT arm only), and the k cubic-spline slices. The output
  is a residual on the spline. The measurement-consistency projection is applied after inference.
- Pilot configs: `p4_unet_k2` (CT) and `p4_unet_noct_k2` (no CT). Both use seed 20261011,
  50 epochs × 500 steps, batch 16, AdamW 2e-4 with cosine decay, and a 192² crop with L-R flip.
  They are evaluated with `p3_cubic_k2` on the 21 test patients.
- `3102f80`–`5ac6e9d`: seed replicates `p4_unet_k2_s2/_s3` and `p4_unet_noct_k2_s2/_s3`.
  Every run now archives its executed notebook to `runs/<exp_id>/notebook.html`.
- k = 3 and 4 pilots: `p4_unet_k3`, `p4_unet_noct_k3`, `p4_unet_k4`, `p4_unet_noct_k4`, the k = 2
  pilot configs with only `k` changed (seed 20261011). Results in `1a037fd`.
- Seed replicates at k = 3 and 4: `p4_unet{,_noct}_k{3,4}_s2/_s3` (seeds 20261012/13), queued
  in `03_train.ipynb` k = 4 first, because k = 4 is where the one-seed CT comparison is undecided.

## Evidence (notebooks, figures, metric tables)

- `paper/tables/main_k2.csv` / `.tex`: global and lesion-stratified metrics, raw and
  after consistency, generated from `results/<exp_id>/metrics.json` by
  `scripts/make_tables.py`.
- `paper/figures/learning_curves_k2.pdf`: validation loss per epoch for each arm, generated
  from `results/<exp_id>/train_history.json` by `scripts/make_figures.py`.
- Pilot result (after consistency): both U-Nets beat cubic in 19–21 of 21 patients on PSNR,
  SSIM and NRMSE (Wilcoxon, Bonferroni-corrected, p ≤ 0.008). Lesion SUVmax bias drops from
  −13% (cubic) to −2% (no CT) and −3.5% (CT); for lesions under 1 mL, from −16% to −5.5% and −7%.
  The CT arm is *not* better than the no-CT arm: −0.29 dB median PSNR, p = 0.09 after
  correction. Its validation loss is also slightly higher and noisier over the whole run.
- `paper/tables/main_k3`, `main_k4` and `paper/figures/learning_curves_k3`, `_k4`: the same
  tables and curves for the one-seed pilots at k = 3 and 4.
- Pilot result at k = 3 and 4 (after consistency, one seed per arm): the no-CT U-Net beats cubic
  by 2.4 dB at k = 3 and about 2 dB at k = 4, and lesion SUVmax bias drops from −24% to −8% (k = 3)
  and from −35% to −12.5% (k = 4). Small lesions keep a large residual (−15% and −24%). Without
  CT the U-Net is better at k = 3 (19/21 patients, corrected p = 0.001); at k = 4 the two arms are
  level (corrected p = 1). For the U-Net the consistency step costs 1–2 points of lesion SUVmax
  at every k while adding 0.2–0.4 dB PSNR.

## Decisions & rationale

- **Block formulation, not single-slice interpolation.** Under averaging degradation no thin
  slice is missing; every thin slice sits inside a measured thick slice. Predicting the whole
  block lets consistency hold exactly and gives the network the slice's position within the
  block for free.
- **Residual on the spline, zero-initialised.** The untrained model is the baseline, so the
  "epoch −1" validation loss is the cubic reference, and any gain is attributable to learning.
- **BatchNorm instead of InstanceNorm.** Training sees 192² crops but inference sees whole
  slices up to 512²; per-image normalisation statistics would shift with the amount of air.
- **The test set is read once per frozen design.** Model selection uses the validation loss
  (best checkpoint); hypotheses are written in the experiment log before results.

## Handed over to next phase

*To be completed when the seed replicates and k = 3, 4 are in.*
