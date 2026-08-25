# CLAUDE.md — working rules

Governs every session in this repository. When a request conflicts with a rule here, say so
before acting.

Project: CT-guided PET through-plane slice interpolation (PSMA PET/CT). Two people, Colab-based
training, paper as the deliverable. See `README.md` for the layout.

## Problem spec

Input `PET_sparse ∈ R^(H×W×(D/k))` + `CT_full ∈ R^(H×W×D)` → output `PET_dense ∈ R^(H×W×D)`.
PSMA is primary; FDG only in the generalization experiment. Every model reports k = 2, 3 and 4.
Always apply measurement consistency, in the loss or after inference:
`pred ← pred + upsample(measured − avg_k(pred))`. No model ships without it.

## Working rules

1. **Configs, not constants.** Every run is a versioned YAML in `configs/`, prefixed `p<N>_` where
   N is its phase directory under `docs/phases/` (0–6, the only phase numbering this repo uses) —
   the 2.5D U-Net at k=2 is `p4_unet_k2.yaml`. No hard-coded parameters leak into code: paths,
   hyperparameters, seeds, degradation factors and split file names all come from the config.
   A run is reproducible from its YAML alone.

2. **Metrics in, tables out.** Every experiment writes `results/<exp_id>/metrics.json`;
   `scripts/make_tables.py` turns those into `paper/tables/` as CSV and LaTeX (booktabs). Tables
   are NEVER written by hand — a wrong number is fixed in the metrics or the generator.

3. **Figures are generated.** Comparison figures (axial / coronal / sagittal + error map + lesion
   zoom) come from `scripts/make_figures.py` into `paper/figures/` as 300-dpi PDFs. No
   screenshots, no manual cropping.

4. **`paper/references.bib` is live.** The moment a method, baseline, dataset or tool is used, its
   BibTeX entry is added — not at writing time. Datasets use the mandatory citation format on
   their TCIA / Zenodo page (data citation + publication citation + TCIA acknowledgement).

5. **Log it the day it happens, in your own voice.** `paper/experiment_log.md` gets an entry per
   run — date, exp_id, hypothesis, result, decision — and for anything else that moves the
   project: a dataset obtained, a preprocessing choice, an idea dropped, a bug that changed a
   number. Failed runs too; a negative result we can cite beats a run we forgot. Write like a
   person telling a colleague what you did and why — prose, first person, reasoning included. No
   template dumps, no status-report voice, no bullet skeletons; this is the raw material for
   Methods and Discussion.

6. **Baselines share one interface.** RIFE, FILM, SAINT and I3Net are separate modules under
   `baselines/`, all driven through `scripts/evaluate.py` (input: LR volume + CT, output: HR
   volume). Metric code exists exactly once, in `src/petinterp/evaluation/` — a baseline never
   carries its own PSNR/SSIM. Baselines are retrained under OUR averaging degradation, our split
   and our metric code, never their published decimation protocol, or the comparison is unfair.

7. **Splits and seeds are immutable.** `splits/*.json` and degradation seeds are frozen once
   written. A change means a NEW versioned file (`split_v2.json`) plus a rationale entry in the
   experiment log. Never edit an existing split file in place. Splits are 70/10/20, patient-level,
   stratified by lesion presence. Degraded test volumes are generated once and written to disk —
   never re-sampled per epoch; a fixed seed alone does not guarantee this.

8. **Everything resumes.** All training supports resume-from-checkpoint; Colab disconnects are the
   normal working mode. Checkpoints go to Drive every epoch and include optimizer state, epoch,
   RNG state and the config hash.

9. **Milestones update the docs.** At each milestone, update the corresponding phase README under
   `docs/phases/` and the relevant draft section under `paper/draft/`. Then tag
   `phase-X-complete`, X matching that same 0–6 numbering.

10. **Code by function, docs by phase.** Code is organized by function under `src/petinterp/`,
    never by phase. Phase directories under `docs/` hold narrative and evidence only, importing
    from `src/petinterp` rather than duplicating code.

## Domain guardrails

Non-negotiable. Violating one invalidates the experiment, not just the code.

- **Degradation is slice AVERAGING, never decimation.** A thick slice is the mean of the thin
  slices, as the scanner integrates counts. Decimation is a physically different problem.
- **PET intensities are absolute SUV.** Never patient-wise min-max normalization. The PSMA
  transform is `log(1+SUV)/log(1+50)`; the [0, 50] clip is an ablation, never the default, and
  [0, 15] crushes lesion peaks. Metrics are computed in SUV space after the inverse transform.
- **CT onto the PET grid, never the reverse**, with HU clipped [−1000, +1000] → [−1, 1].
  Same-session PET/CT is already aligned — grid matching only, no deformable registration.
- **Data splits are patient-level, never slice-level.** Slice-level splitting leaks neighboring
  anatomy into the test set and inflates every metric.
- **Liver SUVmean sanity check per patient**: FDG ≈ 1.5–3.0, Ga-68 PSMA ≈ 4–8. Out-of-range means
  broken SUV conversion (decay correction, injected dose, weight units) — fix the conversion
  before trusting anything downstream. On PSMA, kidneys and bladder must also be the brightest
  structures present.
- **QC gate before a patient enters a split:** affine orientation consistent, no NaN or negative
  SUV, at least 40 slices. Anything else is excluded, and the exclusion is logged.
- **Never augment along z** — the problem is defined in z. L-R flip and in-plane rotation only.
- **No adversarial / GAN loss.** Hallucinated SUV values are not clinically defensible.
- **Default loss is `0.84·MS-SSIM + 0.16·L1`**, never a pure SSIM-family loss alone (training is
  unstable); lesion voxels are weighted 5–10× or they vanish into the aggregate.
- **Report lesion-stratified metrics** — SUVmax bias, small lesions (<1 mL), bladder-adjacent
  lesions (within 2 cm of the bladder mask) — alongside global PSNR/SSIM/NRMSE computed inside
  the body mask only. Air inflates full-FOV metrics; global numbers hide the failures that matter.
  Significance is Wilcoxon signed-rank, patient-paired, Bonferroni-corrected.

## Background

`docs/problem-definition.md` is the reference framing document — clinical motivation, dataset
choices, the model ladder, literature baselines, rejected alternatives. Read it before writing
paper text or when a design decision needs justification.
