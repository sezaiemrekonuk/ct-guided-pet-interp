# CLAUDE.md — working rules

Governs every session in this repository. When a request conflicts with a rule here, say so
before acting.

Project: CT-guided PET through-plane slice interpolation (PSMA PET/CT). Two people, Colab-based
training, paper as the deliverable. See `README.md` for the layout.

## Working rules

1. **Configs, not constants.** Every experiment run is defined by a versioned YAML in `configs/`
   (phase-prefixed, e.g. `p1_unet_k2.yaml`). No hard-coded parameters leak into code — paths,
   hyperparameters, seeds, degradation factors and split file names all come from the config.
   A run is reproducible from its YAML alone.

2. **Metrics in, tables out.** Every experiment writes `results/<exp_id>/metrics.json`.
   `scripts/make_tables.py` reads those JSONs and generates paper tables into `paper/tables/` in
   both CSV and LaTeX (booktabs) form. Tables are NEVER written by hand — if a number is wrong,
   the fix goes into the metrics or the generator, never into the table file.

3. **Figures are generated.** Comparison figures (axial / coronal / sagittal views + error map +
   lesion zoom) are produced by `scripts/make_figures.py` into `paper/figures/` as 300-dpi PDFs.
   No screenshots, no manual cropping.

4. **`paper/references.bib` is live.** The moment a new method, baseline, dataset, or tool is
   used, its BibTeX entry is added — not at writing time. Datasets use the mandatory citation
   format stated on their TCIA / Zenodo page (data citation + publication citation + TCIA
   acknowledgement, as required).

5. **`paper/experiment_log.md` gets one entry per run:** date, exp_id, hypothesis, result,
   decision — 1–3 sentences. Failed runs are logged too; a negative result we can cite is worth
   more than a run we forgot.

6. **Baselines share one interface.** RIFE, FILM, SAINT and I3Net live under `baselines/` as
   separate modules, but all are driven through the shared `scripts/evaluate.py` interface
   (input: LR volume + CT, output: HR volume). Metric code exists exactly once, in
   `src/petinterp/evaluation/` — a baseline never carries its own PSNR/SSIM implementation.

7. **Splits and seeds are immutable.** `splits/*.json` and degradation seeds are frozen once
   written. A change means a NEW versioned file (`split_v2.json`) plus a rationale entry in the
   experiment log. Never edit an existing split file in place.

8. **Everything resumes.** All training supports resume-from-checkpoint — Colab disconnects are
   the normal working mode, not an exception. Checkpoints are written to Drive every epoch and
   include optimizer state, epoch, RNG state and the config hash.

9. **Milestones update the docs.** At each milestone, update the corresponding phase README under
   `docs/phases/` and the relevant draft section under `paper/draft/`. Then tag
   `phase-X-complete`.

10. **Code by function, docs by phase.** Code is organized by function under `src/petinterp/`,
    never by phase. Phase directories under `docs/` contain narrative and evidence only, and
    import from `src/petinterp` rather than duplicating code.

## Domain guardrails

Non-negotiable. Violating one invalidates the experiment, not just the code.

- **Degradation is slice AVERAGING, never decimation.** A thick slice is the mean of the
  corresponding thin slices, matching how the scanner integrates counts. Dropping slices models a
  different (and physically wrong) problem and makes results non-comparable to the literature.
- **PET intensities are absolute SUV.** Never patient-wise min-max normalization — it destroys
  the quantitative meaning that makes PET clinically useful. For PSMA use a wide fixed clip or a
  log transform; all metrics are computed in SUV space after the inverse transform.
- **Data splits are patient-level, never slice-level.** Slice-level splitting leaks neighboring
  anatomy from the same patient into the test set and inflates every metric.
- **Liver SUVmean sanity check per patient**: FDG ≈ 1.5–3.0, Ga-68 PSMA ≈ 4–8. Out-of-range means
  broken SUV conversion (decay correction, injected dose, weight units) — fix the conversion
  before trusting anything downstream.
- **Report lesion-stratified metrics** — SUVmax bias, small lesions (<1 mL), bladder-adjacent
  lesions — alongside global PSNR/SSIM. Global metrics are dominated by background and hide
  exactly the failures that matter clinically.
