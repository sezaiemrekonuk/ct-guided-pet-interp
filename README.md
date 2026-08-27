# CT-Guided PET Through-Plane Slice Interpolation

PET/CT acquisitions give a high-resolution CT volume and a PET volume that is coarse in the
through-plane (z) direction. This project synthesizes the missing PET slices from sparse PET
slices plus the co-acquired full-resolution CT used as an anatomical prior, so that a thick-slice
PET volume can be restored to thin-slice resolution without additional scan time or dose. The
primary target is PSMA PET/CT (prostate cancer). We develop on public data
(TCIA PSMA-PET-CT-Lesions, DEEP-PSMA/Zenodo, autoPET FDG) and later fine-tune and validate on
institutional data. Training runs on Google Colab with Drive as persistent storage, so every run
must be resumable. The project output is a paper, and paper assets accumulate as a by-product of
the experiments rather than being assembled at the end.

## Functional code, notebooks, and phase documentation

The repository has three parallel views of the same work, and they must not be mixed:

- **Code lives in `src/petinterp/`, organized by function** (`data/`, `models/`, `losses/`,
  `evaluation/`) — never by phase. A phase is a moment in time; a module is a responsibility.
  Reorganizing code by phase would leave seven copies of the same dataloader.
- **The phase-by-phase narrative lives in `docs/phases/`**, written for our advisor and for the
  paper. Each phase directory holds goal, what we did, evidence, decisions and handover — prose,
  figures, and metric tables only. Phase docs *import from* `src/petinterp`; they never duplicate
  code. If a phase doc needs a snippet, it references the module and function by path.
- **Notebooks in `notebooks/` orchestrate, they never implement.** They are where a run is
  launched and where its output is read, split into `pipeline/` (config-driven tools) and
  `analysis/` (paper assets). A cell holds setup, a config choice, a call into `src/petinterp`
  and display — never a `def` or a `class`. Logic in a notebook cannot be tested, reviewed or
  shared, so it moves to `src/` before the commit.

## Directory map

```
CLAUDE.md              working rules for all sessions (read this first)
README.md              this file
notebooks/             the execution surface — cells orchestrate, they never implement
  pipeline/            config-driven tools: inspect, preprocess, degrade, train, evaluate
  analysis/            one notebook per paper section or asset, created on demand
src/petinterp/         the installable package — all functional code
  runtime.py           start(): Drive mount, path resolution, run archive, resume
  data/                dataset loading, SUV conversion, resampling, degradation
  models/              network architectures
  losses/              loss functions
  evaluation/          metric implementations (each metric exists exactly once)
baselines/             RIFE / SAINT / I3Net (+ FILM) as separate modules,
                       each behind the shared evaluate.py interface
configs/               experiment YAMLs, phase-prefixed: p4_unet_k2.yaml
scripts/               make_tables.py, make_figures.py, evaluate.py
splits/                frozen patient-level split JSONs (immutable)
results/               one subdir per exp_id, each with metrics.json
paper/
  tables/              generated CSV + LaTeX tables (never hand-written)
  figures/             generated 300-dpi PDFs
  draft/               paper sections
  references.bib       kept live, updated the moment a method/dataset/tool is used
  experiment_log.md    one entry per run
docs/phases/
  phase-0-setup/  phase-1-preprocessing/  phase-2-degradation/
  phase-3-baselines/  phase-4-unet/  phase-5-crossattention/  phase-6-evaluation/
docs/superpowers/specs/ design documents for cross-cutting decisions
paths.local.example.yaml  template; copy to paths.local.yaml (gitignored) per machine
```

Nothing heavy lives in the repository. Data, checkpoints and run archives live on Google Drive
(see below); the repository holds code, configs, small metric JSONs and paper text.

## Where things run, and where things live

Colab is the reference environment: every real run starts from a notebook under
`notebooks/pipeline/`, on a Colab GPU. TPU is not supported. The M2 Air is for development,
tests and smoke runs; the RTX 4070 / 5080 machines are optional and run the same configs with a
different `paths.local.yaml`.

Code reaches Colab by `git clone`, never by living on Drive — so the question "which code
produced this number?" always has an answer. Drive holds everything heavy:

```
PETInterp/                      shared Drive folder, one copy for both of us
  data/raw/                     untouched DICOM / archives
  data/nifti/                   sub-XXXX/{CT,PET,seg}.nii.gz
  data/processed/v1/            resampled onto the PET grid   (+ manifest.json)
  data/degraded/v1/             frozen LR volumes             (+ manifest.json)
  runs/<exp_id>/                config.yaml, provenance.json, log.txt,
                                checkpoints/, metrics.json, notebook.html
  cache/wheels/                 pip wheel cache
```

`data/processed/` and `data/degraded/` are versioned and frozen the same way split files are: a
change makes `v2`, it never edits `v1`.

### Setting up a new machine

1. Clone the repository.
2. `cp paths.local.example.yaml paths.local.yaml` and set the root for this machine.
3. On Colab, add a shortcut to the shared `PETInterp/` folder in My Drive so it mounts at
   `/content/drive/MyDrive/PETInterp`.
4. Run `nbstripout --install` once, so notebook outputs are stripped on commit.

### Long runs and disconnects

Nothing here is one-shot. Splittable stages (preprocessing, degradation, inference, metrics)
write one file per patient and skip what already exists; training checkpoints every epoch and
stops cleanly at `max_session_hours`. Re-running the same cell continues where it left off. All
writes are atomic, so a killed runtime never leaves a truncated file that later looks complete.

## Phase completion convention

A phase is done when its `docs/phases/phase-X-*/README.md` is filled in and the corresponding
paper draft section is updated. Completion is then marked with an annotated git tag:

```
git tag -a phase-2-complete -m "Phase 2 — degradation model frozen"
```

Tags are `phase-X-complete` (`phase-0-complete` … `phase-6-complete`). The tag is what a phase
README's "commit range" field points at, so the narrative and the history stay linked.
