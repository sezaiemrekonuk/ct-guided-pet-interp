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

## Functional code vs. phase documentation

The repository has two parallel views of the same work, and they must not be mixed:

- **Code lives in `src/petinterp/`, organized by function** (`data/`, `models/`, `losses/`,
  `evaluation/`) — never by phase. A phase is a moment in time; a module is a responsibility.
  Reorganizing code by phase would leave seven copies of the same dataloader.
- **The phase-by-phase narrative lives in `docs/phases/`**, written for our advisor and for the
  paper. Each phase directory holds goal, what we did, evidence, decisions and handover — prose,
  figures, and metric tables only. Phase docs *import from* `src/petinterp`; they never duplicate
  code. If a phase doc needs a snippet, it references the module and function by path.

## Directory map

```
CLAUDE.md              working rules for all sessions (read this first)
README.md              this file
src/petinterp/         the installable package — all functional code
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
```

## Phase completion convention

A phase is done when its `docs/phases/phase-X-*/README.md` is filled in and the corresponding
paper draft section is updated. Completion is then marked with an annotated git tag:

```
git tag -a phase-2-complete -m "Phase 2 — degradation model frozen"
```

Tags are `phase-X-complete` (`phase-0-complete` … `phase-6-complete`). The tag is what a phase
README's "commit range" field points at, so the narrative and the history stay linked.
