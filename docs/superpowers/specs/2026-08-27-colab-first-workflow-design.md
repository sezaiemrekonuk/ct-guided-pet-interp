# Colab-first working environment — design

**Date:** 2026-08-27
**Status:** approved, not yet implemented
**Scope of this document:** the working contract (where code runs, where data lives, how a run
survives a crash). It does not specify any preprocessing, model or evaluation behaviour.

## Problem

Two people develop this project. The primary machine is an M2 Air with neither the compute nor
the storage this project needs; the secondary machines are an RTX 4070 laptop and a partner's
RTX 5080. Runs are long and expensive, and Colab sessions end for reasons outside our control —
quota exhaustion, idle timeouts, the 12-hour ceiling, plain disconnects. A run that must restart
from zero after a disconnect is not merely slow, it is unaffordable.

So the environment has to satisfy three things at once:

1. Colab is the reference execution environment, and results are read off a notebook.
2. Google Drive is the root of all heavy I/O, shared between two people.
3. No long-running stage may be one-shot. Everything resumes.

## Decisions

### D1 — Code reaches Colab by `git clone`, never by living on Drive

The notebook's first cell clones (or pulls) the repository from GitHub and installs it editable.
Drive holds data, checkpoints and run archives; it holds no code.

*Rejected:* keeping the repository on Drive and adding it to `sys.path`. It makes editing in
Colab convenient, but the Drive copy and the git history drift apart, two people's Drives
collide, and the question "which code produced this number?" becomes unanswerable — which is
exactly the question a paper has to answer.

*Failure handling:* the clone step is idempotent (`clone || pull`), so re-running it after a
crash costs seconds. A code fix goes Mac → commit → push → re-run the setup cell → restart the
runtime. This is marginally slower than editing a cell in place, and it preserves provenance.

### D2 — Two notebook layers: `pipeline/` (tools) and `analysis/` (paper assets)

`notebooks/pipeline/` holds a small fixed set of notebooks, each parameterised by a config YAML
and run many times with different configs. Experiment identity lives in the YAML, not in the
notebook filename, so twenty experiments produce twenty YAMLs and no new notebooks.

`notebooks/analysis/` holds one notebook per paper section or asset. These read from `results/`,
allow interactive exploration while a figure or table is being shaped, and call the generators in
`scripts/` rather than reimplementing them. They are created when the corresponding result
exists, never pre-created empty.

*Rejected:* one notebook per experiment. It duplicates orchestration code, drifts by
copy-paste, makes the config YAML pointless and guarantees merge conflicts between two people.

*Rejected:* one notebook per phase. A phase is a moment in time and a notebook is a tool; the
repository already separates these (README, "Functional code vs. phase documentation") and this
would re-merge them.

### D3 — Per-run detail lives in a run archive, not in committed notebook outputs

Notebooks are committed with outputs stripped. The record of what a specific run did lives in
`runs/<exp_id>/` on Drive: the config that produced it, the git SHA, a dirty-tree flag, an
append-only log, checkpoints, `metrics.json`, and an HTML export of the executed notebook.

This is strictly more detail than a committed notebook carries, and it is indexed by `exp_id`
rather than buried in a diff. Committed notebook outputs would bloat the repository with embedded
PNGs, produce unreadable diffs on every run, and create a second source of truth alongside
`results/` and `paper/figures/`.

`metrics.json` is deliberately duplicated: it is written next to the run on Drive and also
committed under `results/<exp_id>/`. It is a small file, and the table generator must be able to
run from a git checkout alone, with no Drive access.

### D4 — Drive is the I/O root; configs never contain machine paths

Configs address everything relative to a root. The root itself comes from `paths.local.yaml`,
which is gitignored and differs per machine: the shared Drive folder on Colab, a local directory
on the 4070/5080, a small sample directory on the Mac. The same config therefore runs unmodified
on all three.

Drive layout under the shared `PETInterp/` folder:

```
PETInterp/
  data/raw/                     untouched DICOM / archives
  data/nifti/                   sub-XXXX/{CT,PET,seg}.nii.gz
  data/processed/v1/            resampled onto the PET grid  (+ manifest.json)
  data/degraded/v1/             frozen LR volumes            (+ manifest.json)
  runs/<exp_id>/
    config.yaml                 snapshot of the config that ran
    provenance.json             git SHA, config hash, dirty flag, GPU, start/end
    log.txt                     append-only across restarts
    checkpoints/                last.pt, best.pt
    metrics.json
    notebook.html               export of the executed notebook
  cache/wheels/                 pip wheel cache, to cut fresh-runtime setup time
```

`processed/` and `degraded/` use explicit versions (`v1`, `v2`), matching the `split_v2.json`
convention already in CLAUDE.md rule 7, rather than content hashes — a human has to be able to
read a path and know what it is. Each version directory carries a `manifest.json` recording the
config snapshot, its hash, the date, the patient count and any exclusions. A written version is
frozen; a change creates the next version plus an experiment-log entry.

### D5 — Resume is two mechanisms, never conflated

| Kind | Stages | Mechanism | Worst-case loss on a crash |
| --- | --- | --- | --- |
| Splittable | preprocessing, degradation, inference, metrics | one output file per patient; existing outputs are skipped | one patient |
| Stateful | training | per-epoch checkpoint: model, optimizer, scheduler, epoch, RNG state, config hash | one epoch |

For splittable stages the filesystem *is* the checkpoint — no checkpoint format is needed, which
is why this is the mechanism to prefer wherever a stage can be split at all.

**Every write is atomic:** write to `<name>.tmp`, then `os.replace` onto the final name. Without
this, a runtime killed mid-write leaves a truncated file that *exists*, is therefore skipped on
the next pass, and silently enters training as corrupt data. This single line closes the most
dangerous failure mode in the whole design.

**Chunked training.** The config carries `max_session_hours`. When it elapses the training loop
stops cleanly, writes a checkpoint and reports progress. Re-running the same cell continues from
there. A 300-epoch run therefore spreads itself across as many sessions as it needs, with no
special handling for Colab's session ceiling.

### D6 — `petinterp.start()` is the single entry point

```python
run = petinterp.start("configs/p4_unet_k2.yaml")
```

`start()` mounts Drive, resolves the root from `paths.local.yaml`, loads and hashes the config,
reads the git SHA and working-tree cleanliness, opens `runs/<exp_id>/`, writes
`provenance.json`, attaches the logger to `log.txt` in append mode, and locates any existing
checkpoint. On the Mac, where there is no Drive, the same call resolves to the local root.

*Rejected:* having notebooks shell out to `!python scripts/train.py`. Output capture is poor,
intermediate state cannot be inspected, and interactive figures are impossible — which defeats
the point of working in a notebook at all.

*Rejected:* doing mount/path/config work explicitly in each notebook's cells. The same forty
lines would be copied into five notebooks; one gets fixed and four rot. With two people this is
not a risk, it is a certainty.

The abstraction is justified because what it removes is genuine repetition, and because the
resume logic — the most correctness-critical part of the environment — then exists in one
testable place instead of five notebooks.

### D7 — Notebooks orchestrate; they never implement

Cells contain setup, config selection, calls into `src/petinterp`, and display. Defining a
function or a class in a notebook cell is forbidden. Logic written in a notebook cannot be
tested, cannot be reviewed and cannot be shared between two people; a temporary definition
written while experimenting moves into `src/` before the commit.

This extends the existing single-source rules (CLAUDE.md 2, 3, 6) to notebooks: an analysis
notebook *calls* `make_tables` / `make_figures` / the metric functions, and never re-derives
them.

### D8 — GPU only; no TPU

The stack is PyTorch + MONAI. TPU support requires `torch_xla`, whose coverage of 3D convolutions
and custom losses is poor and whose debugging story is worse. T4 (16 GB) is sufficient for the
2.5D U-Net; L4 or A100 makes the cross-attention model comfortable. No TPU code path is opened.

### D9 — Machine roles

| Machine | Role |
| --- | --- |
| M2 Air (primary) | development, tests, synthetic and single-case smoke runs, paper writing. No real training. |
| Colab GPU | reference environment. All real runs. |
| RTX 4070 / 5080 | optional; runs the same configs with a different `paths.local.yaml`. Not required to work for a result to be valid. |

## Repository changes this design implies

```
notebooks/pipeline/        00_inspect, 01_preprocess, 02_degrade, 03_train, 04_evaluate
notebooks/analysis/        one notebook per paper section, created on demand
src/petinterp/runtime.py   start(), path resolution, run archive, resume helpers
paths.local.example.yaml   committed template; copied to gitignored paths.local.yaml
.pre-commit-config.yaml    nbstripout, so notebook outputs never reach a commit
```

CLAUDE.md gains two rules (Colab as the execution surface; Drive as the I/O root), rewrites
rule 8 to cover every long stage rather than training alone, and picks up small amendments to
rules 1, 2, 3, 6 and 10. README's directory map and setup instructions are updated to match.

## Out of scope

No data is touched and no pipeline code is written under this design. Implementing
`runtime.py`, the notebooks themselves, and the preprocessing they call is separate work,
planned separately.

## Open items

None. Both items raised on 2026-08-27 were resolved the same day:

- **Cloning in Colab.** The repository is public, so the setup cell clones anonymously and no
  token is involved. Should it ever go private, the route is a fine-grained GitHub token scoped
  to this repository with `Contents: Read` only, stored in Colab Secrets and read with
  `google.colab.userdata`. Because the repository is public, nothing identifying may be
  committed — `.gitignore` already blocks the volume formats, and patient IDs, hospital paths and
  institutional notes must stay out of configs, logs and the experiment log.
*(Resolved 2026-08-27: output stripping goes through a committed `.pre-commit-config.yaml`
running `nbstripout`. The `.gitattributes` filter was dropped rather than kept alongside it —
two mechanisms doing the same job is how one of them silently stops working.)*
