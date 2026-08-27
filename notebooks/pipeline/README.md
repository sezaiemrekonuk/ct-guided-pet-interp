# Pipeline notebooks

The tools. A small fixed set of notebooks, each parameterised by a config YAML and run many
times with different configs:

| Notebook | Job |
| --- | --- |
| `00_inspect.ipynb` | geometry and SUV sanity check on one case |
| `01_preprocess.ipynb` | DICOM → NIfTI, SUV conversion, QC gate, CT onto the PET grid, body mask, splits |
| `02_degrade.ipynb` | frozen LR volumes by slice averaging |
| `03_train.ipynb` | trains whatever model the config names, resumes from the last epoch |
| `04_evaluate.ipynb` | evaluates a model or a baseline, writes `metrics.json` |

Experiment identity lives in the config, not in the filename — twenty experiments produce twenty
YAMLs and no new notebooks. Adding a notebook here means adding a new *kind of job*, not a new
run.

Every notebook opens with the same three cells:

```python
# 1 — bootstrap
!git clone $REPO /content/petinterp 2>/dev/null || git -C /content/petinterp pull
%pip install -qe /content/petinterp

# 2 — start the run
import petinterp
run = petinterp.start("configs/p4_unet_k2.yaml")

# 3 — do the work
petinterp.train(run)
```

Cells hold setup, config selection, calls into `src/petinterp` and display. No `def`, no
`class` — see CLAUDE.md rule 11. A helper written while experimenting moves into `src/` before
the commit.
