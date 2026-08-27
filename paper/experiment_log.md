# Experiment Log

Prose, first person, written the day it happens. Runs, datasets, preprocessing choices, dropped
ideas, bugs that changed a number — all of it. Newest entry at the bottom.

---

## 2026-08-26 — First look at a real case, and a liver SUV that doesn't fit our guardrail

Ran `scripts/inspect_case.py` on `train_0014`, the one DEEP-PSMA case we have locally. It ships
both tracers for the same patient, plus TotalSegmentator labels and TTB contours, so I could run
the liver SUV sanity check before writing any preprocessing code.

Geometry first: PET is 200×200×504 at 4.07×4.07×2.0 mm, CT is 512×512×1007 at 0.98×0.98×1.0 mm.
Different grids, as expected — this is the case for resampling CT down onto the PET grid rather
than the reverse, which is what we already committed to. Through-plane spacing is 2 mm on the PET,
so k=2 means synthesising to 1 mm and k=4 to 0.5 mm. Worth remembering that our "thin" target is
itself an interpolation target, not a native acquisition.

Then the guardrail check, using the TotalSegmentator labels resampled onto the PET grid:

| organ | PSMA SUVmean | FDG SUVmean |
| --- | --- | --- |
| liver | 1.62 | 1.71 |
| kidneys | 11.07 / 10.63 | 2.06 / 2.07 |
| bladder | 19.58 | 23.20 |
| aorta | 0.78 | 1.05 |

FDG liver at 1.71 sits inside the 1.5–3.0 range we wrote down, so the SUV conversion itself is
working. PSMA liver at 1.62 is well below the 4–8 we put in CLAUDE.md. Since both tracers come
from the same patient through the same conversion path, and one passes, I don't think this is a
broken decay correction or a weight-units bug — those would sink FDG too.

The PSMA uptake *pattern* is right: kidneys ~11 and bladder ~19.6 are by far the brightest
structures, which is the renal-excretion signature we expect and the second half of our PSMA QC.
What's off is only the absolute liver level.

One more piece of evidence: the shipped `threshold.json` gives PSMA a flat `suv_threshold` of
3.0, while FDG's 2.258 is about 1.32× that patient's liver mean. So DEEP-PSMA anchors FDG to the
liver but uses a fixed threshold for PSMA — they don't treat liver as the PSMA reference either.

**Decision: change nothing yet.** One patient cannot distinguish "this man has low hepatic uptake"
from "our 4–8 range is wrong for this cohort's tracer." Widening a guardrail to make a single case
pass is exactly how a QC gate stops catching anything. When the full DEEP-PSMA download lands I
will compute liver SUVmean across every patient and look at the distribution; if the cohort
centres well below 4 then the range in CLAUDE.md gets revised, with the histogram as the reason.
Until then `train_0014` is flagged, not excluded.

---

## 2026-08-27 — Settling how we actually work: Colab first, Drive as the root, nothing one-shot

No run today, but a decision that shapes every run after it. The setup we had written down
assumed a machine that doesn't exist. My primary machine is an M2 Air — I'm on it maybe 90% of
the time — and it has neither the compute nor the storage this project needs. The 4070 laptop is
better and Seçkin's 5080 is better still, but neither of us wants to be the person who has to be
sitting at a particular machine for the project to move. So Colab is the reference environment,
and I spent the session working out what that actually implies rather than just asserting it.

The first question was how code gets into a Colab session. The tempting answer is to put the
repository on Drive and add it to `sys.path`, because then you can edit a file in Colab and
re-run. I decided against it. Two Drives drift, the Drive copy stops matching the git history,
and six months from now, writing the results section, I cannot answer "which code produced this
number?" — which is the one question a paper has to be able to answer. So the notebook clones
from GitHub and installs editable. A code fix goes Mac → commit → push → re-run the setup cell.
Slightly slower than editing in place; buys provenance.

The second question was what `notebooks/` looks like. My first instinct was one notebook per
pipeline stage, parameterised by a config YAML — six notebooks forever, twenty experiments as
twenty YAMLs. That's right for the tooling, but I pushed back on it initially because I was
worried about the paper: if the same six notebooks get re-run constantly, where does "what we saw
on run 14" live? The answer turned out not to be "in the notebook file". Each run writes a
`runs/<exp_id>/` archive on Drive — config snapshot, git SHA, dirty flag, append-only log,
checkpoints, metrics, and an HTML export of the executed notebook. That's more detail than a
committed notebook carries and it's indexed by exp_id instead of buried in a diff. Notebooks are
committed with outputs stripped. Separately, `notebooks/analysis/` holds one notebook per paper
section — that's the interactive surface for poking at results and shaping a figure, and it was
genuinely missing from my first sketch.

The part I care most about is resume. Our runs are long and Colab sessions end for reasons that
have nothing to do with us: quota, idle timeout, the 12-hour ceiling, plain disconnects. A stage
that has to restart from zero isn't slow, it's unaffordable. Rule 8 only covered training, which
is not enough — converting and resampling a hundred patients is hours too. So it now covers every
long stage, with two mechanisms kept deliberately separate. Splittable stages write one file per
patient and skip what exists; the filesystem is the checkpoint and no checkpoint format is
needed. Training checkpoints per epoch and stops cleanly at `max_session_hours`, so a 300-epoch
run spreads itself across sessions without any special handling.

The detail I'd have missed if I hadn't thought about it explicitly: without atomic writes, a
runtime killed mid-write leaves a truncated file that *exists*. The next pass sees it, skips it,
and it enters training as silently corrupt data — the worst kind of bug, because nothing fails,
the numbers are just quietly wrong. Write to `.tmp`, `os.replace` onto the final name. One line,
and it closes the most dangerous failure mode in the whole setup.

Two smaller calls. TPU is out: PyTorch + MONAI on `torch_xla` means poor 3D-conv coverage and a
miserable debugging story, and a T4 is enough for the 2.5D U-Net anyway. And notebooks may not
define functions or classes — logic in a cell can't be tested, can't be reviewed and can't be
shared between two people. Temporary definitions while experimenting are fine; they move to
`src/` before the commit.

Deliberately not done: no `runtime.py`, no notebooks, no data touched. This was about laying the
ground rules, and the full reasoning — including the alternatives I rejected — is written up in
`docs/superpowers/specs/2026-08-27-colab-first-workflow-design.md`. CLAUDE.md picked up rules 11
and 12, and rule 8 was rewritten.

Later the same day I settled output stripping on `pre-commit` with the `nbstripout` hook, and
dropped the `.gitattributes` filter I had added alongside it. Both do the same job; keeping both
means one of them quietly stops working and nobody notices which. The config is committed, so
Seçkin gets it with a `pre-commit install` instead of a setup instruction he has to remember.
