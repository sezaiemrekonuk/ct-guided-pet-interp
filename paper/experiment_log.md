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

---

## 2026-09-29 — Our advisor suggests looking at foundation models

In today's meeting our advisor suggested we might fine-tune a foundation model for this problem,
or build something on top of one, instead of only training our own networks from scratch. I'm
writing it down now so it doesn't get lost. We haven't decided anything and we haven't looked at
any specific model yet.

For me the open question is where a foundation model would actually fit in our ladder. It could
be a pretrained encoder that replaces the CT branch. It could be a pretrained backbone we
fine-tune for the interpolation itself. Or it could be a baseline we run as-is. Whichever it is,
the rules we already have still apply: training and testing use our averaging degradation, our
split and our metric code, measurement consistency goes on the output, and we don't use
adversarial losses. Before this goes into the plan I want a short list of candidate models, what
each was pretrained on, and whether it can take a PET/CT slab at all. The classical baseline
and the no-CT vs CT early-fusion U-Net come first either way, because a foundation model only
means something once we have those numbers to compare it against.

---

## 2026-09-30 — The first real pipeline: preprocessing, degradation, metrics and a cubic baseline

Today the repository went from rules to a pipeline you can actually run. There's a download
notebook that pulls all five DEEP-PSMA zips (about 24 GB, CC BY-NC 4.0) from Zenodo onto the
shared Drive, and behind it preprocessing, the averaging degradation, the metric code and the
first baseline. Nothing has run on the real data yet; everything below was only checked on
synthetic volumes. I'm writing the choices down now because every number we report later
depends on them.

Preprocessing reads each PSMA case straight out of the zips rather than unzipping onto Drive,
and it writes one folder per patient with `qc.json` written last, so a crash costs one patient.
CT is resampled onto the PET grid with linear interpolation (air, −1000 HU, as the fill value),
and the TTB and TotalSegmentator labels go onto the same grid with nearest neighbour. The body
mask is the largest connected component of CT above −500 HU, with holes filled slice by slice.
The QC gate excludes on PET/CT direction mismatch, non-finite or negative SUV, fewer than 40
slices, or an orientation that differs from the rest of the cohort. The liver range and the
kidney/bladder ordering are recorded as flags only, not exclusions. The liver question from
August is still open, and this run is where we'll finally see the whole-cohort distribution.

One place where I had to depart from the letter of our rules is the split. CLAUDE.md says to
stratify by lesion presence, but every DEEP-PSMA case has measurable disease, so that strata
would be empty. Instead I stratify by tertile of total tumour volume, which keeps the burden
distribution the same across train, val and test. The split is 70/10/20 at patient level, with
seed 20260930, and once written it is frozen: the code refuses to overwrite an existing split
file that differs.

For degradation, a thick slice is the mean of k thin ones, and the thin volume is cropped to a
multiple of k at the high-index end. That cropped native 2 mm volume is the ground truth. This
also corrects something I wrote on 26 August. There I said k=2 meant synthesising 1 mm slices,
but under the averaging protocol it's the other way round: we average 2 mm slices down to 2k mm
and synthesise back to the native 2 mm. So our target is a real acquisition after all.

A few metric definitions need to be stated in the paper. Global metrics are computed inside the
body mask. PSNR and SSIM use each patient's ground-truth SUVmax inside the body as the data
range. SSIM is 3D with a Gaussian window of σ = 1.5 voxels. NRMSE is the error norm divided by
the ground-truth norm. Lesions are the 26-connected components of TTB. A lesion is small below
1 mL and bladder-adjacent within 20 mm of the TotalSegmentator bladder. Every model is scored
twice, raw and after the measurement-consistency projection, so the consistency ablation comes
for free with every run.

The first baseline is a cubic spline along z through the thick-slice centres, which sit at thin
index i·k + (k−1)/2. Runs are `p3_cubic_k2`, `_k3` and `_k4`. This is the floor every learned
model has to clear, and it's also the end-to-end test of the metric code.

---

## 2026-10-11 — The first learned model: a 2.5D residual U-Net, with and without CT

Preprocessing ran on 2 October: all 100 DEEP-PSMA PSMA cases passed the QC gate, and the frozen
split came out as 70 train, 9 validation and 21 test patients (the tertile rounding gives 9/21
rather than 10/20). I copied `split_v1.json` from Drive into `splits/` today so it is in git.
Before designing the first network I looked at the cohort geometry in `qc_summary.json`, and it
is much less uniform than the one case I inspected in August. In-plane grids run from 128×128 to
512×512 and native slice spacing from 1.5 to 5 mm. So "k = 2" does not mean one physical slice
thickness. For a patient scanned at 5 mm we are restoring 5 mm slices from 10 mm ones, and for a
1.5 mm patient we restore 1.5 mm from 3 mm. I'm keeping the native grids as they are, because
resampling would change the frozen `processed/v1`. But I want results broken down by native
spacing later, because I expect the 5 mm patients to be the hard ones. SUVmax is above 50 in 91
of the 100 patients (median 82), which confirms that the log transform was the right choice. A
[0, 50] clip would have flattened the peak of nearly every patient's hottest lesion.

The network is rung 1 of the ladder, but I changed the formulation from the problem definition
to fit our averaging degradation. The plan had been to predict one missing thin slice from its
neighbours. Under averaging, though, no thin slice is missing: every one of them lies inside a
measured thick slice. So the natural unit is the block. The model gets one thick slice and
returns the k thin slices inside it, and the measurement-consistency projection then makes their
mean equal the measured value exactly. The input is a stack of 2D channels: the thick PET slice
and two thick neighbours on each side; the CT slices covering the block, plus one extra on each
side; and the cubic-spline prediction for the k target slices. The network (MONAI's BasicUNet,
7.8M parameters) predicts a correction to that cubic prediction. It uses batch normalisation instead of MONAI's default instance normalisation. Training
sees 192×192 crops, but inference runs on whole slices of up to 512×512 that are mostly air,
and per-image statistics would shift with how much air is in view. Its last layer starts at zero,
so the untrained network *is* the cubic baseline exactly, and the smoke test checks this. I like
that property. It means the first validation loss we log is the cubic baseline's, and every
epoch after it shows directly whether the network has learned anything the spline couldn't do.

Training follows the defaults we wrote down: 0.84·(1 − MS-SSIM) + 0.16·L1 in log-SUV space, with
L1 weighted 5× on TTB lesion voxels. We use AdamW at 2e-4 with cosine decay, batch 16, and
random 192×192 in-plane crops. The smaller grids are zero-padded up to 192, and 192 is also the
smallest size five-scale MS-SSIM accepts. Augmentation is a left-right flip only, with nothing
along z. This first run is a pilot: 50 epochs of 500 steps, roughly 30 passes over the training
blocks at k = 2. Checkpoints go to Drive every epoch, and the run keeps the weights with the
lowest validation loss.

The experiment is the paper's main hypothesis in its smallest form. `p4_unet_k2` (with CT) and
`p4_unet_noct_k2` (without CT) are identical except for the CT channels: same seed, same
schedule, same split. Both are scored against `p3_cubic_k2` with the existing metric code, raw
and after the consistency projection. My hypothesis is that both U-Nets beat cubic, and that CT
helps most where anatomy changes quickly along z, which means small lesions and the bladder
border. I froze this design before seeing any test number, so the test set is read once for this
comparison. Any tuning after it happens on the validation set. One known weakness is that each
arm has a single seed, so a small gap between them won't mean much until we repeat it with more
seeds. k = 3 and k = 4 follow once k = 2 works end to end.
