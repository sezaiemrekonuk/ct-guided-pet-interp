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

Later the same day: the pilot ran end to end on a free Colab T4, and the result is not the one I
predicted. Degradation took about eight minutes for all three k. Caching the 79 training and
validation patients to the Colab disk took half an hour; most of that is reading NIfTI files
through the Drive mount. After that, each epoch took about 87 s with CT and 71 s without, so each
model needed a little over an hour. All runs come from commit e6bd683 on a clean tree. The cubic
baseline's validation loss was 0.0028 (the untrained network's "epoch -1" line). Both U-Nets
dropped below it in the first epoch and then levelled off around 0.0020–0.0021 from about epoch
15 on: 0.00204 at best for the CT model (epoch 41), and essentially the same for the no-CT
model. More epochs would not have changed much.

On the 21 test patients (numbers after the consistency projection, in `paper/tables/main_k2`),
both networks beat the cubic spline everywhere we measure. Patient-paired Wilcoxon tests,
Bonferroni-corrected over nine comparisons, give p ≤ 0.008 for PSNR, SSIM and NRMSE, with the
U-Net ahead in 19–21 of 21 patients. The lesion numbers matter more. The cubic spline
underestimates lesion SUVmax by 13% on average, 16% for lesions under 1 mL, and 7% next to the
bladder. The U-Nets bring that down to 2–3.5%, 5.5–7% and about 1%. The spline smears every
peak into its neighbours, and the network has learned to put the peak back. Measurement
consistency also improved PSNR and NRMSE for all three models in every single patient. It
mattered most for the spline (+0.8 dB) and less for the networks (+0.3 dB), because they already
come close to averaging back to the measured slices.

The surprise is CT. The model that sees the CT was not better than the one that doesn't; if
anything it was slightly worse: a median of 0.29 dB lower PSNR, ahead in only 6 of 21 patients,
and −7.1% against −5.5% SUVmax bias on small lesions. The PSNR difference has p = 0.01 before
correction but 0.09 after, so I can't call it significant. It definitely isn't the improvement we
hypothesised either. I see three explanations worth testing. First, it could be noise: each arm
has one seed, and the two validation curves are almost on top of each other. Second, the network
may simply have no use for CT in this formulation: it already has the measured thick slice and
the spline, and PSMA uptake follows anatomy much less tightly than FDG does. Third, early fusion
may be the wrong way to give it the CT, which is exactly the argument the problem definition
makes for a separate CT encoder with cross-attention. The first explanation has to go before the
others mean anything, so the next step is two more seeds per arm at k = 2. After that come k = 3
and 4, where thicker slices leave more to infer and the anatomy might start to pay off.

Two things I'll need for the write-up. Global PSNR and SSIM are nearly saturated (SSIM ≥ 0.998
for every model), because the data range is each patient's SUVmax, which in PSMA usually means
the bladder or kidneys. The lesion table is where models actually separate, which is what we
expected when we made it the main table. Also, I ran the no-CT training and the evaluation as
extra cells in the same Colab session (to avoid a second Drive authorisation), so the committed
notebooks don't show those two calls. The calls were `train(rt.start('configs/p4_unet_noct_k2.yaml'))`
and `evaluate(rt.start(c))` for the three configs; the configs and the commit fully determine the
runs.

---

## 2026-10-11 (evening) — Is "CT doesn't help" real? Two more seeds per arm

The afternoon's surprise rests on one training run per arm, so before I build anything on it I
want to know how much of the gap is just the seed. I'm repeating both arms twice more at k = 2:
`p4_unet_k2_s2`/`_s3` and `p4_unet_noct_k2_s2`/`_s3`, identical to the pilot configs except for
`train.seed` (20261012 and 20261013; the pilot used 20261011). The seed changes the weight
initialisation, the order of the training blocks and the random crops and flips, and nothing
else. The data, split, degradation and test set stay fixed. The runs go in pairs, CT then
no-CT for seed 2 and then for seed 3, each evaluated straight after it trains. That way, if
Colab's quota runs out halfway, what's finished is still a matched pair.

What I'll look at: for each arm, the mean and spread of the test metrics across the three seeds,
and whether the CT-vs-no-CT difference keeps the same sign in every seed. I'll also average each
patient's metrics over the three seeds and rerun the patient-paired Wilcoxon test on those
averages, which takes most of the seed noise out of the comparison. My expectation, written down
before seeing anything: the two arms will overlap, and early-fused CT will turn out to make no
measurable difference at k = 2. If the no-CT arm wins all three seeds, that is a stronger and
more interesting finding, and I'll have to explain it.

From today on, every run also archives its executed notebook as `runs/<exp_id>/notebook.html`.
Rule 11 asked for this from the start, but nothing implemented it until now.

Later that night: the free Colab GPU quota ran out partway through the replicates. The second CT
seed (`p4_unet_k2_s2`) finished and is evaluated. The second no-CT seed got to epoch 12 of 50
before Colab disconnected the runtime at about 20:00 UTC, and reconnecting now gives "cannot
connect to GPU backend due to usage limits". Its checkpoints up to epoch 12 are on Drive, so it
resumes rather than restarts. The third seed of each arm hasn't started. In total the session
gave us about five and a half hours of T4 time.

One data point so far, and I'm not drawing conclusions from it: the second CT seed lands almost
exactly on the first. PSNR after consistency is 57.18 dB against 56.96, SUVmax bias −3.4%
against −3.5%, small lesions −7.4% against −7.1%, and the best validation loss is 0.00204 in
both. The pilot's no-CT run (57.43 dB, −2.0%, −5.5%) is still ahead of both CT seeds. If the
seed spread really is this narrow, the gap between the arms may turn out to be real, but that
needs the no-CT seeds before I say anything.

To make resuming a one-click job, I turned what I'd been typing into the Colab session into the
committed `03_train.ipynb`. It now loops over a `CONFIGS` list and, for each run, trains (or
resumes), evaluates and archives the executed notebook. The list holds the three runs still
open. One provenance detail: when `p4_unet_noct_k2_s2` resumes, its `provenance.json` will name
the newer commit, but no file under `src/` has changed since `5ac6e9d`, where it started, and its
append-only `log.txt` keeps both start lines.

---

## 2026-10-12 — The PSMA liver question, closed: it's the patients, not the conversion

In August I left one question open: `train_0014`'s PSMA liver came out at 1.62 against the 4–8
range in our guardrail, and I said I'd decide once I could see the whole cohort. Preprocessing on
2 October gave me that: 40 of 100 PSMA livers fall outside 4–8, 39 below and one above. That was
too many to wave through, so before touching the guardrail I wanted to know whether it was the
patients or our SUV conversion. I wrote a cohort check for it (`suv_check`, run
`p1_qc_suv_deep_psma`, commit `f075b00`, clean tree, CPU runtime on the shared root). It only
reads the zips and changes no data.

The first test is the paired FDG scan. FDG liver lands in its 1.5–3.0 range for 96 of 100
patients, and for 38 of the 39 with a low PSMA liver. The one exception is `train_0067`, with an
FDG liver of 1.26. That clears anything tied to the patient, like a weight entered in the wrong
units. It can't clear errors tied to the PSMA scan itself, such as a wrong injected dose or decay
time, because the two tracers are separate acquisitions. So I need to correct my August entry,
where I said both tracers go "through the same conversion path". They go through our code the same
way, but the dose and timing come from different scans.

To catch a scan-level error I looked at organs other than the liver. A calibration error scales
every organ in the scan by the same factor. If it were driving the low livers, log(organ) would
track log(liver) with a slope near 1 across patients. Organ biology alone would give a slope near
0. For the aorta (blood pool) the slope is 0.46 [0.33, 0.57], and still 0.41 [0.26, 0.53] after
adjusting for tumour volume. The kidneys give 0.35 [0.11, 0.56] and the spleen 0.76 [0.46, 1.03].
So part of the liver's patient-to-patient spread is shared with the rest of the body, but not all
of it. Put as group differences, the 39 low-liver patients have livers 39% lower [−47, −33] than
the 60 in range, but aortas only 12% lower [−25, −5], kidneys 7% lower (not significant) and
spleens 34% lower. My first pass made the argument from those group differences alone, and
review caught that it's biased. The groups are chosen on the liver, so even under a pure
calibration error the other organs would drop less than the liver, just from regression to the
mean. The slopes don't have that problem, and they're what I'd quote.

The part not shared with the body looks like biology. Liver SUV falls with total tumour volume
(Spearman ρ −0.44, p 5×10⁻⁶), which is the PSMA tumour-sink effect Gafita et al. described. So
does the aorta (−0.38) and, more weakly, the spleen (−0.22), while the kidneys don't (~0).
Partial volume isn't the explanation either. An 8 mm eroded, lesion-free liver core tracks the
whole-organ mean almost perfectly (ρ 0.995), sits about 9% higher, and still leaves 34 patients
below 4. The cohort's PSMA liver median is 4.47 (IQR 3.44–5.53). For comparison, Gafita et al.
report a median of 3.7 (IQR 3.0–4.7) for Ga-68 PSMA-11. Our 4–8 range was simply set too high
for a real metastatic population, and DEEP-PSMA mixes Ga-68 PSMA-617 and ¹⁸F-DCFPyL on top of
that.

What I can't say is where the shared part comes from. Calibration, the tracer mix, uptake time,
kidney function and body habitus could all produce it, and this data can't tell them apart. What
I can say is that calibration error can't explain the out-of-range livers, and that it wouldn't
matter much for us if it existed. Our metrics are relative (PSNR and SSIM against each patient's
own SUVmax, NRMSE and SUVmax bias as ratios), so a per-scan scale factor that scales
prediction and truth alike cancels out of them.

Decisions. `data/processed/v1` stays as it is. The liver flags never excluded anyone and nothing
downstream reads them, so there's nothing to reprocess. The guardrail in CLAUDE.md now treats the
PSMA liver as recorded, not gated. The conversion is checked through the paired FDG liver plus the
kidney/bladder pattern, and a PSMA source without paired FDG (Ankara's DICOMs, if they come)
needs its own conversion check before use. That's also why `suv_check` stays in the code rather
than being thrown away after one run. One knock-on effect for yesterday's entry: `qc.py` is a new
file under `src/`, so when `p4_unet_noct_k2_s2` resumes it won't be true that nothing under `src/`
changed since `5ac6e9d`. Nothing in training or evaluation imports it, though, so the code that
trains the model is unchanged.

---

## 2026-10-12 (night) — Phase 4 resumed: a resume bug, and three seeds that agree

With Colab Pro I picked up the three replicates the free quota had left open: the rest of
`p4_unet_noct_k2_s2` (stopped after epoch 13) and both third seeds, `p4_unet_k2_s3` and
`p4_unet_noct_k2_s3`, all on a T4 through `03_train.ipynb`.

The first attempt didn't get past loading the checkpoint. It re-cached the 79 training and
validation patients (a quarter of an hour), found `last.pt`, and died with `TypeError: RNG state
must be a torch.ByteTensor` before its first epoch. `train.py` loads the checkpoint with
`map_location=DEVICE`, which also moves the saved RNG states onto the GPU, and
`torch.Generator.set_state` and `torch.set_rng_state` only take CPU tensors. So every resume on a
GPU would have failed the same way. Nobody saw it before because no run had ever been resumed;
the smoke test resumes on the CPU, where nothing moves. The fix (`494ef93`) moves the two states
back to the CPU before restoring them, and on the T4 `tests/test_unet25d.py` fails without it and
passes with it. The failed attempt only added its start and cache lines to the log and touched no
checkpoint. The fix doesn't change any number: it restores exactly the state a working resume
would have restored. A resumed run is still not bit-identical to an uninterrupted one, but
nothing here is, since `cudnn.benchmark` is on. Two sentences in earlier entries are now wrong,
though: `train.py` did change between `5ac6e9d` and the end of `p4_unet_noct_k2_s2`, so it is not
true that nothing under `src/` changed, nor that the training code is unchanged. Its `log.txt`
has three start lines: `5ac6e9d` for epochs 0–13, `1dc7f67` for the failed attempt, and `494ef93`
for epochs 14–49 and the evaluation. `provenance.json` names `494ef93`. Both third seeds ran from
scratch at `494ef93`, and all three runs were on a clean tree. A CT epoch took about 89 s and a
no-CT epoch about 70 s, so the three runs took about three and a half hours of T4 time including
evaluation.

The result is the second outcome I named yesterday, not the one I expected. The no-CT arm wins
every seed. After consistency, PSNR is 57.43, 57.53 and 57.50 dB without CT and 56.96, 57.18 and
56.77 dB with it, so no-CT is ahead by 0.47, 0.35 and 0.72 dB seed by seed. NRMSE goes the same
way in all three (about 0.052 against 0.054–0.056). Lesion SUVmax bias is −2.0, −2.3 and −1.8%
without CT against −3.5, −3.4 and −3.4% with it. For lesions under 1 mL it is −5.5, −5.8 and
−5.2% against −7.1, −7.4 and −7.7%. The bladder-adjacent lesions are the exception: about −1% in
both arms with no consistent order, but there are only 47 of them. The seed spread is small next
to the gap: the PSNR standard deviation across seeds is 0.05 dB for no-CT and 0.20 dB for CT,
against a 0.51 dB difference in means. Averaging each patient's metrics over the three seeds and
rerunning the patient-paired Wilcoxon test (Bonferroni over PSNR, SSIM and NRMSE), no-CT is
better in 20 of 21 patients on each of the three, with a median PSNR gain of 0.43 dB and
corrected p = 0.0009 for PSNR, 1.4×10⁻⁵ for NRMSE and 6×10⁻⁶ for SSIM. Seed by seed, no-CT leads
in 15, 20 and 19 patients on PSNR. I haven't tested the lesion biases per patient: `metrics.json`
only keeps lesion aggregates, and the per-lesion numbers are in `runs/<exp_id>/per_patient/` on
Drive. Their direction is the same in all three seeds, though.

The training curves show it too, and they say something about why. The best validation loss is
0.00197–0.00199 without CT and 0.00204–0.00206 with it, with no overlap. The CT model's training
loss is also higher (about 0.00128 against 0.00124 over the last ten epochs, in every seed). So
the CT arm is not overfitting to anatomy. It fits the training data worse in the first place, as
if the extra channels get in the way of the optimisation rather than adding information. Of the
three explanations from the pilot, noise is now ruled out. That leaves "the network has no use
for CT at k = 2" and "early fusion is the wrong way to give it the CT". The second fits a model
that is held back by its CT channels better than one that merely ignores them.

So at k = 2, stacking the CT onto the input channels makes the 2.5D U-Net slightly but reliably
worse, on global and lesion metrics alike. For phase 4 that means the no-CT U-Net is the
stronger reference, and that's how the paper should report it. For phase 5 it raises the bar:
the cross-attention model has to beat the no-CT U-Net, not the early-fusion one, before we can
say the CT is doing anything. What stays open is k = 3 and 4, where thicker slices leave more to
infer and the anatomy might start to pay off.

---

## 2026-10-13 — The U-Net at k = 3 and k = 4: does the CT start to pay off once slices get thicker?

Every model has to report k = 2, 3 and 4, and so far the U-Net only exists at k = 2. Last night
also left one question open: at k = 2 the CT makes the U-Net slightly but reliably worse, and I
can't yet tell "the network has no use for CT" apart from "early fusion is the wrong way to give
it the CT". Thicker slices should help separate the two. At k = 4 each measured slice averages
four thin ones, so there is much more inside every block that the PET alone can't pin down. That
is exactly where knowing where an organ edge or a lesion sits along z should matter most. If the
CT arm catches up or overtakes at k = 4, the network can use anatomy once PET runs out of
information, and the k = 2 result was about there being little left to infer. If it still loses,
the problem is more likely how the CT gets in, and that is the argument for phase 5.

So the plan is to train both arms at both factors: `p4_unet_k3`, `p4_unet_noct_k3`, `p4_unet_k4` and
`p4_unet_noct_k4`. They are copies of the k = 2 pilot configs with only `k` changed. Same seed
(20261011), same 50 epochs × 500 steps, same loss, same model width. The only knock-on effect is
on the input: the CT arm now gets k + 2 thin CT slices (5 at k = 3, 6 at k = 4) and both arms get
k spline slices, and the output has k channels. Because an epoch is a fixed 8000 blocks and there
are fewer blocks per patient at higher k, an epoch now covers about 0.9 of the training set at
k = 3 and 1.2 at k = 4, against 0.6 at k = 2. I kept the step count fixed rather than the number
of passes, so every run gets the same optimisation budget. The test set and its degraded volumes
are the frozen `data/degraded/v1` files that the cubic baseline already used at k = 3 and 4.

The training code was written for any k, but only k = 2 had ever been run through it. Before
sending four GPU runs to Colab I extended the smoke test (`tests/test_unet25d.py`) to k = 2, 3
and 4. Each one caches a tiny synthetic cohort, checks that the untrained network reproduces the
cubic spline exactly, trains two epochs, resumes and evaluates. All three pass on the CPU, so a
broken k path would have shown up here rather than an hour into a T4 session.

What I expect, written down before seeing anything. First, the U-Net should beat cubic by more
at higher k than at k = 2, because cubic gets much worse: 55.5 → 49.3 → 45.8 dB PSNR after
consistency, and its lesion SUVmax bias goes from −13% to −24% to −35%. There is simply more
left for a learned model to recover. Second, on CT I genuinely don't know. My guess is that the
no-CT arm still leads at k = 3 and the gap closes or flips at k = 4. I'll look at the same things
as at k = 2: global PSNR, SSIM and NRMSE inside the body mask, lesion SUVmax bias with the small
and bladder-adjacent lesions separately, the validation curves, and the patient-paired Wilcoxon
test. These are pilots with one seed per arm. At k = 2 the seed spread was 0.05–0.20 dB, so a gap
much bigger than that is worth taking seriously, but no CT claim at k = 3 or 4 goes into the
paper before two more seeds per arm, the same as at k = 2.

The runs will go in pairs through `03_train.ipynb`, CT then no-CT at k = 3 and then at k = 4. If the
quota runs out halfway, what's finished is still a matched pair. From the k = 2 timings (about
89 s per CT epoch and 70 s per no-CT epoch on a T4), plus a fresh patient cache for each k, I
expect around five to six hours of GPU time for all four. The configs and the notebook are
committed today; we haven't picked a start date for the runs yet.

## 2026-10-14 — The U-Net at k = 4: two dB over cubic, and the CT gap is gone

The k = 4 pair has run. I split the four runs from yesterday across two Colab notebooks so the
k = 3 and k = 4 pairs could train side by side. Both first attempts died about ten minutes in,
while they were still caching patients, because I changed my Colab plan in the middle and the
runtimes were recycled. That cost nothing but time: neither run had written a checkpoint, and
the only trace is an extra start line followed by a half-finished cache in each `log.txt`. I
restarted k = 4 at 16:53 UTC on a T4, at `c57f1d8` on a clean tree. `p4_unet_k4` cached the 79
training and validation patients in 15 minutes, trained at about 96 s per epoch and finished
its evaluation at 18:38. `p4_unet_noct_k4` reused the cache in the same runtime, took about 74 s
per epoch and finished at 19:46. That's a bit under three hours of T4 time for the pair. The k = 3
pair is being restarted separately, so this entry is about k = 4 only.

The first prediction was that the U-Net would beat cubic by more at k = 4 than at k = 2. That
holds for the lesions but not for PSNR. After consistency, cubic gets 45.84 dB and the U-Net
47.87 dB with CT and 47.79 dB without, a gain of about 2 dB. That's the same gain as at k = 2
(55.46 against 57.43 dB without CT). It is completely consistent across patients, though: the
no-CT U-Net beats cubic in all 21 test patients on PSNR, SSIM and NRMSE, and the CT one in 20,
21 and 20 (corrected p ≤ 6×10⁻⁶ for every comparison). The lesions are where the extra room
shows. Cubic loses 34.5% of lesion SUVmax at k = 4 and 37.6% for lesions under 1 mL. The U-Net
brings that to −12.5% and −23.8% without CT, and −13.8% and −24.9% with it. That recovers about
22 points of SUVmax against 11 at k = 2. But it also leaves a much larger residual: at k = 2 the
U-Net was within 2% on SUVmax, and at k = 4 it still underestimates small lesions by almost a
quarter. Bladder-adjacent lesions go from −18.9% with cubic to −6.6% without CT and −8.0% with
it, but there are still only 47 of them. SSIM is 0.9949 in both arms against 0.9914 for cubic.

On CT, the k = 2 picture doesn't carry over. At k = 2 the no-CT arm was better in every seed
and in 20 of 21 patients. At k = 4 the two arms are level: 47.79 dB without CT and 47.87 dB with
it, NRMSE 0.153 against 0.149, and the no-CT arm ahead on PSNR in only 12 of 21 patients. The
patient-paired Wilcoxon test finds nothing (corrected p = 1 for all three metrics).
The mean and the median even point in different directions. The median patient does 0.18 dB
better without CT, but one patient (`train_0054`) does 1.8 dB better with it, which is enough
to tip the mean. The per-patient differences now run from −1.8 to +1.0 dB, so k = 4 is a much
noisier comparison than k = 2. On lesions the no-CT arm is slightly ahead (−12.5 against −13.8%,
small −23.8 against −24.9%), the same direction as at k = 2 but about the size of a seed
difference there. The training curves have changed as well. At k = 2 the CT arm fit the training
data worse. At k = 4 its training loss over the last ten epochs is slightly lower (0.00426
against 0.00433), but its best validation loss is still higher (0.00702 against 0.00690).
That's a small gap and the training loss is noisy, so I wouldn't read overfitting into it yet,
but it isn't the k = 2 pattern any more.

So with one seed per arm, the most I'd say is that the CT penalty from k = 2 isn't there at
k = 4. Whether the CT actually starts to help needs the seeds. A 0.08 dB lead with a ±1.8 dB
patient spread is nothing to go on. It fits the second half of my guess from yesterday (the gap
closes at k = 4) without telling the two explanations apart yet. If the k = 4 seeds come out
level too, "the network only reaches for anatomy when PET runs out" becomes the better story,
and early fusion looks less like the wrong mechanism than a weak one. For phase 5 it doesn't
change the bar. The cross-attention model still has to beat the no-CT U-Net at every k. The
next runs are the k = 3 pair, then two more seeds per arm at k = 3 and k = 4.

## 2026-10-14 (night) — The U-Net at k = 3: the CT penalty is back

The k = 3 pair took three starts. The first one (16:32 UTC) died at patient 38 of the cache when
the plan change recycled the runtime, like the first k = 4 attempt. The second (16:48) got to
patient 26 and stopped at 16:53, the minute the k = 4 notebook restarted. Colab wouldn't keep
two GPU runtimes going for us at once, so the parallel plan didn't really work, and I waited for
k = 4 to finish before starting k = 3 again. Neither attempt wrote a checkpoint, so all that's
left of them is two extra start lines and two half caches in `p4_unet_k3/log.txt`. The third
start, at 20:02 on a T4 at `c57f1d8` on a clean tree, ran through. `p4_unet_k3` cached the 79
patients in 15 minutes, took about 95 s per epoch and finished evaluating at 21:47.
`p4_unet_noct_k3` reused the cache, took about 71 s per epoch and finished at 22:53.

Against cubic, k = 3 sits between the other two factors, as expected. After consistency, cubic
gets 49.29 dB and the U-Net 51.71 dB without CT and 51.17 dB with it. The no-CT gain (2.4 dB) is
a bit larger than at k = 2 and k = 4 (both about 2 dB), so the gain over cubic doesn't simply grow
with k. Both arms beat cubic on SSIM in all 21 patients. On PSNR the no-CT U-Net wins in 20 and
the CT one in 19 (corrected p ≤ 1.3×10⁻⁴ for every comparison). The lesions follow the pattern
from k = 4. Cubic loses 24.3% of lesion SUVmax and 27.3% for lesions under 1 mL. The U-Net brings
that to −7.8% and −15.3% without CT, and −8.3% and −15.9% with it. That's about 16 points of
SUVmax recovered, between the 11 at k = 2 and the 22 at k = 4, and the residual on small lesions
almost triples from k = 2 to k = 3. Bladder-adjacent lesions go from −11.2% to −1.4% and −1.7%
(n = 47 again).

Something I hadn't looked at before shows up when I line up the three factors: for the U-Net,
the consistency step costs a little lesion SUVmax at every k, while it adds 0.2–0.4 dB of PSNR.
Without CT the bias goes from −6.1 to −7.8% at k = 3, from −0.8 to −2.0% at k = 2 and from −10.9
to −12.5% at k = 4. For cubic it's the other way round: consistency improves SUVmax by about five
points at every k. My guess is that the correction spreads each thick slice's residual evenly
over its k thin slices, so when the network has put a peak in the right slice, the correction
partly flattens it (`consistency` in `data/degrade.py` adds the residual back with
`np.repeat`). We report the consistent numbers as the main result anyway, but this needs a
sentence in the Discussion, and it could matter for the phase-5 loss.

On CT, k = 3 looks like k = 2, not k = 4. The no-CT arm is better by 0.54 dB on average (median
0.29 dB) and in 19 of 21 patients on PSNR, 20 on SSIM and 19 on NRMSE. The patient-paired
Wilcoxon test gives corrected p = 0.001 for PSNR, 0.0009 for NRMSE and 9×10⁻⁶ for SSIM, and the
raw outputs say the same. The lesion biases point the same way but by half a point, which is
seed size. The training curves repeat the k = 2 pattern as well. The CT model ends with a higher
training loss (0.00286 against 0.00282 over the last ten epochs) and a higher best validation
loss (0.00462 against 0.00449), so it fits worse instead of overfitting.

What bothers me is `train_0054`. At k = 4 it was the one patient that tipped the mean towards CT,
at +1.8 dB. At k = 3 it goes the other way: no-CT is 2.1 dB better there, the second-largest gap
after `train_0074` (+2.3 dB). A patient that swings by almost four dB between neighbouring
factors tells me that one seed per arm is not enough to call k = 4 level. So with one seed, the
picture is that early fusion hurts at k = 2 and k = 3, and k = 4 is undecided. That doesn't
fit "the network reaches for anatomy when PET runs out" well, because k = 3 has less PET than
k = 2 and the penalty hasn't shrunk (0.51 dB in the k = 2 seed mean against 0.54 here). The
seeds at k = 3 and k = 4 will decide it, and those are the next eight runs. For phase 5 the bar
stays the no-CT U-Net at every k.
