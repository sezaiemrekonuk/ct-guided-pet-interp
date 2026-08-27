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
