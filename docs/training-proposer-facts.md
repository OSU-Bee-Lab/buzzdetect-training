# Training proposer facts

Moved out of `IDEAS.md` (which holds candidate experiments only). Read before
proposing a `03_train` experiment. Era tags E1-E4 below are as of 2026-09-14;
the current log is a later era, so treat E4 as "the era then current".
 
**Read the epoch rule before you read any number here.** The fixed `--epochs`
budget is the default and selects no epoch. **A different budget N is a
different rule**, so still run a matched control.

## Which era a number came from

An untagged claim is a proposal, not a measurement.

| tag | era | how to treat it |
|---|---|---|
| **E1** | `archive/2026-06_fixed-test` — 29 runs | Different metric, different data, two revisions ago. **Not a verdict.** Re-establish or don't cite. |
| **E2** | `archive/2026-08_cv-medium-v1` — 30 runs | Right metric, wrong data and roster. Directions survive; numbers don't. |
| **E3** | `archive/2026-09-08_cv-medium-v2` — 24 runs | Right metric family, but pre-revision data, `Dropout(0.2)`, and a mix of three epoch rules. Directions survive; numbers don't. |
| **E4** | current `03_train/log.jsonl` — 2026-09-11 on | Directly comparable. |

`temporal-context` was a clear E1 negative and, rerun as `context-stack` in E2,
became the largest gain in the log. **E1/E2/E3 negatives are leads, not
answers.**

## Standing facts a proposer needs

- **MDE is ~0.027 on the headline** at n=1. Anything smaller is unreadable from
  one run. Prefer dose ladders to single comparisons.
- **A single fold resolves ~0.09, not ~0.027.** `buzz_frames` is not the sample
  size; folds hold 14-42 buzz *events*. `tools/eval_sampling_sd.py <control>
  --other <exp>` gives each fold's delta its own SD; training stochasticity adds
  a smaller ~0.016-0.026 per fold delta on top (`docs/judging-results.md`).
  Propose against the headline; use folds to ask *where* an effect lives.
- **Context/averaging levers have a standing prior against them: four for
  four, they lift rich folds and leave hard folds flat-to-down.**
  `context-verify`/`context-stack` (+0.085, "three times landed on rich folds
  only" per the era-open summary above), `asym-context-yamnet` (+0.046, 6/8
  folds but both `1_95`/`1_114` flat-to-down even with an explicit-contrast
  design), and `trunk-ft-v3`'s `trunkctx-ft-1e5` (flat, +0.002 over plain
  fine-tuning) all show it. Mechanism: this era's hard folds (`1_95`, `1_114`,
  `1_150`) fail on isolated ~1 s events; averaging or concatenating a wider
  window washes out exactly that signal while flattering `1_29`/`53`'s
  254 s-median sustained drones. This is backwards from the era's goal (hard
  folds are the target, not rich-fold noise). A new context/broadcast/mean-pool
  proposal (Perch included — a naive mean-pooled window has the same
  structure) must argue past this prior: prefer a design that preserves the
  transient (max-pool, attention, or an explicit local-vs-baseline contrast)
  over another plain average, and treat `1_95`/`1_114`/`1_150` as the falsifier,
  not the headline.
- **Never select an epoch on the held-out fold.** Any rule that reads the
  scored fold's own curve reports max-over-epochs of the graded statistic. A
  cross-fold rule stays available offline via `tools/honest_epoch.py`.
- **`_quiet` buzz leaves the score, not the training pool.** Headline is
  `sensitivity_exclquiet`; every run carries per-tier columns. **A lever that
  moves neither `untagged` nor `loud` is not a detection gain.** `1_29` and
  `53` carry 254 s-median `_background` drones no other fold has — read their
  `untagged` column, not their headline.
- **Per-site *score* transforms are inert under this metric.** Each fold's
  threshold is set on its own negatives, so any strictly increasing per-fold
  transform leaves `sens@fpr0.005` bit-identical. Do not queue calibration,
  temperature scaling, per-site z-scoring of scores, or label-free thresholds.
- **Any input normalisation moves the optimum** (four for four:
  `input-standardization`, `standardize-blocks`, `recorder-center`,
  `yamnet-aves-standardize` — the last negative at a fixed budget). A linear
  readout already rescales blocks itself.
- **Reading neighbouring cache rows leaks; reading neighbouring audio does
  not.** Frames inside one snip almost always share a label. Anything that
  lets a frame see adjacent cached *rows* (score smoothing, embedding stacking)
  inherits `context-stack`'s inflation. Embedders that need neighbouring audio
  set `context_frames` (on `main` since `context-frames-fix`) and read it from
  the contiguous buffer.
- **Controls go stale with the annotations.** Loudness tagging is ongoing, so a
  control trained days earlier scores a different buzz-event pool
  (`aves-p3` was caveated for exactly this). If the set's annotations moved
  since your control ran, rerun the control under the same `--name` suffix
  rather than joining across the change.
- **A clean extraction currently fails** `read_fold_roles` on
  roster drift (`…/2026-07-27/1_99` is in `folds.csv`
  but not `annotations.csv`). The known workaround is an empty directory of
  that name under the new embedder's cache (`pitchshift-decimate-up`
  notes.md). Do not edit `01_annotate/`.

### The hard folds, re-measured on the lead (2026-09-14)

**Low thresholds are good**: a fold's threshold is where its own negatives sit
at 0.5% FPR, so a *high* threshold means negatives are scoring like buzz. On
`pitchshift-aves-mid`, four folds lag, and they fail in **three different
ways** — never treat them as one item.

| fold (crop) | lead sens | threshold | what sets it (negatives above threshold) | kind |
|---|---|---|---|---|
| `1_95` (blueberry) | 0.170 | **-0.97** | 33 frames, plane-heavy mix (`mech_plane` 9 alone, more in mixtures) | **FP: aircraft** |
| `1_114` (senna) | 0.210 | **-0.72** | **21 of 21 are `ins_trill`** | **FP: orthopteran trill** — new |
| `1_37` (chicory) | 0.410 | **-1.00** | **20 of 22 are `ambient_background`** | **FP: "background"** — new |
| `1_150` (apple) | 0.468 | -2.07 (good) | diffuse, 23 frames | **positives**: genuine low-SNR buzz |

Every other fold sits at -1.57 to -2.07. `1_114`'s trill confusion **explains
this era's fold pattern at `1_114`**: pitch-shift was +0.181 there, the tile
seam's removal +0.034/+0.043, and adding AVES layers to pitch-shift took it
back down -0.204. Those counts are one draw on 21-34 frames each — a
mechanism lead, not a measurement to optimise.

**How to reproduce the census** (it bit the agent who first ran it): the
thresholds in `folds_sx.csv` are on the **raw activation** scale, as is
`predictions.csv`, but `predictions.csv` has no label column. `surprisal/<fold>/
*_surprisal.csv` has `label`, but its `activation_ins_buzz` is **sigmoid(raw)**
and its rows are in a **different order**. The sorted values match exactly, so
take `logit` of the surprisal activation and compare that to the threshold,
using only rows whose `label` does not contain `ins_buzz`.
