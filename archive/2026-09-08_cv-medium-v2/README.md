# Era: cv-medium-v2

**24 runs, 2026-09-08 → 2026-09-11.**
Four days of structural search on `medium` at 5 rotating folds, opened by the
2026-09-08 annotation revision. It moved the headline from **0.218 to 0.321**,
and almost all of that came from three levers: **wider input representations**
(`yamnet_context`, then `yamnet_aves`, then both concatenated), **a wide shared
hidden layer** over them, and — the methodological result, worth as much as
either — **removing `val_loss` early stopping**.

Closed on 2026-09-11 by three simultaneous changes: a second **annotation
revision** (Luke pulling new annotations by hand), a **scoring change** (buzzes
tagged `_quiet` leave the sensitivity equation, reported both ways), and the
adoption of **`--fixed-epochs` as the stopping rule** in `main`. Any one of the
first two moves every float in this log.

## What made these comparable

| | |
|---|---|
| Set | `medium` — `overlap_event_prop: 0.2`, `framehop_prop: 1` |
| Folds | 5 rotating (`role: rotate`), 78 train-only |
| Metric | `total` row `sensitivity` at `fpr` 0.005 in `folds_sx.csv` — plain mean over folds, threshold set per fold on its own held-out audio |
| Translation | `general` (15 classes). The era opened by flipping the default off `general_v1`, which protected the previous era's log |
| Embedder | `yamnet` (1024-d) unless the experiment *was* the embedder. Six others were tried: `yamnet_context`, `aves`, `yamnet_aves`, `yamnet_context_aves`, `perch`, `yamnet_bandpass` |
| Head | `Dropout(0.2) → Dense(15)`, label smoothing 0.2, Adam 0.002, no weight decay — hardcoded, and YAMNet-tuned. `--hidden` arrived late and only on a branch |

**The epoch rule is not constant across this log, and it is worth more than
most of the levers tested.** Three rules appear: `val_loss` early stopping (18
entries), `--monitor val_sens` (3, all `artifact`), and `--fixed-epochs` (the
last 3, plus `--epoch-rule xfold` once). Removing early stopping is worth
**+0.031 to +0.040** on its own. Read the entries from 2026-09-10 on only
against their matched controls — they say which — and never against
`cv-baseline`.

## Baseline and best

`cv-baseline` (`models/cv_baseline/`) = **0.218**, a linear probe on frozen
YAMNet under `val_loss` early stopping. Every entry before 2026-09-10 compares
to it. It is also the one number this era never re-anchored: **no fixed-budget
`cv_baseline` was ever run**, so the ~0.10 span from baseline to best mixes a
real representation gain with a stopping repair of unknown size (expect +0.02
to +0.04 of it).

| Result | | Trust |
|---|---|---|
| `yamnet-aves-context` | 0.321 | clean |
| `yamnet-aves-head-fixed` | 0.313 | clean |
| `context-monitor-r2` | 0.312 | artifact |
| `context-monitor` | 0.307 | artifact |
| `xfold-epoch` | 0.288 | clean |

Read the floats with their `trust`:

- **`context-monitor` / `context-monitor-r2` (0.307 / 0.312) are `artifact`.**
  They early-stopped on `--monitor val_sens`, i.e. on the held-out fold's own
  score, so they report the max over epochs of the statistic they are graded
  on. The inflation is **budget-dependent** — +0.021 over a 400-epoch argmax,
  ~+0.006 over 150 — so they are not as far off as a flat "+0.021" reading
  suggests, but they are not comparable at face value. `monitor-leakage`
  (0.277) is the honest rerun.
- **`probe-grid` (0.257) is `artifact`** for the same reason in its L1 arms.
- **`aves-probe` (0.074) is `caveated` and should not be read as an embedder
  verdict.** It scored AVES through the YAMNet-tuned head; an offline L2 sweep
  on the same cached embeddings recovered most of the loss (`aves-readout`,
  0.194). The general lesson is in LOOP.md: a negative embedder result is
  `embedder + this head` until you sweep the readout offline.
- **`yamnet-aves-context` (0.321) and `yamnet-aves-head-fixed` (0.313) are the
  two numbers worth carrying forward as leads**, and both are budget-limited:
  every 150-epoch run on disk is **still rising at its cap**.

## Gotchas

Things a reader a year out will get wrong:

1. **Do not compare across epoch rules.** This is the big one. Half the log is
   undertrained, concentrated on `1_150`, which early-stops at epoch 2–32 under
   every embedder tried while its buzz curve climbs to e120–185. A treatment
   that happens to move `best_epoch` shows a gain that is entirely a stopping
   change — `recorder-center`'s textbook +0.082 at `1_150` is **-0.007** once
   both arms are scored at one epoch. Read `best_epoch` beside every delta.
2. **The undertraining is *not* selection leakage, and the two were conflated
   for two days.** `val_loss` selection optimism measured **-0.002** over 17
   runs (8 up, 9 down). The fix is a **budget**, not a non-leaking selector:
   `buzz-selector-curves` showed all four candidate selectors return identical
   per-fold sensitivities, because the pooled curve is flat across the
   plateau. The full correction is
   `exp/pairwise-rank:notes/new-era-audit.md`, which supersedes
   `exp/xfold-epoch:notes/new-era.md` and `notes/epoch-selection.md`; read the
   audit first or you will inherit the wrong diagnosis.
3. **`context-stack`'s +0.050 was an eval bug, not a result.** Frames inside
   one snip almost always share a label, so any lever that lets a frame read
   its cached *neighbours' rows* inflates. Rebuilt from real neighbouring
   audio it is +0.022 (`context-embedder`, `clean`). Any future score-smoothing
   or embedding-stacking idea inherits this.
4. **Annotation moved under the results within the era.** `medium` was being
   annotated toward 24 snips per fold throughout, so a run's folds are not
   byte-identical to a run four days earlier. Small deltas across dates are
   partly this.
5. **Fold spread is not a confidence interval.** Training pools overlap ~90%
   across rotations. And per-fold bootstrap SD (0.010–0.037, `1_150` worst)
   measures *eval sampling only* — `1_150` moved **0.055** between two
   identical runs, so its run-to-run noise is dominated by training
   stochasticity. There is no seed control anywhere in this era.
6. **A repeat spread bounds noise *within* a treatment, not a difference
   *between* two.** The 0.007→0.062 `1_150` figure is quoted all over the era's
   docs and was used, wrongly, to discount a real between-treatment gain.
7. **The hard folds are the point.** `1_150` (apple), `1_95` and `willard` sit
   near chance; a lever that moves only the rich folds is worth little. Do not
   write a concentrated hard-fold gain up as "rests on one thin fold".
8. **`1_95`'s failure mode is a single jet flyover** — ~90 s of `mech_auto`
   sets the whole fold's threshold, and only the buzz neuron fires on it. Four
   interventions failed on it (`harmonic-comb`, `mech-margin`,
   `shared-trunk-head`, `context-embedder`). `yamnet_aves` is the only thing
   that has ever moved it (+0.014 / +0.024), and it is still near chance.
   Diagnostic: `diagnostics/2026-09-09_jet_1_95/`.

## Fold roster at cutover

Counts from `set/`, `ins_buzz` under the translation named above.

| rotating fold (deployment) | annots | buzz | buzz s | annot s | span h |
|---|---|---|---|---|---|
| JamesU - MustardBumbler/1_29 | 1458 | 441 | 2391 | 12191 | 33.8 |
| Lily - Fit+Fast/2023_R3_Marysville/53 | 430 | 157 | 1148 | 8533 | 35.0 |
| Lily Adam - One Hive/recorders/willard/2024-08-07/1_11 | 743 | 191 | 184 | 8479 | 36.8 |
| Luke - Diel Drivers/2026-04-08/1_150 | 1569 | 77 | 88 | 4974 | 21.4 |
| Luke - Diel Drivers/2026-05-06/1_95 | 1258 | 157 | 328 | 7820 | 27.7 |
| **total** | **5458** | **1023** | **4139** | **41997** | **154.6** |

## Notes and code

`notes/` holds 69.

**32 recovered from dangling commits** (their branches were already deleted): `backbone-slight-ft`, `batchnorm`, `buzz-in-train`, `buzz-upweight`, `class-weight-fix`, `context-pooling`, `context-stack`, `context-width`, `deployment-forensics`, `dropout-repro`, `framehop-overlap`, `hyperparam-sweep`, `input-standardization`, `l2-only`, `l2-regularize`, `low-delta`, `ls02-repro`, `mlp-head-repro`, `no-reg-baseline`, `std-convergence`, `supp-freq`, `supp-freq-v2`, `tail-loss`, `tail-loss-retest`, `temporal-context`, `white-noise`, `with-dropout`, `yamnet-bandpass`, `yamnet-combined`, `yamnet-ft`, `yamnet-mask`, `yamnet-native-buzz`. Code at `refs/archive/<name>`.

Branches: `exp/*` on origin where they survive.

## Reading these forward

**Positive results rarely survive a data change; dead ends usually do.** Both
halves of that have already been demonstrated here — `temporal-context` was a
clear negative in era 1 and, rerun as `context-stack`, the largest gain in era
2.

**Worth rerunning first, in this order:**

1. **A fixed-budget `cv_baseline`.** The era never anchored one, so nothing
   here has a clean comparator under the new default rule. Expect +0.02 to
   +0.04 over 0.218, chiefly at `1_150` — and note that era-2's own
   pre-registered falsifier would have read that gain as evidence something
   was wrong. It is not.
2. **The budget ladder.** Every fixed-budget run on disk is still rising at its
   cap (150 for all but one; the one 400-epoch run peaked at e234). Whether the
   wide-head configs keep gaining past 250 is **untested**, and it sets the
   budget for everything after it. This is the highest-value cheap experiment
   the era points at and the only one it left undone.
3. **`yamnet_aves` + `yamnet_context` concatenated, with a 1024-wide hidden
   head** — the era's best config (0.321). It is a structural, one-directional
   result, so it is the likeliest of the positives to survive.
4. **The `general` / `binary` translation control.** Never rerun cleanly. Note
   two confounds that have nothing to do with the taxonomy: `ins_buzz`'s
   positive weight moves ~5.4x because `build_weights` puts the class *count*
   in every denominator, and `val_loss`'s composition goes from ~1/15 buzz to
   ~1/2. The second disappears under a fixed budget; pin the first.

**Dead ends — measured on YAMNet embeddings, and that qualifier matters:**
bandpass, mel masking, MLP heads, L2, handcrafted frequency features, harmonic
combs, margin losses on `mech_auto`, recorder centring, gain normalisation.
Head and regularisation verdicts are statements about a *representation*, not
about the pipeline; if the embedder changes they do not carry.

**Closed for the right reason, do not revive:**

- **Per-site score calibration / label-free threshold selection** — inert under
  this metric by construction (every fold is already tuned on its own audio).
- **`1_150` as an annotation-quality problem.** Luke listened on 2026-09-09:
  "Most of them are very quiet, but still legitimate targets." Its *stopping*
  was broken, not its labels, and a fixed budget recovered +0.160 to +0.172
  there. **This is also the observation the next era's `_quiet` tag comes from**
  — the distinction being drawn is between quiet-but-fair and
  quiet-beyond-reasonable-expectation, not between real and spurious.
- **Leave-one-concept-out as the cause of the hard folds** — ruled out twice.
  r = 0.013 between sublabel coverage and per-fold sensitivity. A rerun that
  appeared to overturn it had filtered to `role == 'rotate'`; check roles
  before reviving it.
- **Which epoch *selector* to use** — the selector does not move the score at
  all on this data. The budget does.

