# Audit: `notes/new-era.md` and `notes/epoch-selection.md` (exp/xfold-epoch)

Written 2026-09-11 on `exp/pairwise-rank`, against the documents as committed
2026-09-09 (`c94f774`). Every number below was measured offline from the
per-fold `summary.json` curves and `folds_sx.csv` files on disk
(`tools/honest_epoch.py`, `tools/compare_folds.py`, `tools/eval_sampling_sd.py`,
plus two ad-hoc reductions of the persisted `val_sens_fpr0.005_curve`s). No
training was run; the `pwr_lam05` CV was left alone throughout.

---

## Verdict

**The diagnosis is sound. The prescription is wrong, and the framing is wrong
in a way that has already changed how the loop reads results.**

Three separate things:

1. **The mechanism the document found is real and important.** Early stopping
   on `val_loss` fires absurdly early on some folds — epoch 6 of 400 on
   `1_150`/apple for `yamnet_context`, epoch 5 for `yav_h0`, epoch 14-32 on
   plain 1024-d YAMNet — and scores those folds on a barely-trained probe.
   Removing it is worth **+0.031 to +0.040** headline on two independent
   configs. That finding stands unqualified and is the era's most valuable
   methodological result.

2. **The document attributes that gain to the wrong cause.** It frames the
   problem as *selection leakage* (bias from choosing an epoch on the scored
   fold) and the fix as *cross-fold epoch selection*. Measured across all 17
   `val_loss`-stopped runs on disk, the optimism from `val_loss` selection is
   **-0.002** (mean of shipped minus a non-leaking cross-fold epoch; 8 up,
   9 down, range -0.038 to +0.075). There is no measurable leak in the
   `val_loss` arm. The problem is **undertraining**, not bias, and the two call
   for different fixes.

3. **Consequently, step 2 of the plan ("flip the default to
   `--epoch-rule xfold`") is superseded by `--fixed-epochs`.** Scored offline
   on every fixed-budget run on disk, taking the *final* epoch of a fixed
   budget is within 0.006 of the cross-fold rule everywhere, and beats it on
   the very config the document was written from:

   | config | budget | fixed-N (final epoch) | `xfold` pooled | oracle fixed-N |
   |---|---|---|---|---|
   | `xfold_context` (yamnet_context, linear) | 400 | **0.294** at e250 | 0.288 | 0.295 at e234 |
   | `yavf_h0` (yamnet_aves, linear) | 150 | **0.276** | 0.275 | 0.276 at e150 |
   | `yavf_h256` | 150 | **0.284** | 0.281 | 0.285 at e147 |
   | `yavf_h1024` | 150 | **0.313** | 0.312 | 0.314 at e149 |
   | `yavf_h1024_r2` | 150 | 0.300 | **0.303** | 0.303 at e149 |
   | `yavx_h1024` (yamnet_context_aves) | 150 | **0.321** | 0.318 | 0.321 at e150 |

   `--fixed-epochs` performs no epoch selection of any kind, so it has no
   nesting residual to apologise for, and it is already implemented, already
   used for the log's three most recent entries, and measurably not worse.
   **Recommendation: adopt `--fixed-epochs`; keep `--epoch-rule xfold` as a
   diagnostic and as insurance against a config that genuinely overfits within
   the budget (none on disk does).**

**On ending the era:** the strongest argument in the document —
"a mixed log would be uninterpretable" — has already been falsified by
practice. Entries from 2026-09-10 and -11 (`yamnet-aves-head-fixed`,
`yamnet-aves-context`) ran under `--fixed-epochs` inside this log and were
read correctly, because each was compared against a **matched control run
under the same rule** (`yavf_h0`, `yavf_h1024`), not against `cv_baseline`.
That discipline, not a log boundary, is what makes a stopping-rule change
survivable. Whether to archive anyway is Luke's call; the evidence does not
compel it, and archiving costs cross-era comparability on unchanged data.

---

## Claim by claim

### Claim 1 — "Every number in `log.jsonl` was produced by scoring each fold at an epoch chosen from that same fold"

**Partially holds as of 2026-09-09; false as of 2026-09-11; and misleading
either way.**

Literally true on the day it was written: every run then on disk used early
stopping monitored on the held-out fold. But the sentence collapses two
mechanisms that measure completely differently, and the document's next
sentence ("that moves every entry") inherits the conflation.

Breakdown of the 23 era entries by how the epoch was chosen:

| selection | entries | measured optimism |
|---|---|---|
| `val_loss` early stopping (different statistic from the score) | 18 | **-0.002** mean over 17 runs on disk (8 up / 9 down) |
| `--monitor val_sens` (the scored statistic itself) | 3 (`probe-grid` L1 arms, `context-monitor`, `-r2`) | +0.006 to +0.021 (see claim 5); all three already marked `artifact` |
| `--fixed-epochs` — **no selection at all** | 2 (`yamnet-aves-head-fixed`, `yamnet-aves-context`), both postdate the document | n/a |
| cross-fold | 1 (`xfold-epoch`) | n/a |
| offline, not a pipeline number | 2 (`aves-readout`, `aves-mlp-head` sweep) | n/a |

The -0.002 figure is the load-bearing one and it is new here. Per-run
`shipped − xfold-pooled` for every `val_loss` run on disk:

```
L2_mindelta0 +0.004  L3_batch4096 -0.038  L5_drop0 +0.008  L5_drop01 -0.002
L6_ls0 -0.006  L6_ls005 +0.001  L6_ls005_r2 -0.027  L7_wd1e4 +0.003
base_r2 -0.001  base_r3 +0.006  context_embedder -0.019  cv_baseline -0.001
harmonic_comb_r2 -0.002  perch_probe +0.075  yav_h0 +0.007
yav_h1024 -0.024  yav_h256 -0.015          mean -0.002
```

This independently reproduces `monitor-leakage`'s own val_loss arm
(0.218 → 0.216, n=3) on 17 runs instead of 3, and it **vindicates README's
"bounded, and small" claim for `val_loss`**, which `epoch-selection.md` §2
dismisses as "Wrong answer A". The a-priori argument in §2 is correct that
`val_loss` selection is *not literally unbiased*; the empirical claim that this
matters at the scale of the log's effects is not supported by the data in the
repo.

Note the sign: the scatter is symmetric and the largest single entry (+0.075 on
`perch_probe`) is a truncation artifact of the estimator, not a leak. If
anything the `val_loss` numbers are **deflated**, because stopping early leaves
sensitivity unclaimed — which is claim 2's real story.

### Claim 2 — "That moves every entry — measured at +0.030 on `yamnet_context`"

**The number is right. "Moves every entry" is not what was measured, and the
framing inverts the finding.**

Paired, from the two model dirs that share one set of trajectories
(`tools/compare_folds.py models/xfold_context_earlystop models/xfold_context`):

| fold | early-stop epoch | earlystop | xfold | delta |
|---|---|---|---|---|
| `1_150` (apple, 146 buzz frames) | **6** | 0.027 | 0.199 | **+0.172** |
| `53` | 91 | 0.489 | 0.516 | +0.027 |
| `1_95` | 118 | 0.016 | 0.009 | -0.007 |
| `1_11` (willard) | 61 | 0.216 | 0.200 | -0.016 |
| `1_29` | 155 | 0.537 | 0.518 | -0.019 |
| **total** | | **0.257** | **0.288** | **+0.031** |

The four folds whose stopping was sane contribute **-0.003** to the headline
mean; `1_150` alone contributes **+0.034**. So **110% of the headline comes
from one fold**, and the document's own "insurance, not improvement" section
says exactly this — but the era-ending argument is built on the headline
sentence, not on the section three paragraphs later.

(The document and the log entry quote -0.008 / +0.185 for these two quantities.
Recomputed from both `folds_sx.csv` and the persisted curves I get **-0.004 /
+0.172**. Not material to any conclusion, but the smaller pair is what the
files support.)

Correct phrasing: **the rule repairs a specific pathology on the folds where
early stopping fired absurdly early, and costs a few thousandths everywhere
else.** Per-fold bootstrap SD on this run is 0.025-0.035
(`tools/eval_sampling_sd.py`), so -0.004 is ~0.15 of a per-fold SD and +0.172
is ~5x it.

The corollary the document does not draw: if the gain is a stopping *repair*
rather than a *scoring* correction, then the entries needing re-reading are the
ones whose stopping broke — which is diagnosable per-entry from `best_epoch`,
already a standing LOOP.md/IDEAS.md check — not all of them.

### Claim 3 — "Budget 250 — cost against the pooled peak is -0.001 at e250, -0.004 at e150, -0.016 at e100"

**Verified on `yamnet_context`. Does not transfer, and 150 is demonstrably too
short for the current best configs.**

Reproduced on `xfold_context` (400-epoch budget, the only untruncated long run
on disk): pooled argmax e234, and deficit against the pooled peak
-0.022 (e50) / -0.013 (e100) / -0.004 (e150) / -0.002 (e250) / -0.004 (e400).
The document's figures are confirmed.

They do not generalise. Every other fixed-budget run on disk is at 150 epochs,
and **all of them are still rising at the cap**. Pooled mean sensitivity,
mean(last 21 epochs) − mean(e100-120):

| config | pooled argmax | still-rising | e100 deficit |
|---|---|---|---|
| `xfold_context` (3072-d, linear, 400) | 234 | +0.006 over e100→e400 | -0.013 |
| `yavf_h0` (1792-d, linear, 150) | **150 (capped)** | **+0.009** | -0.016 |
| `yavf_h256` (150) | 147 | +0.005 | -0.011 |
| `yavf_h1024` (150) | 149 | **+0.014** | -0.026 |
| `yavf_h1024_r2` (150) | 149 | +0.011 | -0.017 |
| `yavx_h1024` (3840-d, 150) | **150 (capped)** | +0.008 | -0.012 |

Per-fold peak epochs for the h1024 configs cluster at 140-150 on the hard folds
(`yavf_h1024`: 1_150 at 145, 1_95 at 149, 53 at 147; `yavx_h1024`: 1_150 at 150,
53 at 147). **The e100 cost is -0.026 on `yavf_h1024`, 1.6x the -0.016 the
document quotes**, and whether e250 suffices for these configs is **untested** —
no run has gone past 150 on them. Directionally, wider input and a hidden layer
both push the optimum later, so 250 is a floor, not a settled value.

Not verifiable offline: whether the h1024 configs would keep gaining past 250.
That needs one long run and is the single highest-value cheap experiment left
from this document.

Second-order but worth recording: `yamnet-aves-head-fixed`'s log entry reads
`yavf_h0`'s "own-peak 0.277 == shipped 0.276" as evidence that the config *"had
plateaued by e150 rather than overfitting."* Under a fixed budget with no
restore-best, own-peak == shipped means the per-fold peak is **at or adjacent to
the final epoch** — which is evidence the curve is **still rising**, i.e. the
budget is too short. The conclusion drawn (h1024 is worth +0.030) is unaffected,
since both arms ran the same budget; the plateau reading is not.

### Claim 4 — "The stopping failure is specific to wide inputs; 1024-d YAMNet did not have one (its early-stop epochs were 70-174). If plain YAMNet jumps +0.03 too, something else is going on."

**Fails.** The 70-174 range is computed over the folds that behaved and silently
excludes the fold the whole argument is about.

`best_epoch` on `1_150` across every plain-1024-d-YAMNet `val_loss` run on disk:

```
cv_baseline 26/70   base_r2 23/69   base_r3 20/67   harmonic_comb_r2 14/64
L2_mindelta0 18/68  L3_batch4096 2/52  L5_drop0 18/67  L5_drop01 20/68
L6_ls0 32/78  L6_ls005 22/68  L6_ls005_r2 25/71  L7_wd1e4 20/68
yamnet_medium_general_v2 18/68
```

Every one of them stops on `1_150` at epoch 14-32 while stopping the other four
folds at 65-180. That is the same pathology, at the same fold, differing from
`yamnet_context`'s epoch 5 only in degree. `L3_batch4096` stops `1_150` at
**epoch 2**. The pathology is **fold-specific first and width-specific second**:
`1_150`'s multi-class `val_loss` bottoms out almost immediately on every
representation tried, while its buzz curve keeps climbing to e120-185.

Direct corroborating measurement at an intermediate width: `yav_h0`
(yamnet_aves, 1792-d, linear) early-stopped vs `yavf_h0` (identical config,
fixed-150) is **+0.040**, with `1_150` going 0.000 → 0.160 — and unlike the
`xfold` comparison, the four healthy folds also gain (+0.010 mean:
willard +0.028, 1_95 +0.013, 53 +0.003, 1_29 -0.003).

`aves-mlp-head` already recorded the plain-YAMNet version of this incidentally:
"converged YAMNet linear offline 0.248 vs shipped 0.218 — the stopping rule
leaves ~0.03 on the table for YAMNet too."

**So the document's pre-registered falsifier would fire, and firing would mean
nothing is wrong.** A plain-YAMNet fixed-budget run should be *expected* to gain
~+0.02 to +0.04, chiefly at `1_150`. The falsifier must be rewritten or it will
mislead the next agent into disbelieving a correct result.

### Claim 5 — "`--monitor val_sens` stays dead. Measured inflation +0.021."

**The verdict holds. The number is budget-dependent and should not be quoted as
a constant.**

The mechanical finding is exact and I reproduce it: on every `val_sens` run,
`own-peak` and `shipped` are identical to three decimals (`L1_sens` 0.241/0.241,
`L1_sens_r2` 0.257/0.257, `L1L6_combo` 0.262/0.262, `context_monitor`
0.307/0.307), and on no `val_loss` run are they. A `val_sens` run reports the
max over epochs of the statistic it is scored on. That is not arguable.

The **size** is a function of how many epochs the argmax ranges over.
`own-peak − xfold-pooled` on untruncated fixed-budget runs:

| config | budget | own-peak | xfold-pooled | leak |
|---|---|---|---|---|
| `xfold_context` | 400 | 0.309 | 0.288 | **+0.021** |
| `yavx_h1024` | 150 | 0.330 | 0.318 | +0.012 |
| `yavf_h1024` | 150 | 0.319 | 0.312 | +0.007 |
| `yavf_h256` | 150 | 0.288 | 0.281 | +0.007 |
| `yavf_h0` | 150 | 0.277 | 0.275 | +0.002 |
| `yavf_h1024_r2` | 150 | 0.305 | 0.303 | +0.002 |

Mean over the five 150-epoch runs: **+0.006**. The +0.021 is the 400-epoch
figure. Both are correct measurements of different things; the document
presents +0.021 as *the* leak, and IDEAS.md and three `log.jsonl` amendments
now quote it that way.

The leak also spreads across folds rather than concentrating
(`xfold_context` per fold: 1_11 +0.023, 1_150 +0.020, 1_29 +0.023, 1_95 +0.030,
53 +0.006), which is what you expect from an argmax over a noisy curve and is
good evidence it *is* selection rather than a fold repair.

Practical consequence: `--monitor val_sens` should stay dead — but under a fixed
budget, own-peak is a *bounded* overstatement (~+0.006 at e150), so the three
`artifact` entries are not as far off as the current amendments imply. Their
`trust` should stay `artifact` (the number still is not comparable at face
value) but the magnitude language needs the budget qualifier.

### Claim 6 — the "-0.008 cost on healthy folds" and the buzz-only-selector lead

**Both stale.** The cost is **-0.004**, not -0.008 (claim 2). And the open lead
was closed *negative* by `buzz-selector-curves`, committed `549c1e5` on
2026-09-09 — **71 seconds after** the document's own commit `c94f774`, which is
why the document does not know about it.

`buzz-selector-curves`' finding is structural, not a null: under a non-leaking
cross-fold rule, all four candidate selectors (`val_loss`, `sens@fpr0.005`,
`auc_buzz`, `ce_buzz`) return **identical per-fold sensitivities**
`[0.20, 0.41, 0.38, 0.04, 0.09]`, because the pooled sens curve is flat over
e129-164 and any in-plateau selector lands on the same score. That closes the
lead for the right reason and also, incidentally, undercuts the whole premise
that *which* selector you use is where the sensitivity is. It is not; the
**budget** is.

The document's "Open lead to fold in at the cutover" section should be deleted,
not softened.

### `epoch-selection.md`

- **§2's retraction of the epoch-smoothing estimate is correct and the reasoning
  is good.** "To estimate how much an argmax can inflate a statistic you need
  the sampling variability of that statistic under a fresh draw of the
  evaluation data, never the smoothness of the curve" is right, and the
  0.010-0.037 bootstrap SD is the correct scale. Keep it.
- **§3's Cawley & Talbot variance-not-bias argument is correctly applied.** The
  claim that a biased low-variance selector can beat an unbiased high-variance
  one for *model selection* is exactly what that paper argues, and
  `sens@fpr0.005` resting on 24-35 negative frames per fold is a textbook
  high-variance criterion. I could not fetch the paper to verify the quoted
  sentences verbatim (no network access used); the argument is internally sound
  and matches the standard reading of it. **Its practical conclusion is now
  overtaken by `buzz-selector-curves`**: on this data the choice of selector
  does not move the score at all, because the plateau is wide.
- **§4's "same regime" argument is the weakest part.** It reasons from Cawley &
  Talbot's benchmark bias magnitudes to this pipeline's, and then infers that
  the log's 0.02-0.04 effects are in danger. The direct measurement in this
  repo says the `val_loss` bias here is **-0.002**, i.e. two orders below the
  effects being compared. The concern is legitimate *for `val_sens`* and was
  correctly acted on; extending it to the whole log is not supported.
- **§5's truncation confound is real and I reproduce it.** `tools/honest_epoch.py`
  caps the pooled epoch at the shortest other-fold curve, and on the
  early-stopped runs that cap binds on **4 or 5 of 5 folds** in 13 of 17 runs
  (`cv_baseline` 5/5, `context_embedder` 4/5, `perch_probe` 5/5,
  `harmonic_comb_r2` 5/5). `perch_probe` is the worst case as the document says
  (0.244 → 0.169). The `*` marker in the tool's output is the right guard and
  should be read every time.
- **§6's two residuals are honestly stated, and residual 1 is the reason to
  prefer `--fixed-epochs`.** The document admits the design "is not fully
  nested" — fold `t`'s audio sits in the training pool of the models whose
  curves choose `t`'s epoch — and prices full nesting at 4x compute.
  `--fixed-epochs` removes the residual for free, because no curve is consulted
  at all. This is the strongest single argument in the audit and the document
  contains its premise without drawing the conclusion.
- **Luke's arbitrariness objection is answered correctly** for the design under
  discussion (there is no validate *role* to assign; all k-1 contribute equally),
  and the honest downgrading of the Lostanlen citation (6 identical sensors, one
  night, one area, versus twelve crops / four sites / four years) is good
  practice. Keep.

---

## What changed because of evidence postdating the document

1. **`--fixed-epochs` exists** (2026-09-10, `exp/yamnet-aves-head-fixed`,
   `03_train/{main,train}.py`). Trains every rotation for exactly N epochs, no
   early stopping, no restore-best, ships the final weights. Zero epoch
   selection.
2. **It is measurably as good as `xfold`** on all six fixed-budget runs on disk
   (table in the verdict), while being strictly simpler and strictly stricter.
   The only scenario where `xfold` wins is a config that overfits *within* the
   budget — and no config on disk does: even at e400, `xfold_context`'s pooled
   curve is 0.292 against a 0.295 peak at e234.
3. **The fixed-budget effect has been independently measured twice at +0.031
   and +0.040**, on two different embedders, both times with `1_150` supplying
   the bulk and the healthy folds moving by less than a per-fold SD.
4. **`buzz-selector-curves` closed the open lead negative** (claim 6).
5. **Three entries in the log already run under the new-style rule** and were
   read correctly against matched controls, which weakens the "mixed log is
   uninterpretable" argument for ending the era.

---

## Recommendations

**Do now (done in this audit):**

1. Edit `new-era.md` to correct claims 1, 2, 4, 6 and to mark step 2
   (`--epoch-rule xfold` as default) superseded by `--fixed-epochs`. Done.
2. Edit `epoch-selection.md` §4 and §7 for the measured `val_loss` bias and the
   closed lead. Done.
3. Amend, not rewrite, the `xfold-epoch` entry in `log.jsonl` with the
   decomposition and the fixed-epochs supersession. Done.
4. Add the budget-dependence qualifier to IDEAS.md's `+0.021` and to the
   `--monitor val_sens` standing fact. Done.

**Do next (needs a run):**

5. **One long fixed-budget run on the current best config** — `yavf_h1024` or
   `yavx_h1024` at `--fixed-epochs 400`. Every 150-epoch run on disk is still
   rising at its cap, so the era's best numbers are budget-limited by an
   unknown amount. This is the highest-value cheap experiment the document
   points at and the only one it gets wrong. It also settles the budget for
   everything after it.
6. **A fixed-budget `cv_baseline`** as the comparator anchor, once (5) fixes the
   budget. Expect ~+0.02 to +0.04, chiefly `1_150` — and note that under the
   document's original falsifier this would have read as "something else is
   going on."

**Needs Luke's judgement, not an agent's:**

7. **Whether to archive the era.** The data and fold roster are unchanged; the
   loop has already absorbed one rule change inside this log by using matched
   controls. Archiving buys a clean anchor and costs cross-era comparability on
   unchanged data. I have not archived anything and have not run
   `tools/archive_era.py`.
8. **What the default should be.** My reading is `--fixed-epochs <N>` with N set
   by (5), `--epoch-rule xfold` retained as a diagnostic, and `--monitor
   val_sens` removed outright. But "remove `early`" is a destructive change to
   a rule every archived number was produced under, and that is a call for the
   person who owns the log.

## Things I could not verify

- Whether the h1024 configs plateau before e250. No run exceeds 150 epochs on
  them; offline re-scoring cannot extrapolate a curve.
- The verbatim Cawley & Talbot quotations in `epoch-selection.md` §3 and §4. No
  network access was used. The argument is sound as stated and I am not
  disputing it, only its application scope.
- The document's -0.008 / +0.185 figures for the healthy-fold cost and the
  `1_150` repair. Both `folds_sx.csv` and the persisted curves give -0.004 /
  +0.172. I could not reconstruct a path that yields the document's pair.
- The two `pwr_lam*` runs in flight were deliberately not read or disturbed.
