# Starting the next era: cross-fold epoch selection

Written 2026-09-09 on `exp/xfold-epoch`. **Luke approved the finding but is not
ready for the refactor.** This file is the handoff, so whoever picks it up does
not have to re-derive any of it. Nothing here is urgent; the current era stays
valid until someone runs step 3.

> **AUDITED AND SUBSTANTIALLY REVISED 2026-09-11.** The mechanism this file
> found is real; its diagnosis and its prescription were both wrong.
> Corrections are inline and marked. Full evidence:
> `exp/pairwise-rank:notes/new-era-audit.md`. In one paragraph:
>
> - The gain is **not** removal of selection leakage. Across all 17
>   `val_loss`-stopped runs on disk, `shipped − xfold-pooled` averages
>   **-0.002** — there is no measurable leak in the `val_loss` arm. The gain is
>   removal of **undertraining**: early stopping fires at epoch 5-32 on
>   `1_150` under every embedder tried, and scores that fold on a
>   barely-trained probe.
> - Because the cause is undertraining rather than selection, the fix is a
>   **fixed epoch budget**, not a cleverer selector. `--fixed-epochs`
>   (added 2026-09-10 on `exp/yamnet-aves-head-fixed`) does no epoch selection
>   at all, and measured offline on every fixed-budget run on disk it is within
>   0.006 of `--epoch-rule xfold` everywhere and **better on the very config
>   this file was written from** (0.294 at e250 vs xfold's 0.288). It also
>   dissolves the "not fully nested" residual in `epoch-selection.md` §6 for
>   free. **Step 2 below is superseded.**
> - **Budget 250 is a floor, not a settled value.** It was measured on
>   `yamnet_context` only. Every 150-epoch run on disk — `yavf_h0/h256/h1024`,
>   `yavx_h1024` — is still *rising* at its cap.

## Why this ends the era rather than re-baselining it

The change is to **how every fold's score is produced**, not to a model. Every
number in `log.jsonl` was produced by scoring each fold at an epoch chosen from
that same fold. Under the new rule the epoch comes from the other four. That
moves every entry — measured at **+0.030 on `yamnet_context`**, and it will not
be a constant offset, because the size depends on whether that config's early
stopping happened to fail (see "insurance, not improvement" in `../notes.md`).

So a mixed log would be uninterpretable: a config's number would depend on which
rule was in force when it ran. `LOOP.md`'s standing rule already covers this —
a change that moves every number ends the era.

> **Corrected 2026-09-11.** Three problems with the paragraphs above.
>
> 1. **"Every number ... an epoch chosen from that same fold"** was true on
>    2026-09-09 and is false now: `yamnet-aves-head-fixed` and
>    `yamnet-aves-context` ran under `--fixed-epochs`, which selects no epoch
>    at all. It also blurs two mechanisms that measure an order of magnitude
>    apart — `val_loss` selection (a *different* statistic; measured optimism
>    **-0.002** over 17 runs) versus `--monitor val_sens` (the *same*
>    statistic; +0.006 at a 150-epoch budget, +0.021 at 400). Only the second
>    is leakage. README's "bounded, and small" claim for `val_loss` is
>    empirically **confirmed**, not refuted.
> 2. **"That moves every entry"** overstates what +0.030 was. Paired, on the
>    one config it was measured on: the four folds whose early stopping picked
>    a sane epoch (61-155) contribute **-0.003** to the headline mean, and
>    `1_150`/apple — early-stopped at epoch **6** of 400 — contributes
>    **+0.034**. 110% of the headline is one fold. The honest sentence is:
>    *the rule repairs a stopping failure where one occurred, and costs a few
>    thousandths where it did not.* Which entries need re-reading is therefore
>    diagnosable per-entry from `best_epoch`, not "all of them".
> 3. **"A mixed log would be uninterpretable"** has since been tested and did
>    not happen. Three entries ran under a different rule inside this log and
>    read correctly, because each was compared against a **matched control run
>    under the same rule** (`yavf_h0`, `yavf_h1024`), never against
>    `cv_baseline`. Matched controls, not a log boundary, are what make a
>    stopping-rule change survivable. Whether to archive anyway is a judgement
>    call about the convenience of the anchor, not a forced consequence — and
>    the training data is unchanged, which is the usual era-ender.

## What the new rule is, exactly

1. **Train a fixed epoch budget with no early stopping.** `--epoch-rule xfold`
   already does this. **Budget 250** — measured cost against the pooled peak is
   -0.001 at e250, -0.004 at e150, -0.016 at e100. 400 works but wastes ~40%.

   > **Amended 2026-09-11 — this point survives and is the whole result;
   > the budget number does not transfer.** The e250/-0.001 figures reproduce
   > exactly on `xfold_context` (3072-d, linear, pooled argmax e234). They were
   > measured on that config alone. Every other fixed-budget run on disk sits
   > at 150 epochs and is **still climbing at the cap**: pooled mean over the
   > last 21 epochs minus e100-120 is +0.009 (`yavf_h0`), +0.005 (`yavf_h256`),
   > +0.014 (`yavf_h1024`), +0.011 (`yavf_h1024_r2`), +0.008 (`yavx_h1024`),
   > and the pooled argmax lands *on* the 150 cap for `yavf_h0` and
   > `yavx_h1024`. The e100 cost is **-0.026** on `yavf_h1024`, 1.6x the
   > -0.016 quoted here. Wider inputs and a hidden layer both push the optimum
   > later. **Treat 250 as a floor and verify the pooled curve has turned over
   > before trusting a number at any budget.** Whether the h1024 configs
   > plateau before 250 is untested — nothing on disk runs them past 150.
2. **Score fold `t` at the argmax of the mean of the OTHER folds' sens curves.**
   Never `t`'s own curve. Implemented as `train._xfold_epoch`.

   > **Superseded 2026-09-11 by `--fixed-epochs`.** Score every fold at the
   > final epoch of the budget — no curve consulted, no selection performed.
   > Measured offline on every fixed-budget run on disk (final epoch vs
   > `xfold` pooled): `xfold_context` **0.294** (at e250) vs 0.288,
   > `yavf_h0` 0.276 vs 0.275, `yavf_h256` 0.284 vs 0.281, `yavf_h1024` 0.313
   > vs 0.312, `yavf_h1024_r2` 0.300 vs 0.303, `yavx_h1024` 0.321 vs 0.318 —
   > within 0.006 everywhere, and ahead on 4 of 6 including the config this
   > file was written from. The mechanism: these curves rise roughly
   > monotonically to the end of the budget, so "run longer" is uniformly good
   > while "land on a pooled argmax" can stop short of a fold's own optimum.
   > `xfold` earns its complexity only against a config that overfits *within*
   > the budget, and none on disk does (`xfold_context` at e400 is 0.292
   > against a 0.295 peak at e234). Keep `--epoch-rule xfold` as a diagnostic
   > and as insurance; do not make it the default.
3. **Scoring itself does not change.** `sens@fpr0.005` per fold, plain mean over
   folds, `folds_sx.csv`. Do not touch `metrics.py` or `sx.py`. The metric was
   never the problem; using it as a *selector on the scored fold* was.
4. **`--monitor val_sens` stays dead.** Measured inflation **+0.021**.

   > **Amended 2026-09-11 — verdict holds, number is budget-dependent.** The
   > mechanical finding reproduces exactly: `own-peak == shipped` to three
   > decimals on every `val_sens` run and on no `val_loss` run. But
   > `own-peak − xfold-pooled` is +0.021 only at a **400**-epoch budget; on the
   > five 150-epoch runs it is +0.002 / +0.002 / +0.007 / +0.007 / +0.012,
   > mean **+0.006**. The leak scales with how many epochs the argmax ranges
   > over. Quote it with its budget, and do not carry +0.021 into a discussion
   > of a 150-epoch run.
5. **Shipped model:** its epoch should come from the pooled sens curve over all
   five folds, replacing `_consensus_epoch`'s val_loss consensus. **Not yet
   implemented** — `_consensus_epoch` is untouched on this branch.

   > **2026-09-11:** under a fixed budget the shipped model simply trains the
   > same budget. `_consensus_epoch` is bypassed, not replaced. That is what
   > `--fixed-epochs` already does.

## The steps

1. **Merge `exp/xfold-epoch`** (or re-implement; the diff is small and confined
   to `03_train/{train,callbacks,main}.py`).
2. **Flip the default** to `--epoch-rule xfold` and set `--epochs 250`. Consider
   removing `early` rather than leaving two rules that produce incomparable
   numbers; if it stays, it is a diagnostic, not an option.

   > **Superseded 2026-09-11.** Flip the default to `--fixed-epochs N` instead
   > (see point 2 above). `--epoch-rule xfold` stays as a diagnostic. The
   > argument for retiring `early` stands and is strengthened — but it is a
   > destructive change to the rule every archived number was produced under,
   > so it is Luke's call, not an agent's.
3. **Archive the era:** `python tools/archive_era.py --slug 2026-09_cv-medium-v2`
   (preflight; `--write` to do it). It refuses while any branch is unpushed or a
   worktree is dirty. Write the era README's prose yourself — it should say that
   what ended the era was the epoch rule, not a data change, which is unusual and
   means **the training data is unchanged across this boundary.** That is worth
   stating: structural verdicts from the old era carry over better than they
   would across a data revision.

   > **2026-09-11: not compelled by the evidence, and it is Luke's call.** See
   > the correction above — the log has already absorbed one rule change via
   > matched controls. The cheaper alternative is a **fixed-budget
   > `cv_baseline`** as a second anchor in the same log, plus a standing rule
   > that any comparison must use a control run under the same epoch rule
   > (which is what the loop has in fact been doing since 2026-09-10). Nothing
   > has been archived and `tools/archive_era.py` has not been run.
4. **Establish the new baseline.** Two runs, both at 250 epochs:
   - `cv-baseline` on `yamnet` — the new anchor.
   - `context-embedder` on `yamnet_context` — expected near **0.288**.
   Budget ~21 min and ~35 min respectively (measured 1.69 s/epoch at 3072-d;
   ~1.0 s/epoch at 1024-d).

   > **2026-09-11: still the right first two runs, but do them under
   > `--fixed-epochs`, and settle the budget first.** The highest-value run is
   > actually a third one that comes before both: **`yavf_h1024` or
   > `yavx_h1024` at `--fixed-epochs 400`**, because every 150-epoch run is
   > still rising and the era's best numbers are budget-limited by an unknown
   > amount. Fix the budget there, then re-anchor. Expect `yamnet_context`
   > near 0.29 (0.294 at e250 by offline replay, 0.288 under xfold).
5. **Re-run the cheap embedder comparisons** if you want them comparable:
   `aves` (~19 min) and `perch` (~16 min) at 250 epochs. Both caches exist, so
   no re-extraction. `perch` especially — its 5.0 s framelength means ~14k
   frames, and it was the run most distorted by truncated re-scoring.

   > **2026-09-11: `perch`'s distortion is confirmed** — `honest_epoch.py` caps
   > its pooled epoch on **5 of 5** folds and reads it 0.244 → 0.169, the worst
   > truncation artifact in the log. The re-run is worth it; treat the 0.169 as
   > an estimator artifact and the 0.244 as confounded by frame-length
   > dilution (its own entry), i.e. **no usable number exists for perch.**

## What to expect, and what would falsify it

- `yamnet_context` should land near **0.288**, with `1_150` (apple) near
  **0.199** rather than 0.014. If apple comes back near 0.014, the epoch rule is
  not being applied — check that early stopping is really off.
- The new baseline on plain `yamnet` should move **less** than `yamnet_context`
  did. The whole gain came from repairing a stopping failure specific to wide
  inputs; 1024-d YAMNet did not have one (its early-stop epochs were 70-174).
  If plain YAMNet jumps +0.03 too, something else is going on.

  > **WITHDRAWN 2026-09-11 — this falsifier is wrong and would mislead.** The
  > "70-174" range silently excludes `1_150`, the fold the whole argument is
  > about. On plain 1024-d YAMNet, `1_150`'s `best_epoch` is **14-32** in every
  > `val_loss` run on disk (`cv_baseline` 26/70, `base_r2` 23/69, `base_r3`
  > 20/67, `harmonic_comb_r2` 14/64, `L3_batch4096` **2**/52, …) while the
  > other four folds stop at 65-180. Same pathology, same fold, milder degree.
  > It is **fold-specific first and width-specific second**: `1_150`'s
  > multi-class `val_loss` bottoms out almost immediately under every
  > representation tried, while its buzz curve climbs to e120-185.
  > Corroboration at an intermediate width: `yav_h0` (1792-d) early-stopped
  > 0.236 → `yavf_h0` fixed-150 **0.276**, `+0.040`, with `1_150` 0.000 →
  > 0.160 and the four healthy folds *also* up (+0.010 mean). `aves-mlp-head`
  > separately recorded "converged YAMNet linear offline 0.248 vs shipped
  > 0.218". **A plain-YAMNet fixed-budget run should be expected to gain ~+0.02
  > to +0.04, mostly at `1_150`.** Under the original falsifier that reads as
  > "something else is going on"; it is not.
- Expect **small negative** per-fold deltas on folds whose early stopping was
  already healthy. That is the rule working as designed, not a regression.

  > **2026-09-11: true of `xfold`, not of `--fixed-epochs`.** Under `xfold` the
  > healthy folds average -0.004 (not -0.008; see the audit). Under a fixed
  > budget they averaged **+0.010** on `yavf_h0`, because running longer helps
  > a rising curve whereas a pooled argmax can land before a fold's own
  > optimum. Another reason to prefer the fixed budget.

## Entries that need re-reading under the new rule

- `probe-grid`, `context-monitor`, `context-monitor-r2` are marked `artifact`.
  Correct in direction, **overstated in magnitude**: `monitor-leakage` re-scored
  them with truncated curves. The honest value of the composed config is 0.288,
  not the 0.263/0.264 recorded there. Amended 2026-09-09.
- `monitor-leakage` itself: its mechanical finding (peak == at_best on every
  `val_sens` fold) is exact and stands. Its *magnitudes* are truncation-limited.
- `aves-probe` (0.074) and `perch-probe` (0.244) were both run under the old
  rule on wide-ish inputs. Neither verdict should be carried into the new era
  without a re-run — this is the same trap the old `aves-probe` fell into.

> **2026-09-11, narrowing this list.** The set needing a re-run is not "the
> era"; it is **the entries whose `best_epoch` shows a stopping failure**, which
> is already a standing per-entry check in `LOOP.md` and `IDEAS.md`. The 17
> `val_loss` runs' mean `shipped − xfold-pooled` is -0.002, so most entries are
> not displaced at all. The ones that are:
> `aves-probe` (best_epoch collapse documented in its own amendment),
> `aves-mlp-head` (folds stop at e6-40), `perch-probe` (5/5 truncation caps),
> `yamnet-aves-head` (already superseded by `yamnet-aves-head-fixed`), and the
> three `val_sens` entries already marked `artifact`. Everything else can be
> read as logged, with a matched control.

## Open lead to fold in at the cutover

The rule costs **-0.008** on folds whose early stopping was healthy, because a
pooled epoch is a compromise. A lower-variance *buzz-only* selector (buzz AUC or
buzz cross-entropy instead of `sens@fpr0.005`) might close that gap. See
`IDEAS.md`, "A buzz-only, low-variance selection statistic" — **note that the
premise was measured and did not hold in raw terms**, so it is a lead, not a
plan. Cheap to test: `callbacks.SensAtFPR` already computes the buzz activations
every epoch.

> **CLOSED NEGATIVE 2026-09-11 — do not pick this up.** `buzz-selector-curves`
> (`log.jsonl`, committed `549c1e5`) closed it 71 seconds after this file was
> committed, which is why this section exists. Under a non-leaking cross-fold
> rule, all four candidate selectors — `val_loss`, `sens@fpr0.005`, `auc_buzz`,
> `ce_buzz` — return **identical** per-fold sensitivities
> `[0.20, 0.41, 0.38, 0.04, 0.09]`, because the pooled sens curve is flat over
> e129-164 and every in-plateau selector scores the same. The premise (AUC is
> lower-variance) had already failed IDEAS' bootstrap. The corollary is the
> useful part: **the sensitivity is not in which selector you use, it is in the
> budget.** Also the cost being patched is -0.004, not -0.008.

## Background you should not re-derive

`notes/epoch-selection.md` on this branch: why the epoch-smoothing estimate of
selection bias was invalid, the variance-not-bias argument for a selector
(Cawley & Talbot 2010), why `monitor-leakage`'s re-scoring was confounded by
truncation, and Luke's objection that rotating CV roles still embeds an
arbitrary choice — plus the two residuals this design does **not** fix (it is
not fully nested, and the fold roster itself is a tiny sample of a very
heterogeneous population).

> **2026-09-11:** read that file with its own 2026-09-11 header. §2 (smoothing
> is not an estimate of argmax inflation) and §5 (the truncation confound, 4-5
> of 5 folds capped in 13 of 17 runs) are confirmed and worth keeping. §4's
> "same regime" extrapolation from Cawley & Talbot to the whole log is not
> supported by the measured -0.002. And §6's residual 1 — "the design is not
> fully nested" — is the argument that retires `xfold` in favour of
> `--fixed-epochs`, which has no residual because it consults no curve.
