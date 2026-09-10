# Starting the next era: cross-fold epoch selection

Written 2026-09-09 on `exp/xfold-epoch`. **Luke approved the finding but is not
ready for the refactor.** This file is the handoff, so whoever picks it up does
not have to re-derive any of it. Nothing here is urgent; the current era stays
valid until someone runs step 3.

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

## What the new rule is, exactly

1. **Train a fixed epoch budget with no early stopping.** `--epoch-rule xfold`
   already does this. **Budget 250** — measured cost against the pooled peak is
   -0.001 at e250, -0.004 at e150, -0.016 at e100. 400 works but wastes ~40%.
2. **Score fold `t` at the argmax of the mean of the OTHER folds' sens curves.**
   Never `t`'s own curve. Implemented as `train._xfold_epoch`.
3. **Scoring itself does not change.** `sens@fpr0.005` per fold, plain mean over
   folds, `folds_sx.csv`. Do not touch `metrics.py` or `sx.py`. The metric was
   never the problem; using it as a *selector on the scored fold* was.
4. **`--monitor val_sens` stays dead.** Measured inflation **+0.021**.
5. **Shipped model:** its epoch should come from the pooled sens curve over all
   five folds, replacing `_consensus_epoch`'s val_loss consensus. **Not yet
   implemented** — `_consensus_epoch` is untouched on this branch.

## The steps

1. **Merge `exp/xfold-epoch`** (or re-implement; the diff is small and confined
   to `03_train/{train,callbacks,main}.py`).
2. **Flip the default** to `--epoch-rule xfold` and set `--epochs 250`. Consider
   removing `early` rather than leaving two rules that produce incomparable
   numbers; if it stays, it is a diagnostic, not an option.
3. **Archive the era:** `python tools/archive_era.py --slug 2026-09_cv-medium-v2`
   (preflight; `--write` to do it). It refuses while any branch is unpushed or a
   worktree is dirty. Write the era README's prose yourself — it should say that
   what ended the era was the epoch rule, not a data change, which is unusual and
   means **the training data is unchanged across this boundary.** That is worth
   stating: structural verdicts from the old era carry over better than they
   would across a data revision.
4. **Establish the new baseline.** Two runs, both at 250 epochs:
   - `cv-baseline` on `yamnet` — the new anchor.
   - `context-embedder` on `yamnet_context` — expected near **0.288**.
   Budget ~21 min and ~35 min respectively (measured 1.69 s/epoch at 3072-d;
   ~1.0 s/epoch at 1024-d).
5. **Re-run the cheap embedder comparisons** if you want them comparable:
   `aves` (~19 min) and `perch` (~16 min) at 250 epochs. Both caches exist, so
   no re-extraction. `perch` especially — its 5.0 s framelength means ~14k
   frames, and it was the run most distorted by truncated re-scoring.

## What to expect, and what would falsify it

- `yamnet_context` should land near **0.288**, with `1_150` (apple) near
  **0.199** rather than 0.014. If apple comes back near 0.014, the epoch rule is
  not being applied — check that early stopping is really off.
- The new baseline on plain `yamnet` should move **less** than `yamnet_context`
  did. The whole gain came from repairing a stopping failure specific to wide
  inputs; 1024-d YAMNet did not have one (its early-stop epochs were 70-174).
  If plain YAMNet jumps +0.03 too, something else is going on.
- Expect **small negative** per-fold deltas on folds whose early stopping was
  already healthy. That is the rule working as designed, not a regression.

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

## Open lead to fold in at the cutover

The rule costs **-0.008** on folds whose early stopping was healthy, because a
pooled epoch is a compromise. A lower-variance *buzz-only* selector (buzz AUC or
buzz cross-entropy instead of `sens@fpr0.005`) might close that gap. See
`IDEAS.md`, "A buzz-only, low-variance selection statistic" — **note that the
premise was measured and did not hold in raw terms**, so it is a lead, not a
plan. Cheap to test: `callbacks.SensAtFPR` already computes the buzz activations
every epoch.

## Background you should not re-derive

`notes/epoch-selection.md` on this branch: why the epoch-smoothing estimate of
selection bias was invalid, the variance-not-bias argument for a selector
(Cawley & Talbot 2010), why `monitor-leakage`'s re-scoring was confounded by
truncation, and Luke's objection that rotating CV roles still embeds an
arbitrary choice — plus the two residuals this design does **not** fix (it is
not fully nested, and the fold roster itself is a tiny sample of a very
heterogeneous population).
