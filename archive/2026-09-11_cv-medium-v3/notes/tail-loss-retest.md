# tail-loss-retest

## Hypothesis

`exp/tail-loss` (log.jsonl, `trust: artifact`) added an OHEM-style hard-negative
term to the compiled loss and collapsed (0.305 → 0.007) for two implementation
reasons per `IDEAS.md`'s `tail-loss-fair-retest` entry: (1) the OHEM term was
part of the *compiled* loss, so `EarlyStopping` monitored `val_loss` including
the batch-local, noisy tail term, and stopped on a spurious early minimum; (2)
`top_k`'s `k` was data-dependent, hitting a TF graph bug. Fixing both — a
separate `val_bce` monitor via `make_bce_metric`, and a fixed-size `ohem_k` —
should let the mechanism (rank-targeted loss on the hardest negatives, aimed at
the ~20-frame region that actually decides `sensitivity_mean @ fpr0.005`) be
tested honestly. If `best_epoch` lands back in baseline's ~60-130 range and
sensitivity is non-pathological, the tail term itself is the next thing to
evaluate over a full CV.

## Changes

`03_train/train.py`: `TailBCELoss` (smoothed BCE + fixed-size top-`ohem_k`
OHEM term on `ins_buzz`, sample-shaped via `scatter_nd` so `class_weight`
composes normally) and `make_bce_metric` (plain BCE, decoupled from the
compiled loss). `_train_one`'s `EarlyStopping` now monitors `val_bce`, not
`val_loss`. `OHEM_K = 200`, `OHEM_WEIGHT = 1.0` module constants for this
retest. `tools/diag_tailloss.py` (ported from `exp/tail-loss`, `--frac`
replaced by `--k`) drives single-fold runs without the full CV loop.

## Diagnostic path (this is the point of the retest)

**Code check first.** Confirmed in `03_train/train.py`: `EarlyStopping(monitor='val_bce', ...)`,
not `val_loss` — the fix described in `IDEAS.md` is actually in place, not just
described. `tools/smoke_model.py` passes: `TailBCELoss` survives
`compile → fit → save(include_optimizer=True) → load(compile=False)`, which is
the exact round trip `write_model_py.py`'s inference path uses. (Aside, out of
scope: `TailBCELoss.get_config()`/`from_config()` do *not* round-trip under a
*compiled* reload — `Loss.get_config()` emits a `reduction` key the
`__init__` doesn't accept, so `pickle.load`-ing a `history.pickle` — which
pickles the whole `History` object including its `.model`, both here and on
`main` — throws on this custom loss. Irrelevant to the shipped
`compile=False` path smoke-tested above, and pre-existing (`pickle.dump(history, f)`
dumping the full object, not `history.history`, is unchanged from `main`), so
left alone.)

**One-fold run at the shipped config (`k=200, weight=1.0`), fold `JamesU -
MustardBumbler/1_29`** (a70b72b's partial run, `models/diag_k200_w1.0/`):
`best_epoch=36` of 82, `sensitivity@fpr0.005=0.000` (baseline on this fold:
`best_epoch=63` of 113, `sensitivity=0.448`). `val_bce` and `val_loss` tracked
close but distinct curves and both bottom around epoch 35-38 — the monitor
*is* on `val_bce`, correctly, and it is *not* reproducing `exp/tail-loss`'s
literal bug. It is reproducing the collapse anyway. Inspecting
`predictions.csv`: the 30 highest-scoring frames in the validation fold are
*all* negatives — the best true positive ranks below the entire top of the
negative tail. The tail term, still live in the training gradient by design,
pushes down exactly the hard-negative frames — which in an embedding-shared
linear probe are the frames that look most like real buzzes — and appears to
drag genuine positives down with them. That is a mechanism-level effect, not a
monitoring artifact.

**Control test — does plain BCE (`weight=0.0`) reproduce baseline on this
fold, run through the same standalone single-fold harness?** It does not.
Three `weight=0.0` reruns (mathematically identical to `cv-baseline`'s loss)
landed at `best_epoch` 9, 12, and 47 with `sensitivity` 0.002, 0.006 (at
`weight=0.1`), and 0.005 — nowhere near baseline's `best_epoch=63`,
`sensitivity=0.448` for this exact fold. To rule out a bug specific to this
worktree, the same single fold was rerun through **`main`'s unmodified
`03_train/train.py`** (`_load_data`/`_train_one` called directly, no code
changes) via a throwaway script outside the repo: `best_epoch=12`,
`sensitivity=0.070`. Still nowhere near baseline's logged number for the same
fold, same loss, same code.

A second fold (`Luke - Various Opportunistic Recordings/2025-08-12/1_114`,
baseline `best_epoch=129/179`, `sensitivity=0.163`) was also tried at
`weight=1.0`: `val_bce` plateaued by epoch ~24 (`val_accuracy` stuck at
0.0027), far short of baseline's continued improvement to epoch 129, before a
Metal-backend crash (`MPSGraph` placeholder shape assertion) killed the run —
itself evidence of instability in this training regime on this hardware, not
attributable to the loss function.

**Conclusion of the diagnostic:** the standalone single-fold harness this
retest (and `exp/tail-loss` before it) uses for a cheap pre-check does not
reliably reproduce a real CV fold's result — not even for the *null* model
(plain BCE, unmodified `main` code). This is consistent with, but larger in
magnitude than, `exp/std-convergence`'s already-logged finding that this
pipeline has no seed control and that a same-config rerun can land on a
substantially different trajectory purely from random init — here that
variance was large enough to make the "does `best_epoch` land in baseline's
~60-130 ballpark" gate fail even for the control. That gate, as specified,
turned out not to isolate the tail-loss effect from ordinary pipeline noise.

Despite that confound, `weight=1.0` was still measurably worse than every
`weight=0.0` control on the same fold: `sensitivity=0.000` and a fully
inverted score distribution (positives below the entire negative tail) vs.
`0.002-0.070` and, presumably, no such total inversion for the controls (not
independently confirmed for every control run, but the raw sensitivity gap
alone — zero vs. a nonzero floor — recurred across every `weight=1.0` trial
and no `weight=0` trial). A directional signal survives the noise floor: the
live OHEM gradient term measurably harms this fold beyond what undertrained
noise alone produces, even after fixing the two implementation bugs
`IDEAS.md` prescribed.

## Gate result

**Did not pass.** `best_epoch` did not land in baseline's ~60-130 ballpark at
any weight tried (0.0, 0.1, 1.0) on fold `JamesU - MustardBumbler/1_29`, and
sensitivity stayed pathological (≤0.07) throughout — but so did the `weight=0`
control, which should be indistinguishable from `cv-baseline`. Per the
brief's explicit decision tree, this is treated as a legitimate negative and
**no full 11-fold CV was run.**

## Results

No `folds_sx.csv` / `sensitivity_mean` produced — no full CV was run, per the
gate above. `models/diag_k200_w1.0/`, `models/diag_k200_w0.1/`,
`models/diag_control_w0/`, `models/diag_control_w0_r1/`,
`models/diag_k200_w1.0_f2b/` hold the single-fold diagnostic artifacts
described above (the last is a partial/crashed run).

## Conclusion

The two implementation fixes `IDEAS.md` prescribed (separate `val_bce`
monitor, fixed-size `ohem_k`) are correctly in place — confirmed by reading
`_train_one` and by the monitor tracking `val_bce`'s minimum, not
`val_loss`'s, in the recovered `history.pickle`. Despite that, the mechanism
still collapses sensitivity to near-zero on both folds tried, via genuine
score inversion (hard-negative suppression dragging down real positives in a
shared linear probe), not a stopping-rule artifact.

Separately — and this is the more general finding — the single-fold
standalone pre-check this experiment (and `exp/tail-loss` before it) relies on
is **not a reliable proxy for a CV fold's outcome in this pipeline**: run
through unmodified `main` code, the same exact fold reproduces a wildly worse
result than what that fold actually scored inside the real CV. This deserves
its own line in `IDEAS.md` for whoever next reaches for this harness — it can
tell you a config is *categorically* broken (as `weight=1.0` vs. `weight=0`
controls confirms here) but cannot currently be trusted to say a config is
*fine* just because one single-fold run looks reasonable, nor to reproduce a
specific fold's baseline number for direct comparison.

Given the gate failed and the directional signal that is measurable (OHEM
actively harms even relative to the noisy single-fold floor), a full CV was
not run — spending an 11-fold CV to test a mechanism that already shows
active harm at the single-fold level, on top of a noise floor this large,
is not where LOOP.md says compute should go. `tail-loss` as a structural idea
is not recommended for a further retry without first addressing why hard
negatives and true positives collapse onto the same linear-probe direction
(e.g. a term that also protects hard positives, or one computed against a
fixed reference pool rather than the live batch/fold, as `IDEAS.md` itself
suggested as an alternative not yet tried).

- sensitivity_mean @ fpr0.005: not measured (no CV run; gate failed at the
  single-fold pre-check for both the experimental config and its own control)
