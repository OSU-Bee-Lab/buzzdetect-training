# context-monitor

## Hypothesis

This era has produced exactly two `clean` gains over `cv_baseline` (0.218), and
neither has been combined with the other:

- **`context-embedder`** (+0.040 → 0.258). `yamnet_context` embeds
  `concat(t-1, t, t+1)` of real neighbouring audio, 3072-d. Its gain is carried
  entirely by the three folds with enough buzz to resolve an effect
  (+0.037 / +0.069 / +0.125 at 305–2144 buzz frames); the two near-chance Diel
  Drivers folds moved -0.007 and -0.023, i.e. nothing.
- **`--monitor val_sens`** (+0.031 → 0.257, `probe-grid`'s L1). Stopping and
  restoring on `val_sens_fpr0.005` instead of `val_loss`. Its gain is carried
  **entirely by `1_150`**, one of those same two near-chance folds
  (0.021 tight over three baseline draws → 0.089 / 0.158).

**The two pay on disjoint folds.** One improves the representation where there
is buzz to see; the other fixes which epoch gets shipped, and only shows up
where the `val_loss` argmin was badly placed. Nothing about either mechanism
routes through the other: `--monitor` changes no features, and the context
stack changes no stopping rule. So the prediction is **additive**, ~0.29, with
the context gain reappearing on the rich folds and the monitor gain on `1_150`.

The falsifier that matters: if the composed run lands at ~0.258 or ~0.257 —
i.e. at whichever component is larger, not their sum — then the two are
competing for the same headroom and at least one of them was really measuring
something about how the *stopping rule interacts with the input scale*, not two
independent effects. A 3072-d input reaches its `val_loss` argmin on a
different curve than a 1024-d one, and `--monitor val_sens` is precisely a
change to which curve is scored.

**Comparator is `models/context_embedder` (0.258), not `cv_baseline`.** One
variable moves: the monitor. The paired per-fold table against
`context_embedder` is the result; the comparison to `cv_baseline` is context
only.

Expected effect (+0.031, if it survives) is above `probe-grid`'s ~0.027
single-run MDE, but not by much — read the per-fold direction, and specifically
whether `1_150` moves, before the headline.

## Changes

Code, both ported unchanged from existing branches, no new logic:

- `embedders/yamnet_context/embedder.py` from `exp/context-embedder`. Not a
  symlink — main has no `yamnet_context` embedder, so it is a real directory on
  this branch. It is `.gitignore`d by `embedders/.gitignore`'s whitelist and
  needs `git add -f`.
- `03_train/{main.py,train.py}` from `exp/probe-grid` (`ProbeConfig` + the
  `--dropout/--label-smoothing/--learning-rate/--weight-decay/--batch-size/`
  `--monitor/--min-delta` flags). Every default is the previously hardcoded
  value, so only `--monitor val_sens` is actually exercised. Checked out by
  path rather than merged: both branches are *behind* main, so a merge would
  revert main's later doc and extractor commits.

No extraction. `02_set/sets/medium/embeddings/yamnet_context` already exists
(982 MB, shared through the symlink). Its ident set matches `yamnet`'s except
for a stale `Luke - External Data Sources/InsectSound1000` that is not in
`folds.csv` at all, so nothing in the run reads it.

Run: `--set medium --embedder yamnet_context --translation general
--monitor val_sens`.

## Results

CV: 5 folds, 16:51:05 → 17:08:05 (17 min, ~4 min/fold). Shipped model not
trained, as intended.

**Primary comparison — paired against `context_embedder` (one variable: the monitor):**

| fold | context_embedder | this exp | delta | buzz frames | best_epoch (base → exp) |
|---|---|---|---|---|---|
| Diel Drivers/1_150 | 0.014 | **0.219** | **+0.205** | 146 | **5 → 160** |
| willard/1_11 | 0.217 | 0.249 | +0.032 | 305 | 28 → 22 |
| Diel Drivers/1_95 | 0.014 | 0.044 | +0.030 | 433 | 121 → 18 |
| JamesU/1_29 | 0.551 | 0.556 | +0.005 | 2144 | 108 → 107 |
| Fit+Fast/53 | 0.494 | 0.465 | -0.029 | 1031 | 144 → 17 |

- mean sens@fpr0.005: **0.258 → 0.307 (+0.049)**, 4 folds up / 1 down.
- against `cv_baseline`: **0.218 → 0.307 (+0.089)**, **5 up / 0 down**.

**The headline is the largest in the era, and the composition prediction held.**
The four numbers that test it:

| | cv_baseline | +context | +monitor | both |
|---|---|---|---|---|
| headline | 0.218 | 0.258 | 0.241 / 0.257 | **0.307** |
| 1_29 (rich) | 0.426 | 0.551 | 0.431 / 0.421 | 0.556 |
| 1_150 (thin, hard) | 0.021 | 0.014 | 0.089 / 0.158 | 0.219 |

Each component keeps its own fold. Context carries the rich folds and the
monitor does nothing there (1_29: 0.426 → 0.431/0.421 alone, i.e. flat);
the monitor carries 1_150 and context alone makes it *worse* (0.021 → 0.014).
Together, both. Sum of the parts is +0.071 against an observed +0.089; the
+0.018 excess is well inside noise, so read this as **additive**, not
synergistic. The falsifier — landing at ~0.258, the larger component alone —
did not happen.

**Mechanism, and it is not fold variance.** `context_embedder` shipped
**epoch 5** on 1_150: under `val_loss` the 3072-d model hit its argmin almost
immediately and the fold was scored on a barely-trained probe (0.014, precision
0.077). With `val_sens` it ships epoch 160 and scores 0.219. That is a
demonstrable stopping failure being fixed, not a draw going the right way, and
it explains why the wider input needed this flag more than YAMNet did — more
dimensions, an earlier `val_loss` argmin, a worse epoch shipped. Note the same
signature at Fit+Fast in the other direction (144 → 17, -0.029): the monitor
moves which epoch ships on every fold, and one of them ships worse.

**What the gain rests on — and why that is the result, not a caveat.** Over the
primary comparison's other four folds the mean delta is +0.0095. The +0.049 is
`1_150`, going 0.014 -> 0.219. That is the fold the project is *for*: the
endpoint is a new deployment, it may well be a quiet one, and `1_29` sits at
~0.43 whatever we do.

The obvious objection is that `1_150` is the 146-buzz-frame fold that moved
0.007 -> 0.062 between two *identical* baseline runs. **That spread is a
within-treatment number and does not bound a between-treatment difference.**
Splitting all 17 runs of this era by which monitor they used:

| monitor | `1_150` | n |
|---|---|---|
| `val_loss` | 0.007, 0.014 x5, 0.021 x4, 0.027, 0.034, 0.041, 0.062 | 13 |
| `val_sens` | 0.089 (L1_sens), 0.158 (L1_sens_r2), 0.171 (L1L6_combo), **0.219** (this) | 4 |

**The two groups do not overlap.** Every `val_sens` run beats every `val_loss`
run on this fold. Combined with the `best_epoch` 5 -> 160 mechanism, this is a
treatment effect and not a fold that swung; the repeat spread describes noise
*within* the `val_loss` group, which is where it was measured.

One repeat is still worth running — confirming a large hard-fold gain is
standing practice with no seed control in the pipeline — but as confirmation,
not because the number is suspect. It would also test the one genuinely n=1
part: this run tops the `val_sens` group by 0.048, which may be an extra
context x monitor lift on `1_150` or may be draw.

1_95 remains near chance (0.044) — the fourth intervention not to move it.
Every fold reached the target FPR; `neg_frames` is 24–35 as always.

## Conclusion

**`--monitor val_sens` and `yamnet_context` compose, additively, to 0.307 —
+0.089 over `cv_baseline` with all 5 folds up, the largest and best-signed
result of the era.** The two act on disjoint folds and neither blocks the
other, which was the hypothesis.

The composed config is the new best-known structure and the right base for the
next experiment. Two follow-ups, in order:

1. **`--monitor val_sens` should become the default** (already the standing
   `probe-grid` follow-up, Luke's call, probably an era boundary). This run
   raises the stakes: the flag is not a small hyperparameter win, it is
   load-bearing for the context embedder, which without it throws away a fold
   to an epoch-5 restore.
2. **One repeat of this config**, as confirmation of a large hard-fold gain
   rather than as a check on a suspect number (see above: the `val_sens` and
   `val_loss` groups do not overlap on `1_150`). 17 minutes, and it resolves
   whether this run's 0.219 — top of the `val_sens` group by 0.048 — carries an
   extra context x monitor lift.
