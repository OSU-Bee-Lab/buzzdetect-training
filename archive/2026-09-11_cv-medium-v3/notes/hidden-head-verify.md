# hidden-head-verify

## Hypothesis

IDEAS.md item 1c: re-verify E3's `yamnet-aves-head-fixed` result (a shared
1024-wide ReLU hidden layer between the input and the class logits was worth
~+0.030 over the bare linear probe, measured on `yamnet_aves` at a matched
150-epoch budget) against the cv-medium-v3 era anchor. Per LOOP.md's
change-one-thing rule and the queue note ("run it on plain YAMNet against
cv_baseline_v3 first ... the E3 result was measured on yamnet_aves rather than
on YAMNet, so it does not transfer for free"), this run tests the hidden head
on plain YAMNet only. `yamnet_context` and `yamnet_aves` are next.

Expectation entering the run: some gain, plausibly smaller than the E3 figure
since that number was never measured on this representation.

## Changes

Ported just the `--hidden` mechanism from `exp/yamnet-aves-head-fixed` onto
current `main` (that branch predates the era cutover and also reverts the
tier/fixed-epochs machinery that's now the anchor's premise, so a straight
rebase would have fought itself). `03_train/main.py` gets a `--hidden` int
flag (default 0 = current decoupled head); `03_train/train.py` threads it
through `train_set` → `_train_one` and inserts
`Dense(hidden, relu) [+ Dropout(dropout) if set]` between the input dropout and
the final `Dense(n_classes)` when `hidden > 0`. Recorded in
`config_model.json`. Smoke-tested the architecture (build + one `fit()` step)
before spending the CV on it, per LOOP.md.

## Run

```
03_train/main.py --name hidden_yamnet_verify --set medium --embedder yamnet \
  --translation general --hidden 1024 --dropout 0 --fixed-epochs 400 --verbose -y
```

Matched to `cv_baseline_v3` exactly except `--hidden 1024`. CPU only
(`CUDA_VISIBLE_DEVICES=""`). First fold ~27 min; 8 folds, ~3.7 h total. Carried
on a per-fold Monitor per LOOP.md's gate — no handoff needed.

## Results

| fold | anchor sens_exclquiet | hidden(1024) | delta |
|---|---|---|---|
| 1_150 | 0.266 | 0.213 | **-0.053** |
| 1_37 | 0.369 | 0.338 | -0.031 |
| 1_114 | 0.250 | 0.226 | -0.024 |
| 1_95 | 0.052 | 0.062 | +0.010 |
| 1_143 | 0.461 | 0.486 | +0.025 |
| 53 | 0.429 | 0.479 | +0.050 |
| 1_29 | 0.441 | 0.493 | +0.052 |
| willard (1_11) | 0.368 | 0.440 | +0.072 |

5 folds up, 3 down. **Headline: 0.330 -> 0.342 (+0.012), inside the ~0.027 MDE.**
Inclusive `sensitivity`: 0.269 -> 0.283 (+0.014).

Pooled tiers (same threshold both arms): `background` 0.322 -> 0.380 (+0.058,
consistent, both folds carrying it up), `quiet` 0.070 -> 0.089 (+0.019),
`untagged` 0.350 -> 0.362 (+0.012, and *mixed*: up at 1_29/53/willard/1_143,
down at 1_150/1_37/1_114), `loud` 0.778 -> 0.786 (+0.008, flat, thin: 122
frames total).

**Hard folds, the named target, split rather than move together.** `1_150`
(the low-SNR-positives hard fold) went *down* -0.053, on its `untagged` column
specifically (0.266 -> 0.213) — the opposite of a detection gain. `1_95` (the
jet-flyover fold) barely moved, +0.010, still near chance. Neither of the two
named hard folds shows the kind of gain that would make this a clear
structural win.

## Conclusion

**Marginal and mixed, not a confirmed gain.** +0.012 headline is inside the
era's own MDE (~0.027), 5/3 fold split with no consistent hard-fold story
(`1_150` down, `1_95` flat), and the one consistent tier movement
(`background` +0.058) is concentrated in the two folds (`1_29`, `53`) whose
sensitivity is substantially about continuous drone detection rather than
discrete buzz — see LOOP.md's standing caveat on those two folds.

This does **not** reproduce E3's `yamnet-aves-head-fixed` +0.030 on plain
YAMNet, which is consistent with the queue's own caveat that that number was
measured on `yamnet_aves`, not YAMNet, and might not transfer. The wide hidden
head's case rests entirely on whether it helps *on top of* a concat
representation (`yamnet_context` or `yamnet_aves`) — run those next before
concluding anything about `--hidden` in general. Not banking this as a
hyperparameter result either way; it's structural in the sense of changing
what's learnable, but the read is "no clear effect on plain YAMNet," not
"negative."

No stopping-rule confound: both arms ran the identical `--fixed-epochs 400`,
so every fold in both arms trained to the same cap with no early selection.
