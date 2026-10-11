# ps-d8-aves
## Hypothesis
Training arm. One-off test under LOOP.md's AVES injunction, allowed by Luke on 2026-10-11 ("I'll allow
your cheap test this once"). A frozen AVES block next to YAMNet features was a small, repeated gain on
frozen probes in closed eras (`yamnet-aves-verify` +0.024, `aves-mid` +0.032 / +0.025,
`pitchshift-aves-mid` +0.043 / +0.032 over their controls), but it has never been tried on a fine-tuned
trunk, and fine-tuning has since passed it without AVES (`ps-depth8` ~0.50 against a 0.324 baseline).
The question is whether AVES still adds information once the YAMNet tail is trained on the task.

Embedder `yamnet_trunk_pitchshift_depth8_aves`: ps-depth8's two trunk views (bit-identical to
`yamnet_trunk_pitchshift_depth8` on a test clip) plus AVES layers 6, 9, 12 mean-pooled over the same
0.96 s frame (2304 columns, frozen); the class Dense reads [2048 tail code, 2304 AVES]. Everything else
is the ps-depth8 recipe.

Prediction: headline above the same-recipe control `ps-depth8-r3` (0.497; three draws 0.497-0.515) by
more than the headline delta's SD (~0.014), with `untagged`/`loud` up. Falsified if it lands within
~0.014 of the control, or if the gain is only on `1_29`/`53` background drones. `1_114` is the fold to
watch: adding AVES layers to pitch-shift took it down 0.204 in era v3.

Caveat written before the run: the head is ps-depth8's (Dense at lr 2e-4, 30 epochs), tuned for a sparse
non-negative YAMNet code; AVES columns are dense and signed. A null here gets one offline readout sweep
on the cached AVES columns before it is logged as an embedder verdict (LOOP.md, "A new embedder doesn't
inherit the probe config").

## Changes
- Embedder `yamnet_trunk_pitchshift_depth8_aves` (main, commit 1a78e6a): ps-depth8's two trunk views plus
  AVES layers 6, 9, 12 mean-pooled over the same 0.96 s frame (2304 frozen columns); `build_head`
  concatenates them to the pooled tail code before the class Dense.
- Extraction: medium, 31 min on the GPU (`BUZZDETECT_NO_GPU=1 ... 02_set/main.py --set medium --embedder
  yamnet_trunk_pitchshift_depth8_aves --workers 0`).
- Training: the ps-depth8 recipe unchanged (`TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`,
  `--epochs 30`), 16.4 min per fold. Buzz logit SD 0.61-1.33 on every fold: no collapse.
- `offline_aves_readout.py` (this worktree): the readout check below.

## Results
Control `ps-depth8-r3` (same recipe, 2026-10-10; the recipe's other draws are 0.509 and 0.515).

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.486 | 0.618 | +0.132 | 0.049 | 32 |
| 53 | 0.550 | 0.551 | +0.001 | 0.022 | 28 |
| 1_11 | 0.573 | 0.603 | +0.030 | 0.026 | 26 |
| 1_143 | 0.586 | 0.604 | +0.018 | 0.037 | 22 |
| 1_150 | 0.435 | 0.463 | +0.028 | 0.046 | 21 |
| 1_95 | 0.284 | 0.266 | -0.018 | 0.023 | 46 |
| 1_37 | 0.648 | 0.623 | -0.025 | 0.031 | 14 |
| 1_114 | 0.414 | 0.274 | -0.140 | 0.038 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.497 → this 0.500 (+0.003 ± 0.012)
- inclusive (sensitivity), same thresholds: 0.411 → 0.411 (+0.000)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.856 | 0.819 | -0.037 | 122 |
| untagged | 0.524 | 0.514 | -0.010 | 2418 |
| background | 0.406 | 0.503 | +0.097 | 1874 |
| quiet | 0.130 | 0.128 | -0.002 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Headline +0.003 ± 0.012: no change, and inside the control recipe's own three draws (0.497-0.515). The
prediction is falsified on both of its conditions: the delta is within 0.014, and what did move up is
`1_29`'s background drone.

Folds, each against its own SD: `1_29` +0.132 ± 0.049 (up, ~2.7 SD) and `1_114` -0.140 ± 0.038 (down,
~3.7 SD) cancel in the mean. `1_114` is the fold named in advance: AVES layers took it down 0.204 in era
v3, so this is the second time. `1_11` +0.030 ± 0.026, `1_150` +0.028 ± 0.046, `1_143` +0.018 ± 0.037,
`1_95` -0.018 ± 0.023, `1_37` -0.025 ± 0.031, `53` +0.001 ± 0.022: unsure, all within about one SD.

Tiers: `background` +0.097 (`1_29`; `53`, the other background fold, did not move), `loud` -0.037 on 122
frames, `untagged` -0.010. The isolated-buzz tiers the prediction asked for did not rise.

### Offline readout check (LOOP.md, "A new embedder doesn't inherit the probe config")
`offline_aves_readout.py`: buzz-vs-rest logistic readouts on the cached AVES columns alone (2304 columns,
85,394 frames; mean 0.00, column SD ~0.20, row norm ~36), one per rotating fold, scored with
`metrics.sens_at_fpr` at 0.005, quiet/faint frames out. What it measures: whether the head's regime can
read the frozen block, not the joint head (fold weights are not saved, so the trained Dense's AVES share
of the logit was not measured).

| readout on AVES columns only | mean sens | logit SD |
|---|---|---|
| raw, head regime (Adam 2e-4, 30 epochs, label smoothing 0.2) | 0.217 | 0.69 |
| standardized, head regime | 0.231 | 0.69 |
| standardized, converged ridge logistic, l2 1e-4 | 0.277 | 3.00 |
| l2 1e-3 | 0.280 | 2.47 |
| l2 1e-2 | 0.255 | 1.91 |
| l2 1e-1 | 0.216 | 1.34 |

The head regime does read the block (logit SD 0.69 from AVES alone, 0.217), and reaches about 78% of
what a converged, standardized readout gets (0.280). Standardizing is worth +0.014 and convergence about
+0.05, on a block whose best alone is 0.28 against the fine-tuned tail's ~0.50. So the null is not a
collapsed or unread block; a better readout would hand the head a somewhat stronger version of a much
weaker feature. Per fold, AVES alone is weakest exactly on `1_95` (0.08-0.11) and `1_114` (0.11-0.19).

## Conclusion
Null. A frozen AVES block (layers 6/9/12) next to ps-depth8's fine-tuned tail leaves the headline where it
was (0.500 against 0.497, +0.003 ± 0.012). The repeated +0.02 to +0.04 AVES gave frozen YAMNet probes in
closed eras does not carry over once the tail is trained on the task. It is not neutral per fold: `1_29`
+0.132 ± 0.049 with `background` +0.097, `1_114` -0.140 ± 0.038 (the second era AVES layers cost that
fold), `loud` -0.037. One run, one draw, so the two fold moves are single samples; the offline check says
the head could read the block (0.217 alone in the head's regime, 0.280 converged), so this is not a
readout artifact. At 31 min of GPU extraction for no headline gain, the injunction stands on accuracy as
well as speed. IDEAS.md item 29 (AVES2/BEATs) was not run: it is the same question with a ~1.5x faster
encoder and stays under the injunction.
