# aug-snr15-baseline

## Hypothesis

Luke's prompt (2026-09-25): raw-audio augmentation has a track record here and
in prior eras that keeps coming back negative or neutral, which is surprising
against the literature, and is worth digging into rather than just repeating
the same recipe.

Reading the actual history (`archive/2026-08_cv-medium-v1/log.jsonl` +
notes): the negative results are not clean negatives of "augmentation doesn't
work here" — each carries a specific, identified confound:

- `white-noise` (E1) and the historical `NoiseSpec` defaults (`prop` in
  {0.05, 0.075, 0.2}) add noise at a **fixed absolute amplitude** that sits
  13-25 dB *above* the median buzz frame's RMS (measured in `aug-snr-noise`,
  E2) — training "buzz" as broadband static, not a useful augmentation.
  Consistent with a run of bioacoustic-augmentation literature this session
  found: AWGN added at the spectrogram level *hurt* on a small labeled set in
  at least one directly comparable study (search: "SNR-relative additive
  noise augmentation rare event bioacoustic detection small dataset").
- `aug-snr-noise` itself (E2) fixed the SNR mismatch (noise scaled per-frame
  to 15 dB below signal) and got a materially different, near-neutral result
  (-0.007 vs the old recipe's clear harm) — but that -0.007 was itself
  confounded: augmentation roughly doubles frames/epoch, and the era's
  `min_delta`-based early-stopping rule fired 3-6x sooner on several folds,
  undertraining them. The folds that trained a normal number of epochs were
  net *positive* (+0.072, +0.031, +0.028, +0.022).
- `trunk-ft-1e5-aug` (E3) retried noise augmentation on a fine-tuned trunk and
  got a clear negative (-0.060 paired, 4/5 down) — but `val_loss` spiked on
  the augmented pool and `restore_best_weights` locked a near-random
  best-epoch-1 snapshot on 2 of 5 folds. Same stopping-rule failure mode as
  `aug-snr-noise`, worse because fine-tuning is more sensitive to it.
- `embed-mixup` (E2, frozen probe) found manifold mixup doesn't rescue
  augmentation either, with a literature-grounded reason: augmentation gains
  in the literature are mostly measured at encoder-training time; a low-
  capacity linear probe over a frozen general embedder has little frame-level
  nuisance variance to regularise away.

**This era removed early stopping outright** (`--fixed-epochs`, fixed budget,
no `restore_best_weights`, 2026-09-17). That is exactly the failure mode that
contaminated both `aug-snr-noise` and `trunk-ft-1e5-aug`. Nobody has re-run
SNR-relative noise augmentation since that rule changed. This run is a clean,
single-variable retest of `aug-snr-noise`'s exact design (`NoiseSpec(snr_db=15)`,
all-rotate folds, no cross-fold mixing) under the era's current rules, on the
plain frozen-YAMNet probe — isolating "does the era-3 stopping-rule confound
explain the old near-neutral read" from "does fine-tuning capacity matter"
(the latter is `aug-snr15-trunk`, a separate run).

**Tooling status:** `augment_specs.NoiseSpec.snr_db` and `02_set/augment.py
--all-folds`/`--snr` did not exist on main (the SNR fix lived only in
`aug-snr-noise`'s now-archived worktree, and `--all-folds` was a documented
follow-up that was never landed). Ported both onto main directly
(`5909de4`), smoke-tested on `lite`/`yamnet` (5 folds, correct `(1024,)`
float32 embeddings) before running on `medium`.

**Falsifier:** if the headline is still flat-to-negative here (not just
smaller than the old confounded read), the stopping-rule story is wrong and
augmentation is genuinely inert (or harmful) for this data even unconfounded
— a real, now-clean negative worth logging plainly rather than re-litigating.
A positive delta clearing the era's ~0.012-0.016 headline noise floor means
the old confound was hiding a real, if modest, gain.

**Comparator:** `cv-baseline-v3-refresh` (0.329), the standing plain-YAMNet
comparator this era.

## Changes

- Landed `NoiseSpec.snr_db` and `augment.py --all-folds`/`--snr` on main
  (see commit message for detail).
- Extraction: `02_set/augment.py --set medium --embedder yamnet --all-folds
  --snr 15` (matches `aug-snr-noise`'s spec exactly: SNR 15 dB below each
  frame's own RMS, applied to all labels, no cross-fold mixing since it's a
  per-frame transform run independently per fold).
- Training: `03_train/main.py --name aug-snr15-baseline --set medium
  --embedder yamnet --translation general --augment augment_noise_snr15.0 -y`

## Results

```
tools/results.py <cv-baseline-v3-refresh> aug-snr15-baseline
```

| fold | baseline | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.443 | 0.444 | +0.001 | 0.011 | 32 |
| 53 | 0.440 | 0.424 | -0.016 | 0.009 | 28 |
| 1_11 | 0.368 | 0.415 | +0.047 | 0.022 | 26 |
| 1_143 | 0.459 | 0.468 | +0.009 | 0.020 | 22 |
| 1_150 | 0.231 | 0.250 | +0.019 | 0.026 | 21 |
| 1_95 | 0.042 | 0.038 | -0.004 | 0.009 | 46 |
| 1_37 | 0.399 | 0.395 | -0.004 | 0.021 | 14 |
| 1_114 | 0.252 | 0.232 | -0.020 | 0.019 | 28 |

Headline: 0.329 → 0.333 (+0.004 ± 0.006), inclusive 0.268 → 0.272 (+0.004).
Tiers: loud flat (+0.000, n=122), untagged +0.006 (n=2418), background -0.010
(n=1874), quiet +0.008 (n=627), faint flat (n=12, both zero).

No fold moves past ~2.1σ (1_11's +0.047 ± 0.022 is the largest, weak). No
fold collapses. The headline delta (+0.004 ± 0.006) sits well inside the
era's ~0.012-0.016 headline noise floor.

## Conclusion

Falsifier does not clear either direction: this is a genuinely flat result,
not the "stopping-rule confound was hiding a real gain" outcome the
hypothesis's more optimistic branch predicted, but also not the "SNR-fixed
noise is still harmful" outcome. Removing the early-stopping confound
(compare to `aug-snr-noise`'s own -0.007, itself confounded, with 4 folds
undertrained 3-6x) moves the read from "leaning negative, contaminated" to
"clean, flat, ± 0.006" — a real result now, not noise from a training-budget
artifact. No tier shows a real lift (loud flat, untagged barely moves).

This is consistent with `embed-mixup`'s standing explanation: a frozen linear
probe over a general-purpose embedder has little frame-level nuisance
variance worth regularising away, so SNR-matched noise augmentation is inert
here — not harmful (the old fixed-amplitude recipe's actual failure mode),
just genuinely unhelpful at this capacity. `aug-snr15-trunk` tests whether
that changes once part of the encoder is being fine-tuned.

Trust: **clean** — single run, no seed control (per LOOP.md, a flat/negative
result doesn't need a repeat; only a large gain does), but the measurement
itself carries no known defect.
