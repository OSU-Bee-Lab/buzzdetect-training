# aug-snr15-trunk

## Hypothesis

Companion run to `aug-snr15-baseline` (frozen probe). `trunk-ft-1e5-aug` (E3)
tried noise augmentation on top of a fine-tuned trunk and got a clear
negative (-0.060 paired, 4/5 down), but `val_loss` spiked on the augmented
pool and the era's `restore_best_weights` early stopping locked a
near-random epoch-1 snapshot on 2 of 5 folds — the identical stopping-rule
failure mode that contaminated `aug-snr-noise`'s frozen-probe result, worse
here because fine-tuning is more sensitive to a distribution shift in the
training pool. That confound is gone this era (`--fixed-epochs`, no early
stopping, no `restore_best_weights`, since 2026-09-17). Nobody has retried
noise augmentation on a fine-tuned trunk since.

Separately, `embed-mixup`'s literature-grounded argument (augmentation gains
are mostly measured at encoder-training time; a frozen linear probe has
little frame-level nuisance variance to regularise away) predicts
augmentation should matter *more*, not less, once part of the encoder
(`yamnet_trunk_pitchshift_depth12`'s layers 12-14) is actually being trained
— this run is the direct test of that prediction, on the era's current best
structural lever instead of E3's plain fine-tuned trunk.

**Falsifier:** if augmentation is still flat-to-negative here even unconfounded,
that's real evidence against the capacity argument, not just a stopping-rule
artifact — log it plainly. A clear positive delta (past the ~0.012-0.016
headline noise floor) confirms augmentation needs training capacity to help
here, matching `embed-mixup`'s reasoning and the wider literature it cites.

**Comparator:** `trunk-pitchshift-depth12`'s own control chain — read against
`trunk-pitchshift-depth12-repeat`'s two-draw mean (0.454) as the matched
fine-tuned-no-aug baseline, and against `aug-snr15-baseline`'s own delta to
see whether the aug effect (if any) is bigger or smaller once fine-tuning is
in play.

## Changes

- Extraction: `BUZZDETECT_CHUNK_FRAMES=48 02_set/augment.py --set medium
  --embedder yamnet_trunk_pitchshift_depth12 --all-folds --snr 15` (same
  SNR-relative noise spec as `aug-snr15-baseline`, same all-rotate-fold
  coverage, no cross-fold mixing). Shares `aug-snr15-baseline`'s
  noise-augmented raw-audio cache (same `audio_cache_key`: 0.96 s / 16 kHz),
  so this only pays for re-embedding through the trunk, not re-noising.
- Training, matching `trunk-pitchshift-depth12`/`trunk-depth-headtohead`'s
  fine-tuning convention exactly (`--epochs 60`, not the default 400 —
  the trunk family's own budget):
  `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 03_train/main.py
  --name aug-snr15-trunk --set medium --embedder
  yamnet_trunk_pitchshift_depth12 --translation general --epochs 60
  --augment augment_noise_snr15.0 -y`

## Outcome: abandoned, never run (2026-09-25)

Noise augmentation completed (`augment_snr15.log`, exit 0), but training died at
startup at 10:42 with no folds and no `[launch_job] exit` line; the last log line
was a 7.4 GB host allocation (probable host OOM on the doubled training pool).
Not relaunched: the era closed the same day for the Hard Negatives annotation
revision, and the run must use the old annotations to compare against
`trunk-pitchshift-depth12`. Given `aug-snr15-baseline` was flat on the frozen probe,
Luke chose to abandon rather than hold the cutover ~6 h.

**Open lead, not a verdict:** noise augmentation on a fine-tuned trunk has still
never been tested without the stopping-rule confound. If retried in a later era,
expect host-RAM pressure from the augmented pool; check memory before launching.
