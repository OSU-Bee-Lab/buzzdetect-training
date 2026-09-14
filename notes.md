# pitchshift-decimate-up

## Hypothesis

IDEAS.md item 5b: `yamnet_pitchshift`'s x2 up-shift rung (+0.069 over
`cv_baseline_v3`, 8/8 folds, twice confirmed) creates a real artifact on the
way up — it resamples a 0.96 s frame down to 7680 samples then **tiles** it
back to 15360, so the shifted block is 0.48 s of real audio played twice with
a hard seam at the midpoint. This experiment removes the tile: it decimates
30720 real, contiguous samples (frame t and its true successor frame t+1) down
to 15360, so YAMNet sees a genuinely-shifted 1.92 s of real content with no
repeat and no seam. Falsifier (pre-registered in IDEAS.md): if the x2/x4
rungs' hard-fold pattern — particularly the read on `1_95`/`1_114` — doesn't
change once the seam is removed, the seam wasn't doing the work and this is a
purity improvement, not a result mover.

Built via `context_frames = 1` (the mechanism `context-frames-fix` put on
`main`): `embed()` gets real contiguous audio, padded across chunk boundaries
by one true frame of audio from the snip, and reads the raw samples for the
decimation directly out of that buffer rather than reconstructing them from
label-bucketed cache rows. No cross-frame leak.

## Changes

New embedder `embedders/yamnet_pitchshift_decimate/` (written to the shared
tree, `embedders/.gitignore` updated to unignore it). Subclasses
`EmbedderYamnet`, `context_frames = 1`, 2048-d (1024 unshifted + 1024
real-decimated-up block). Unit-verified the unshifted block is byte-identical
to plain `yamnet` and the shifted block differs from it. One fresh `medium`
extraction (2122 s, no crashes). Config matched to `pitchshift-repeat`:
`--translation general --fixed-epochs 400`, no dropout.

**Pre-existing repo issue hit and worked around, not fixed**: a clean
extraction of any embedder currently fails `read_fold_roles` on
`Luke - Various Opportunistic Recordings/2026-07-27/1_99` (in `folds.csv`,
dropped from `annotations.csv`) — flagged in `context-frames-fix`'s log entry
as passing elsewhere only because of an "empty leftover directory". Created
the matching empty directory under this embedder's `raw/` cache so training
could proceed; did not touch `01_annotate/` or any annotation file.

## Results

vs `yamnet_pitchshift` (the x2 rung this replaces the shifted block of):

| fold | x2 sens@fpr0.005 | decimate (this run) | delta | delta SD (eval sampling) | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.479 | 0.470 | -0.009 | 0.014 | 32 |
| 53 | 0.433 | 0.442 | +0.009 | 0.021 | 28 |
| willard/1_11 | 0.446 | 0.426 | -0.020 | 0.029 | 26 |
| wooster/1_143 | 0.541 | 0.479 | -0.062 | 0.031 | 22 |
| 1_150 | 0.352 | 0.303 | -0.049 | 0.048 | 21 |
| **1_95** | **0.076** | **0.128** | **+0.052** | **0.021** | 46 |
| 1_37 | 0.430 | 0.419 | -0.011 | 0.033 | 14 |
| **1_114** | **0.431** | **0.465** | **+0.034** | **0.030** | 28 |

mean sens@fpr0.005 (excl. quiet): 0.398 -> 0.391 (delta -0.007 +/- 0.011,
headline eval-sampling SD). Inclusive figure 0.325 -> 0.323.

Tiers: loud 0.837 -> 0.788 (thin, ~115-122 frames), untagged 0.430 -> 0.418,
background 0.326 -> 0.334, quiet 0.088 -> 0.085, faint 0.0 -> 0.0 — broadly
flat, not a tier-specific story.

**The falsifier does not cleanly fire.** Headline is flat (well inside
`pitchshift-repeat`'s own r1-r2 spread of -0.007), but the *pattern* named in
the falsifier moved exactly where predicted: `1_95` +0.052 +/- 0.021 (~2.5
sigma) and `1_114` +0.034 +/- 0.030 (~1.1 sigma), both up, against `1_143`
(-0.062 +/- 0.031, ~2 sigma) and `1_150` (-0.049 +/- 0.048, ~1 sigma) both
down. Read as: removing the tile seam trades some of the rich-fold gain for
movement on the era's two named hard folds, `1_95` in particular — the fold
`yamnet_aves` is the only other intervention ever to move (per IDEAS.md's
standing table). Single-draw; no repeat yet, and per-fold resolution is ~4x
coarser than the headline (LOOP.md), so "unsure but suggestive" is the honest
read on `1_114`, and even `1_95`'s 2.5 sigma is one draw with no training-
stochasticity term measured.

No eval-sampling SD available against `cv_baseline_v3` directly (predates
`predictions.csv` retention); comparisons above are against `yamnet_pitchshift`
(x2), the mechanistically matched control.

## Conclusion

A lateral move on the headline (-0.007, inside noise) that reshapes which
folds carry the gain: real hard-fold movement on `1_95` (the jet-flyover fold,
the era's most stubborn) at the cost of two rich folds. Worth a repeat draw
before trusting `1_95`'s move, per LOOP.md's hard-fold-gain protocol — see
`exp/pitchshift-decimate-up-repeat`.
