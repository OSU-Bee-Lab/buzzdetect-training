# pitchshift-context

## Hypothesis

IDEAS.md item 16: stack context onto the pitch-shift block. Two of the era's
best-confirmed levers look mechanistically distinct -- pitch-shift is a
frequency-register transform of a single frame's own audio, context is
temporal neighbour-stacking -- so there's no obvious *a priori* reason to
expect them to be redundant. Built as `[yam(t-1), yam(t), yam(t+1),
shifted(t)]` (4096-d): only the unshifted half is widened with context,
mirroring `yamnet_context_aves`'s choice to widen the plain-YAMNet block and
leave the second, already-differently-encoded block alone.

Pre-registered read (from IDEAS.md, not a go/no-go falsifier): if hard folds
hold near pitchshift's own level while rich folds gain further, this is the
best config this era. If hard folds regress toward context's flat/down
pattern while rich folds gain, it's still a net headline win but should be
written up as two separate wins stacked, not a fix for `1_150`/`1_95`.

## Changes

New embedder `embedders/yamnet_pitchshift_context/` (shared tree). Subclasses
`EmbedderYamnetPitchshift`, `context_frames = 1`, 4096-d. Unit-verified: the
shifted quarter-block is byte-identical to plain `yamnet_pitchshift`'s shifted
half, the centre unshifted block matches it too, and the left-context block at
row i equals the unshifted block at row i-1 (clamped at edges). One fresh
`medium` extraction (2444 s, no crashes). Same pre-existing `1_99` workaround
as `pitchshift-decimate-up` (empty leftover directory, no annotation edits).
Config matched to `pitchshift-repeat`: `--translation general --fixed-epochs
400`, no dropout.

## Results

vs `pitchshift-repeat` (plain `yamnet_pitchshift`, the matched control):

| fold | control sens@fpr0.005 | this run | delta | delta SD (eval sampling) | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.468 | 0.580 | +0.112 | 0.043 | 32 |
| 53 | 0.436 | 0.520 | +0.084 | 0.048 | 28 |
| 1_143 | 0.533 | 0.616 | +0.083 | 0.043 | 22 |
| 1_37 | 0.439 | 0.445 | +0.006 | 0.058 | 14 |
| willard/1_11 | 0.430 | 0.421 | -0.009 | 0.042 | 26 |
| 1_150 | 0.331 | 0.306 | -0.025 | 0.036 | 21 |
| **1_95** | 0.076 | 0.045 | -0.031 | 0.019 | 46 |
| **1_114** | 0.414 | 0.368 | -0.046 | 0.028 | 28 |

mean sens@fpr0.005 (excl. quiet): 0.391 -> 0.413 (**+0.022 +/- 0.015**,
headline eval-sampling SD -- ~1.5 sigma). Inclusive figure 0.319 -> 0.339
(+0.020).

Tiers: `background` 0.324 -> 0.460 (+0.136 -- the two rich folds, `1_29`/`53`,
are the only ones carrying `_background` annotations, so this single tier
explains most of the gain), `loud` 0.837 -> 0.867, `untagged` 0.422 -> 0.427
(flat), `quiet` 0.084 -> 0.093.

## Conclusion

**The pre-registered second branch fires, for a third time this era.**
`yamnet-aves-context` (context on `aves`) and `hidden-context-verify`
(`--hidden` on `context`) both showed rich-fold gain paired with hard-fold
cost when context was stacked onto something else; this is the same shape a
third time. `1_29`/`53`/`1_143` gain strongly (+0.112/+0.084/+0.083, ~1.9-2.6
sigma each) while `1_95`/`1_114`/`1_150` all move down (-0.031/-0.046/-0.025,
~0.7-1.6 sigma). The headline is a genuine improvement (+0.022 +/- 0.015) of
about the same size as `context-frames-fix`'s honest `yamnet_context` result
(+0.030 +/- 0.015), and most of it traces to the `background` tier -- i.e.
mostly the continuous-drone annotations concentrated in `1_29`/`53`, not
newly-caught discrete buzz.

Read plainly, per the idea's own pre-registered framing: **this is two
separate wins stacked, not a fix for the hard folds.** `pitchshift_decimate`
(a same-era, independently-confirmed lever) moves `1_95`/`1_114` in the
opposite direction context does here, so combining pitch-shift-decimate with
context is very unlikely to help those two folds either -- do not chase that
combination expecting a hard-fold win. If this config is carried forward as a
headline improvement, log it with its true character (rich-fold, background-
tier gain) rather than as a general detection improvement. Single draw; no
repeat run given how closely this matches an already-established pattern
across three independent stacking experiments this era.
