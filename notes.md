# context-width

## Hypothesis

`exp/context-embedder` (+0.022, clean) stacked each frame's YAMNet embedding
with its real audio neighbours `[t-1, t, t+1]` (3072-d) and shipped. Its notes
flagged "sweep k=2,3" as the untested next step, and never claimed k=1 is the
right width — only that some context beats none.

This experiment runs k=2 only: `[t-2, t-1, t, t+1, t+2]`, 5 frames, 5120-d.
Same mechanism, same code path (`context_frames` in `02_set/extract.py`,
`stack_context` in the embedder), only the width changes.

A concurrent diagnostic (`exp/deployment-forensics`) produced a mechanistic
prediction to test explicitly: `Lily Adam - One Hive/recorders/willard/
2024-08-07/1_11` regressed under k=1 context (-0.074 in context-stack, -0.059
in context-embedder) because it has the highest fraction of short (<1s: 68%)
and isolated (>5s gap: 68%) buzz events of any fold — stacking dilutes a brief
isolated buzz with silent neighbours.

**Prediction: widening to k=2 should hurt willard MORE than k=1 did (more
silent frames folded in around a <1s event), and should help folds with long
sustained buzz more (more context to average over, same signal throughout).**
If that holds, context width should be adaptive/event-length-dependent rather
than a single fixed k — the natural next experiment. If it doesn't hold
(willard flat or improved, no clean split by buzz duration), the mechanistic
story from deployment-forensics needs revision.

Prior: dose-response could go either way past k=1 — more context could keep
helping (more duration evidence) or the willard-style dilution could start
costing more than the sustained-buzz folds gain, making k=1 a local optimum.

## Changes

`embedders/yamnet_context_k2/embedder.py` (new): copy of `exp/context-embedder`'s
`embedders/yamnet_context/embedder.py`, `CONTEXT_FRAMES = 1 -> 2`, class/embeddername
renamed to `yamnet_context_k2` / `EmbedderYamnetContextK2`. No other lines changed.
`n_embeddings` derives automatically (`1024 * (2*2+1) = 5120`).

`02_set/extract.py`: main has NO `context_frames` support at all — that code
only ever existed on `exp/context-embedder`. Brought it onto this branch by
taking `02_set/extract.py` from commit `ade2503` (main's version plus the
context additions, nothing else), committed as `ea4fcbe`. Without this, this
experiment would have silently taken the default (non-context) extraction
path — label-grouped audio, giving every frame same-label "neighbours" — while
looking like it succeeded.

New embeddings dir: `02_set/sets/medium/embeddings/yamnet_context_k2/` (fresh
name; did not touch `yamnet_context`'s existing cache under `medium/embeddings/`).
Note: `embedders/yamnet_context/embedder.py` itself isn't present in main's
working tree (only committed on `exp/context-embedder`), so it was pulled via
`git show exp/context-embedder:embedders/yamnet_context/embedder.py` as the
base to copy from, per the task instructions.

## Results

### Extraction verification (before training)

`yamnet_context_k2` embeddings: 2626 ident dirs matching `yamnet`'s coverage
exactly (1837 pickles, 686 `annotations.fingerprint` files), exit status 0.
Main's pre-existing caches confirmed byte-unchanged after extraction:
`audio/sr16000_fl0.96` 1821 files, `embeddings/yamnet` 1821,
`embeddings/yamnet_context` 1837, `embeddings/yamnet_combined` 1821 — identical
counts before and after.

Content checks, run before training:

- **5120-d on disk**: sampled 15 of 1837 pickle files across idents; all
  `(n, 5120)`.
- **Stacking check, controlled buffer**: fed a clean synthetic 6s buffer
  straight to `embed()` — 100% of adjacent frame pairs satisfy `out[i,
  3072:4096] == out[i+1, 2048:3072]` (frame *i*'s offset+1 slot equals frame
  *i+1*'s centre slot), and edge clamping is exact at both ends (`out[0,
  0:1024] == out[0, 1024:2048] == out[0, 2048:3072]` at the start, mirrored at
  the end).
- **Stacking check, real extracted data**: one label-bucket file (3552
  contiguous-labelled frames, `ambient_background` from a full day's audio)
  gives 3452/3551 adjacent matches; the 99 mismatches cluster into 100 short
  runs (mean length ~35 frames) rather than scattering evenly — exactly the
  signature of a label bucket concatenating many separate coverage-chunk
  stretches from across the day, not a stacking bug.

This rules out the artifact `context-stack` had (same-label neighbours) —
k=2's extraction path is the same contiguous-audio-before-bucketing path
`context-embedder` built and verified for k=1.

### Training

`models/context_k2/folds_sx.csv`: **sensitivity_mean @ fpr0.005 = 0.185**.

| fold | baseline (k=0) | k=1 (context-embedder) | k=2 (this exp) | Δ vs baseline | Δ vs k=1 | frames_val |
|---|---|---|---|---|---|---|
| Various Opportunistic/2025-08-05/31 | 0.157 | 0.297 | 0.013 | -0.144 | -0.284 | 942 |
| Various Opportunistic/2025-06-23/1_23 | 0.326 | 0.314 | 0.194 | -0.132 | -0.120 | 315 |
| willard (One Hive, 2024-08-07/1_11) | 0.177 | 0.118 | 0.066 | -0.111 | -0.052 | 4730 |
| Various Opportunistic/2025-08-12/1_114 | 0.163 | 0.122 | 0.087 | -0.076 | -0.035 | 3768 |
| JamesU - MustardBumbler/1_29 | 0.448 | 0.550 | 0.518 | +0.070 | -0.032 | 6984 |
| Luke - Diel Drivers/2026-05-06/1_95 | 0.027 | 0.042 | 0.014 | -0.013 | -0.028 | 6628 |
| wooster (One Hive, 2024-07-26/1_143) | 0.260 | 0.247 | 0.247 | -0.013 | 0.000 | 4708 |
| Various Opportunistic/2025-08-27/48 (sustained-drone fold) | 0.013 | 0.023 | 0.027 | +0.014 | +0.004 | 1571 |
| Lily - Fit+Fast/2023_R3_Marysville/53 | 0.386 | 0.390 | 0.395 | +0.009 | +0.005 | 4712 |
| Luke - Diel Drivers/2026-04-08/1_150 | 0.028 | 0.068 | 0.075 | +0.047 | +0.007 | 4947 |
| Various Opportunistic/2025-07-03/1_37 | 0.279 | 0.336 | 0.397 | +0.118 | +0.061 | 4715 |

- sensitivity_mean @ fpr0.005: baseline **0.206** → k=1 **0.228** (+0.022) →
  k=2 **0.185** (-0.021 vs baseline, -0.043 vs k=1).
- vs baseline: 5 folds up, 6 down.
- vs k=1: 4 folds up, 6 down, 1 flat (`wooster`, exactly 0.000).

**Willard prediction: held.** The regression is monotonic and grew with
width: baseline 0.177 → k=1 0.118 (-0.059) → k=2 0.066 (-0.111), i.e. widening
to k=2 cost willard roughly double what k=1 did (-0.111 vs -0.059 from
baseline; -0.052 further from k=1 alone). `willard`'s frames_val (4730) is
mid-pack, not a thin fold, and `best_epoch` for its k=2 model is 5 of 55 —
early stopping cut it off fast, consistent with `deployment-forensics`'
mechanistic story: widening the context dilutes willard's short, isolated
(<1s, >5s-gap) buzz events with more silent neighbours, and the model gives up
on that fold earlier.

**Long-sustained-buzz prediction: weak support, not a clean confirmation.**
`deployment-forensics` names exactly one fold with genuinely sustained,
continuous buzz events (median 7.8s, up to 93.4s) rather than short discrete
pulses: `Various Opportunistic Recordings/2025-08-27/48`. That fold does gain
monotonically with width (0.013 → 0.023 → 0.027), the direction the hypothesis
predicted — but the k=1→k=2 increment is tiny (+0.004) and this fold has only
19 annotated buzz events total and 1571 val frames, squarely in the range
README flags as ±0.25-or-worse unreliable. `MustardBumbler` and `Marysville`
(the other two loudest/best-scoring folds) are not sustained-drone folds by
`deployment-forensics`' own characterization, and both are flat-to-slightly-down
under k=2 (MustardBumbler -0.032 vs k=1, Marysville +0.005 vs k=1) — no
sustained-buzz-fold bloc actually appears in the data.

**Non-monotonic shape, and what drove the turnover.** k=1 helped (+0.022,
clean-verified) and k=2 hurts (-0.021 vs baseline) — this is not simply "more
context keeps not mattering," it is a real reversal past k=1. The turnover is
driven disproportionately by two folds that swung hardest in the *opposite*
direction from their k=1 result: `2025-08-05/31` was k=1's single biggest gain
(+0.140 vs baseline) and became k=2's single biggest loss (-0.144 vs baseline,
-0.284 vs k=1); `2025-06-23/1_23` similarly reverses (-0.132 vs baseline
having been roughly flat at k=1). Both are also `best_epoch`-early or thin
folds (`2025-08-05/31`: `best_epoch=1` of 51 — the model barely trained before
early stopping picked it as best; `2025-06-23/1_23`: 315 val frames, the
thinnest fold in the set). `willard`, `1_114`, `1_95`, and `MustardBumbler`
also move down but by smaller, more plausible-as-real amounts, in the direction
the dilution mechanism predicts.

**Noise caveat (added after the fact, from a concurrent finding).**
`exp/tail-loss-retest`, run in parallel today, retrained the *identical*
baseline config on one fold and reproduced 0.070 against a logged 0.448 for
that same fold — i.e. single-fold sensitivity on this set can swing by
0.3+ from seed/init noise alone, with no code or data change at all. That
materially discounts how much weight any single fold's delta here can bear,
especially `2025-08-05/31`'s -0.284 swing (also flagged independently above by
its `best_epoch=1`) and `2025-06-23/1_23`'s -0.132 on 315 frames. The
aggregate direction (majority of folds down against both baseline and k=1) is
still the most defensible read, but the *size* of the turnover, and specifically
how much of it is "real dose-response reversal" vs. run-to-run noise
concentrated on two volatile folds, is not something this single CV can
distinguish.

## Conclusion

Widening temporal context from k=1 to k=2 does not continue k=1's gain — it
reverses it. `sensitivity_mean` @ fpr0.005: baseline 0.206 → k=1 0.228 (+0.022)
→ k=2 0.185 (-0.021 vs baseline). Against k=1 specifically, 6 of 11 folds move
down, 4 up, 1 flat (mean -0.043) — context width is not "more is better," and a
single fixed k is not obviously the right lever to keep pulling in this
direction; k=1 looks like it may already be past or near the local optimum, not
a floor.

The two mechanistic predictions from `deployment-forensics` fare differently:
the **willard prediction held cleanly** — widening cost willard roughly double
what k=1 did (-0.059 → -0.111 vs baseline), monotonic across both widths, on a
mid-sized fold with an early-stopped model consistent with the dilution story.
The **sustained-buzz prediction is only weakly supported** — the one fold
`deployment-forensics` actually characterizes as sustained-drone
(`2025-08-27/48`) does move in the predicted direction but by a
negligible amount on a fold too thin to trust, and no broader bloc of
loud/sustained folds shows the predicted gain. So the willard mechanism (short
isolated events diluted by wider context) looks solid; the complementary claim
that sustained-buzz folds benefit more from width is not confirmed by this run
and would need a fold with more sustained-buzz mass and data to test properly.

**Trust: caveated.** The extraction is clean — verified byte-identical
main caches, correct 5120-d shape, and a direct stacking + edge-clamp check on
both synthetic and real data, so this is not a repeat of `context-stack`'s
same-label-neighbour artifact. But the *size* of the turnover rests partly on
two volatile folds (`2025-08-05/31`, `2025-06-23/1_23`) that also show
independent red flags (an early-stopped, apparently undertrained model at
`best_epoch=1`; a 315-frame fold), and a same-day concurrent finding
(`exp/tail-loss-retest`) shows this set's per-fold sensitivity can swing
0.3+ from seed noise alone with zero code change. The direction — k=2 net
negative against both baseline and k=1, majority of folds down against k=1 —
is the part worth acting on. The exact magnitude of the reversal, and whether
k=2 is "worse than no context at all" by a real margin or is sitting within
unquantified run-to-run noise, is not something one CV run (no seed control)
can settle. Next step, if this line is pursued further: don't sweep k upward
again without either a repeated-seed baseline on the volatile folds or a
per-event-duration-adaptive context width, since a single fixed k trades
willard-style dilution against sustained-buzz gains that this run couldn't
even locate.
