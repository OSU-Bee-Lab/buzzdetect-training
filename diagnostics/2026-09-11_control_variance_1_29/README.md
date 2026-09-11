# Control variance at n=3, and an unexplained +0.105 at `1_29`

2026-09-11. No training beyond the third control draw itself; everything else is
offline re-scoring of runs already on disk.

## Why

`yamnet-aves-context` and `pairwise-rank` were mechanically unrelated — one a
3840-d context representation, one an auxiliary loss term that never touches the
representation — yet both put fold `1_29` at ~0.57 against a control whose two
draws sat at 0.464/0.450. A third independent control draw (`yavf_h1024_r3`,
identical config, no seed control) was run to test whether the control was the
outlier.

## The control is not the outlier

`yavf_h1024` = `yamnet_aves`, `--hidden 1024`, `--fixed-epochs 150`,
`--translation general`, three independent draws:

| fold | r1 | r2 | r3 | mean | SD |
|---|---|---|---|---|---|
| `1_29` | 0.464 | 0.450 | 0.468 | 0.461 | 0.009 |
| `53` | 0.491 | 0.476 | 0.495 | 0.488 | 0.010 |
| `willard` | 0.294 | 0.287 | 0.274 | 0.285 | 0.010 |
| `1_150` | 0.236 | 0.215 | 0.201 | 0.218 | 0.017 |
| `1_95` | 0.078 | 0.071 | 0.061 | 0.070 | 0.009 |
| **headline** | **0.313** | **0.300** | **0.300** | **0.304** | **0.007** |

**Use these numbers, not `probe-grid`'s.** That `n=3` SD of 0.0095 was measured
on the `cv_baseline` config (linear probe, plain YAMNet, early stopping). This
one is measured on the config the era actually runs now, and it is the right
comparator for anything built on `yavf_h1024`. Note `1_150`'s 0.017 is the
largest per-fold SD and its three draws trend monotonically down (0.236 → 0.215
→ 0.201), which is worth watching but is n=3.

## The anomaly

Against that three-draw control, `yamnet-aves-context` is:

| fold | delta | in units of the control's own SD |
|---|---|---|
| `1_29` | **+0.105** | **11x** |
| `53` | +0.014 | 1.4x |
| `willard` | -0.011 | 1.1x |
| `1_150` | -0.016 | 0.9x |
| `1_95` | -0.008 | 0.9x |

And `pairwise-rank` lands `1_29` at 0.570 (λ=0.5) and 0.575 (λ=2.0) — a 4x dose
change moves it 0.005 — while being monotonically *harmful* on all four other
folds. So three treatment runs cluster at 0.566-0.575 (spread 0.009) and three
control runs at 0.450-0.468 (spread 0.018), with no overlap.

## What it is not

Four explanations tested and rejected:

1. **Not control noise.** Three draws, SD 0.009 at this fold; the gap is 11x it.
2. **Not undertraining.** All six runs peak at e72-131 on `1_29` and are
   *declining* at the 150 cap. The gap is already fully present at **e50**
   (control 0.472/0.446/0.441, treatments 0.587/0.558/0.553). This is the one
   place the era's "every run is still rising at its cap" finding does **not**
   apply — it is true of the pooled curve and false of this fold.
3. **Not a code difference between worktrees.** `03_train/{dataset,callbacks,sx,
   metrics}.py` are byte-identical between the control worktree and both
   treatment worktrees; `train.py` differs only by the documented rank-loss
   additions, which are inert at `lam=0`.
4. **Not one long drone event crossing the threshold.** This was the expected
   answer, since seven annotations in a single 300 s file supply 63% of this
   site's buzz seconds, which would make its sensitivity quasi-quantized. It is
   wrong: the 259 frames `pairwise-rank` gains sit in **192 separate runs, longest
   run 5 frames**. Diffuse, not one event.

## What it is

Unexplained. The two treatments reach the same sensitivity by different
distributional routes and on substantially different frames:

| run | threshold | buzz mean | buzz median | buzz−neg mean gap | detected / 2046 |
|---|---|---|---|---|---|
| ctrl r1 | -1.391 | -1.341 | -1.532 | 1.652 | 959 |
| context | -1.458 | -1.095 | -1.130 | 1.925 | 1174 |
| rank λ=0.5 | -1.333 | -1.433 | **-0.843** | **5.860** | 1172 |

`context` lifts the buzz distribution bodily. `rank` leaves its mean *below* the
control's while raising its median by 0.69 and crushing the negatives. Of the
frames each gains over the control, only **120 of ~255 are shared (47%)** — yet
they land within two frames of each other in total detected.

So: two unrelated interventions each buy ~+0.11 at `1_29`, via largely different
frames, by different distributional mechanisms, reproducibly, while one of them
is harmful everywhere else.

## What to do with it

- **Do not read "+0.11 at `1_29`" as evidence for whatever produced it.** A
  treatment that is monotonically harmful on four folds produces the same
  effect, so the effect is not diagnostic of the treatment.
- **`1_29` is not a good fold to carry a headline.** It is the buzz-richest
  fold and the one `IDEAS.md` flags as trading systematically against `53`
  across `shared-trunk-head`'s whole width ladder. Report it, but do not let it
  decide an adopt/reject.
- **The open question is what the control is failing to do at `1_29` that any
  perturbation fixes.** Worth one dedicated diagnostic before any more compute
  goes into configs whose gains live on this fold.
