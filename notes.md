# pairwise-rank

## Hypothesis

`mech-margin` closed as a clear negative, monotone across a 64x dose ladder,
and its conclusion names its own successor:

> the margin is specified in absolute logit space while the metric reads only
> RANK (every fold is thresholded on its own held-out audio), so driving a
> subpopulation below a fixed logit buys nothing... Any future confuser penalty
> should be PAIRWISE (buzz frame above confuser frame), not absolute.

The diagnostic that makes this concrete: under `mech-margin` the `mech_auto`
*share* of threshold-setting false positives ROSE with the dose (26.5% → 32.1%
→ 39.3%) while absolute logits fell. Both populations sank; buzz sank faster.

So: same target, rank-shaped. For each buzz frame, sample `n_pairs` confuser
frames and penalise `softplus(conf - buzz + margin)` on the `ins_buzz` logit.
Adding a constant to every logit leaves the term unchanged — verified exactly
before launch — which is the invariance the metric has and the hinge lacked.

**Target: `1_95`**, at 0.062-0.078 the worst fold in the era. It is a
false-positive problem, not a positives problem: p99.5 of its negatives is
0.391 against ~0.139 elsewhere, its threshold is -0.401 against -1.59..-1.90
for every other fold, and **32 of the 35 frames that set that threshold are
`mech_auto`**. Luke on this failure: "this must not happen."

**Falsifier:** if `1_95` does not move at either dose, the plainer reading of
`mech-margin` is right — this is a loss fix to a representation problem, the
populations do not separate in the frozen embedding at all, and the
buzz-vs-`mech_auto` lead is closed for good rather than merely reshaped.

**Why now, and why the two historical failure modes do not apply:**

- `tail-loss` (E2) was a ranking loss whose failure was a **monitor artifact** —
  batch-local ranking makes the compiled loss invalid as `val_loss`, and early
  stopping read it. `--fixed-epochs 150` uses no monitor at all, so that mode
  cannot fire. This flag did not exist when `tail-loss` ran.
- IDEAS proposed pairing against a **frozen reference model** to make pairing
  deterministic per frame. At this pipeline's `size_batch` of 65568 against
  ~72k training frames there are ~2 steps per epoch, so a batch is most of the
  pool and in-batch pairing is already near-global. The extra machinery buys
  little; noted as the fallback if the result is noisy rather than null.
- `shared-trunk-head` tested whether a non-linear head unlocks this and found
  `1_95` flat at every width — but it tested the **head alone, with no confuser
  term**. The two have never been run together, and this runs on `h1024`.

## Changes

- `03_train/train_utils.py`: `pairwise_rank_loss()`. `lam=0` reduces exactly to
  `weighted_bce_loss` (verified bit-equal), so the control is the existing arm.
- `03_train/main.py`: `--rank-lam`, `--rank-confuser`, `--rank-margin`.
- Branch is main + `exp/yamnet-aves-head-fixed`'s `--hidden`/`--fixed-epochs`
  patch, as `exp/yamnet-aves-context` was: those flags never merged to main and
  the matched control was trained with them.

**`tools/smoke_model.py` caught a real failure and is why this did not burn a
CV.** The first implementation selected pools with `tf.boolean_mask`, which
produces a bounded-dynamic dimension (`f32[<=n]`) that XLA's dynamic padder
refuses to broadcast into an add — it fails at compile time inside `model.fit`
with `Dynamic input unexpectedly found for unsupported instruction`. Rewritten
to sample via `tf.random.categorical` over the membership mask, keeping every
shape static.

Pre-launch checks beyond the smoke gate: `lam=0` bit-identical to the stock
loss; shift-invariance exact under +5 on every buzz logit; degenerate batches
(no buzz, no confuser, all-ambiguous) contribute exactly 0; correctly-ordered
pairs cost strictly less than inverted ones.

## Run

```
03_train/main.py --name pwr_lam05 --set medium --embedder yamnet_aves \
  --translation general --hidden 1024 --fixed-epochs 150 --rank-lam 0.5 --verbose -y
03_train/main.py --name pwr_lam20 ... --rank-lam 2.0
```

Comparator is **`yavf_h1024` / `yavf_h1024_r2`** (two-draw mean 0.307, `1_95`
0.078/0.071) in `.local/worktrees/yamnet-aves-head-fixed/models/` — identical
embedder, head and epoch budget, `lam=0`.

## Related audit

This branch also carries `notes/new-era-audit.md` — an audit of
`exp/xfold-epoch:notes/new-era.md` (the proposal to end the era on cross-fold
epoch selection), written this session. Verdict: the mechanism is real but the
diagnosis and prescription were both wrong — the distortion is undertraining,
not selection leakage, and `--fixed-epochs` supersedes `--epoch-rule xfold`.

## Results

Against the matched control's two draws (`yavf_h1024` 0.313 / `_r2` 0.300):

| fold | buzz frames | ctrl r1 | ctrl r2 | λ=0.5 | λ=2.0 | dose response |
|---|---|---|---|---|---|---|
| `1_29` | 2144 | 0.464 | 0.450 | 0.570 | 0.575 | up, flat in dose |
| `53` | 1031 | 0.491 | 0.476 | 0.444 | 0.360 | **monotone down** |
| `willard` | 305 | 0.294 | 0.287 | 0.203 | 0.176 | **monotone down** |
| `1_150` | 146 | 0.236 | 0.215 | 0.132 | 0.099 | **monotone down** |
| **`1_95`** | 433 | 0.078 | 0.071 | **0.049** | **0.022** | **monotone down** |
| **mean** | | **0.313** | **0.300** | **0.280** | **0.246** | |

-0.033 at λ=0.5 and -0.067 at λ=2.0, **1 up / 4 down at both doses**, monotone in
the dose on four of five folds.

## Conclusion

**CLEAR NEGATIVE, MONOTONE IN THE DOSE, AND THE PRE-REGISTERED FALSIFIER FIRED
ON THE TARGET FOLD.** `1_95` — the fold this was built for, where 32 of 35
threshold-setting frames are `mech_auto` — goes 0.075 (control mean) → 0.049 →
0.022, *down* at both doses and monotone. That is the same shape `mech-margin`
produced, at the same target, in the same direction.

**So the reshaping was not the problem, and the lead is now closed in both of
its forms.** `mech-margin` closed the absolute hinge and its conclusion argued
the failure was one of *specification* — absolute logits against a rank-reading
metric — naming a pairwise term as the fix. That diagnosis is now falsified:
the pairwise term is exactly scale-free (verified: +5 on every buzz logit leaves
it bit-identical), applies to the same class, is fully in-batch over ~2 steps
per epoch, and runs under a budget with no monitor so `tail-loss`'s E2 failure
mode cannot fire. Every objection to the previous attempt was removed, and the
result got worse rather than better.

**What remains is the plainer reading `mech-margin` itself offered and
`shared-trunk-head` independently reached: this is a REPRESENTATION problem.**
In the frozen `yamnet_aves` code the jet and the buzz populations do not
separate — `1_95`'s buzz/non-buzz mean-logit gap is 0.513 against 1.578 at
`1_29`. A loss term can only rotate a readout through a space that already
holds the distinction. No loss-shaped intervention should be spent on `1_95`
again without first showing the separation exists in the input. The measured
next step is `IDEAS.md` item 2: on the held-out jet probe
(`diagnostics/2026-09-09_jet_1_95/`), P(jet frame outranks a buzz frame) is
0.860 for YAMNet and 0.558 for AVES but **0.239 for Perch** — the only cached
representation in which the jet is not preferentially mistaken for buzz.

**Collateral damage is the tell.** The losses are not confined to the target:
`53` -0.131 and `willard` -0.118 at λ=2.0, both monotone. Pushing `mech_auto`
frames below buzz frames costs sensitivity at sites where `mech_auto` was never
the problem, which is what a term does when it rotates a readout away from a
direction that was carrying real signal — the same mechanism `mech-margin`
diagnosed as "rotating away from mech_auto frames rotates away from buzz."

**The one fold that goes up is the one to distrust.** `1_29` lands at 0.570 and
0.575 — a 4x dose change moves it 0.005 — against a control whose two draws sit
at 0.464/0.450. `yamnet-aves-context`, a pure representation change with nothing
mechanically in common with this one, put the same fold at 0.566. Three
unrelated treatments within 0.009 of each other, ~+0.11 above a control whose
own draws differ by 0.014, does not read as three treatments each helping
`1_29`; it reads as `1_29` having a second attractor the control is not in. A
third control draw (`yavf_h1024_r3`, same config, independent draw) was queued
behind this ladder to test exactly that. **Until it reports, no `1_29` gain in
this era should be banked** — including the one inside `yamnet-aves-context`'s
headline.

Budget caveat, per `notes/new-era-audit.md`: all arms ran `--fixed-epochs 150`
and the pooled sens curve is still rising at the cap for the control (+0.019
over the last 40 epochs) though nearly flat for λ=0.5 (+0.002). A longer budget
would therefore widen this negative, not narrow it. The verdict is conservative
as stated.
