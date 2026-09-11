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

## Results

## Conclusion
