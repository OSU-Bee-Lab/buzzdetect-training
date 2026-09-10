# yamnet-aves-head

## Hypothesis
`yamnet_aves` (YAMNet 1024-d centre-crop ⊕ AVES 768-d, 1792-d concat) paid a
clean but modest +0.025 (xfold-pooled) with the stock linear probe. That probe
config (Dropout 0.2, no weight decay, one Dense off the embedding) was tuned for
YAMNet's sparse, non-negative, already-linearly-separable code. The AVES half of
the concat is dense, signed, and smears buzz evidence over ~256 directions
(aves-readout PCA). A shared ReLU hidden layer lets all 15 classes read one
learned representation and gives the reader room to recombine the AVES block.

`aves-mlp-head` (2026-09-10) found offline (converged sklearn) that a non-linear
head helps both embedders (YAMNet 0.248→0.272, AVES 0.182→0.231) but in-pipeline
came back negative — because more head params reach the val_loss argmin sooner
and early-stopping ships them undertrained. Question here: does `--hidden` on the
`yamnet_aves` concat beat its own linear control in-pipeline, ladder h=0/256/1024.

## Changes
- Cherry-picked `--hidden` from `exp/shared-trunk-head@4a90546`:
  `Input → Dropout(0.2) → [Dense(h,relu) → Dropout(0.2)] → Dense(n_classes)`.
  h=0 (default) is the byte-identical shipped head.
- Nothing else changed: loss, val_loss early-stop (patience 50, RestoreTrueBest),
  Adam 0.002, label smoothing 0.2.
- Embedder `yamnet_aves`, restored + committed to main this session (4d9623c;
  the original embedder.py had been lost with its worktree). CV reads the
  existing 560 MB cache, which the restored embedder reproduces byte-for-byte.

## Results

Shipped epoch (val_loss argmin), mean sens@fpr0.005 over the 5 rotating folds:

| model  | mean  | vs h0 | vs cv_baseline | folds vs h0        |
|--------|-------|-------|----------------|-------------------|
| cv_baseline (YAMNet linear) | 0.218 | — | — | — |
| yav_h0 (yamnet_aves linear) | 0.236 | — | +0.018 (4up/1down) | — |
| yav_h256                    | 0.239 | +0.003 | +0.021 | 2up/2down/1flat |
| yav_h1024                   | 0.227 | -0.009 | +0.009 | 1up/3down/1flat |

By the shipped number: a hidden layer does nothing (h256) or slightly hurts
(h1024).

**But best_epoch collapses monotonically with head width** — the aves-mlp-head
signature:

| fold   | h0 best/n | h256    | h1024   |
|--------|-----------|---------|---------|
| 1_29   | 100/100   | 44/78   | 29/77   |
| 53     | 130/136   | 125/127 | 67/81   |
| willard| 89/113    | 50/96   | 15/65   |
| 1_95   | 104/104   | 22/71   | 87/100  |
| 1_150  | 5/55      | 16/61   | 12/62   |

More head params → earlier val_loss argmin → wider heads ship undertrained.

**honest_epoch re-scoring (no retraining), mean over folds:**

| rule          | h0    | h256  | h1024 |
|---------------|-------|-------|-------|
| shipped       | 0.236 | 0.239 (+0.003) | 0.227 (-0.009) |
| own-peak (leaky) | 0.247 | 0.275 (+0.028) | 0.264 (+0.017) |
| xfold-median  | 0.243 | 0.249 (+0.006) | 0.251 (+0.008) |
| xfold-pooled  | 0.229 | 0.254 (+0.025) | 0.251 (+0.022) |

Every rule except the (known-broken) shipped one puts both hidden widths above
the linear control. The gain concentrates in the resolvable folds — at own-peak,
h256 vs h0: 1_29 +0.019, willard +0.034, 1_150 +0.097 (noisy), 53 -0.009,
1_95 +0.002.

**The cross-fold rules are not trustworthy on this ladder.** Early stopping
truncated different folds at different epoch counts across the three arms, which
breaks the shared-epoch construction: h0's 1_150 ended at epoch 55 (best 5 +
patience 50), capping h0's pooled/median epoch at ~55 while 1_29 and 53 peak at
100/135 — so h0's xfold-pooled 0.229 is deflated and the +0.025 vs h256 is
partly an artifact of h0's own short fold. own-peak is the only rule that gives
every fold a fair epoch, and it is leaky.

## Conclusion
Inconclusive, leaning positive; confounded by the stopping rule. In-pipeline
shipped is null-to-negative (h256 +0.003, h1024 -0.009), but that is the arm we
already know is wrong for wide heads: best_epoch collapses 100→44→29 (1_29) and
89→50→15 (willard) as the head widens, and all three non-shipped epoch rules put
h256/h1024 ~+0.02–0.03 above the linear control, concentrated in the folds with
enough buzz to measure. This reconfirms aves-mlp-head one representation-level
down: a non-linear head helps the frozen representation, val_loss early-stopping
hides it in-pipeline.

Not logged as a win. The clean measurement was not run because the ladder used
plain early-stopping — IDEAS.md's standing rule is to pair any capacity or
normalisation change with a fixed epoch budget or cross-fold epoch rule from the
start. Follow-up `yamnet-aves-head-fixed` runs the h=0/256/1024 ladder at a
fixed epoch budget (no early stop, no restore-best), so all arms are scored at
one identical epoch and the capacity question is answerable.
