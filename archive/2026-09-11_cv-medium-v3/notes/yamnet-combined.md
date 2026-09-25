# yamnet-combined

## Hypothesis

YAMNet's 521 AudioSet class scores are a semantically meaningful, *nonlinear*
projection of the same audio — "Insect", "Bee, wasp, etc.", "Buzz", "Vehicle",
"Wind" — computed by a head trained on far more data than we have. The probe is
linear, so it can only use structure already linearly available in its input;
class scores are exactly the kind of feature it cannot construct for itself.
Concatenating them with the 1024-d embedding (1545-d total) should therefore
give it signal it can't recover from the embedding alone.

## Changes

None to any code. `embedders/yamnet_combined/` already existed, emits
`concat(embedding, scores)` and declares `n_embeddings = 1545`; `03_train`
sizes its input layer from that attribute and does not normalize inputs, so
nothing downstream needed touching. Verified by extracting and training the
`tiny` set end to end before committing to the medium extraction.

The experiment is an extraction plus a training run:

    python 02_set/main.py --set medium --embedder yamnet_combined --workers 2
    python 03_train/main.py --name yamnet_medium_combined --set medium \
        --embedder yamnet_combined --translation general -y

Slug note: `setup_worktree.sh combined-embedder` fails — `exp/combined-embedder`
is taken by a fixed-test-era experiment that changed validation class scope,
not the embedder. This worktree is `exp/yamnet-combined`, matching the IDEAS.md
entry.

## Results

| fold | baseline sens@fpr0.005 | this exp | delta | val frames |
|---|---|---|---|---|
| JamesU - MustardBumbler/1_29 | 0.448 | 0.441 | -0.007 | 6908 |
| Lily - Fit+Fast/2023_R3_Marysville/53 | 0.386 | 0.378 | -0.008 | 4710 |
| Lily Adam - One Hive/recorders/willard/2024-08-07/1_11 | 0.177 | 0.204 | +0.027 | 4713 |
| Lily Adam - One Hive/recorders/wooster/2024-07-26/1_143 | 0.260 | 0.296 | +0.036 | 4707 |
| Luke - Diel Drivers/2026-04-08/1_150 | 0.028 | 0.069 | +0.041 | 4891 |
| Luke - Diel Drivers/2026-05-06/1_95 | 0.027 | 0.029 | +0.002 | 6606 |
| Luke - Various Opportunistic Recordings/2025-06-23/1_23 | 0.326 | 0.326 | +0.000 | 314 |
| Luke - Various Opportunistic Recordings/2025-07-03/1_37 | 0.279 | 0.305 | +0.026 | 4712 |
| Luke - Various Opportunistic Recordings/2025-08-05/31 | 0.157 | 0.004 | -0.153 | 942 |
| Luke - Various Opportunistic Recordings/2025-08-12/1_114 | 0.163 | 0.194 | +0.031 | 3768 |
| Luke - Various Opportunistic Recordings/2025-08-27/48 | 0.013 | 0.193 | +0.180 | 1570 |

- sens_persite @ fpr0.005: baseline 0.206 -> this 0.222 (+0.016)
- all 11 folds reached the target FPR in both runs (`folds_scored` 11/11, same
  5641 buzz frames scored, same median 22 negative frames per threshold), so
  the two runs are scored on identical footing.

7 folds up, 3 down, 1 unchanged. The two largest movers are both among the
three smallest validation folds and they nearly cancel:

- `2025-08-27/48` +0.180 on 1570 val frames. This is one of the two
  near-chance deployments from IDEAS.md; it went 0.013 -> 0.193. Real movement
  there would matter, but the fold is small and its baseline was essentially
  zero, so a swing of this size is exactly what the README warns per-fold
  sensitivity does in quiet deployments.
- `2025-08-05/31` -0.153 on 942 val frames, the smallest fold with any real
  buzz. Its training run also looks degenerate rather than embedder-driven:
  early stopping fired at epoch 6 with val_loss 0.465 and val_accuracy 0.097,
  against 22 epochs / 0.454 / 0.378 for the baseline. Read this as a bad draw
  in an unseeded pipeline.

Dropping both, the remaining 9 folds still average +0.016. Restricting to the
8 folds with >=3000 validation frames — the ones the README says carry enough
buzz to read — gives +0.019 with 6 up and 2 down, and both declines are under
0.01 while five of the six gains are 0.026 or larger. So the direction is
consistent among the folds worth trusting, and it does not depend on either
wild small fold.

The scale worry in the IDEAS entry is real but did not prevent the probe from
using the scores: the sigmoid class scores run about an order of magnitude
smaller than the embedding activations (score block mean 0.004 / max 0.41 vs
embedding block mean 0.041 / max 2.6 on a sample frame). Nothing normalizes
inputs, so any further work here should try standardizing the two blocks
before concluding the ceiling has been reached.

## Conclusion

Positive but modest: +0.016 on the endpoint, 7/11 folds up, 6/8 among folds
with enough validation buzz to read. About a third of what `context-stack`
bought, and it costs a full re-extraction plus a 1545-dim shipped model, where
context-stack cost neither. With no seed control the per-fold noise is
unquantified and one fold here clearly drew badly, so this is a consistent
direction rather than a demonstrated effect — worth keeping as a candidate, not
worth adopting on this evidence alone.

Next, in order of cost:

1. Standardize the embedding and score blocks before concatenation; the scale
   mismatch is the obvious reason a linear probe would under-use the scores.
2. Compose with `context-stack` only after that — LOOP.md's change-one-thing
   rule, and both are unconfirmed at k=1.
3. The +0.180 at `2025-08-27/48` is the most interesting single number here
   given `near-chance-deployments`; it is worth a diagnostic look rather than
   another CV.
