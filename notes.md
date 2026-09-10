# yamnet-aves-head

## Hypothesis
`yamnet_aves` (YAMNet 1024-d ⊕ AVES 768-d, 1792-d concat) paid a clean but
modest +0.025 (xfold-pooled) with the stock linear probe. The probe config
(Dropout 0.2, no weight decay, one Dense straight off the embedding) was tuned
for YAMNet's sparse, non-negative, already-linearly-separable code. The AVES
half of the concat is dense, signed, and smears buzz evidence over ~256
directions (aves-readout PCA). A linear readout is the worst case for that
half. A shared ReLU hidden layer lets all 15 classes read one learned
representation and gives the reader room to recombine the AVES dimensions.

Offline converged sweeps (aves-mlp-head, 2026-09-10) found a non-linear head
helps *both* embedders offline (YAMNet 0.248→0.272, AVES 0.182→0.231) but does
not close representation gaps. The concat + non-linear head is the untested
cell. Question: does `--hidden` on `yamnet_aves` beat its own linear control
in-pipeline, and by more than the ~0.027 single-run MDE?

Known trap (aves-mlp-head): more head params → earlier val_loss argmin →
early-stopping ships an undertrained head. Read `best_epoch` on every fold; if
the hidden arms stop much earlier than h=0, re-score both arms at a common
cross-fold epoch (`tools/honest_epoch.py`) before believing the delta.

## Changes
- Cherry-picked `--hidden` from `exp/shared-trunk-head@4a90546`:
  `Input → Dropout(0.2) → [Dense(h,relu) → Dropout(0.2)] → Dense(n_classes)`.
  h=0 (default) is the byte-identical shipped head.
- Nothing else changes: loss, stopping rule (val_loss, patience 50,
  RestoreTrueBest), optimizer, label smoothing all unchanged.
- Embedder `yamnet_aves` (cached, no re-extraction). Translation `general`.

## Runs
- `yav_h0`    — matched linear control on yamnet_aves
- `yav_h256`  — hidden width 256
- `yav_h1024` — hidden width 1024

## Results
| fold | yav_h0 | h256 | h1024 | best_epoch h0→h256→h1024 |
|---|---|---|---|---|

## Conclusion
