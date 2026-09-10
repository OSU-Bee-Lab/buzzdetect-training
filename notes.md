# yamnet-aves

## Hypothesis
YAMNet's 64 mel bins over 125-7500 Hz are not tuned to bee buzz (~200 Hz
fundamental + harmonics to 1-2 kHz), and every attempt to reshape YAMNet's
frontend has collapsed the embedding (gain4 -0.028, recorder-center -0.016,
harmonic-comb null) because the frozen conv stack was trained jointly with those
exact mel params. The productive move is not editing YAMNet's frontend but
adding a second representation with different frequency coverage *alongside* it.

AVES (bioacoustic wav2vec2) is individually predictive (0.194 converged offline
readout) and encodes buzz evidence differently from YAMNet: PCA-whitening puts
YAMNet at 90% of its own best sensitivity in 16 principal directions and AVES at
28%, needing ~256 -- sparse-and-concentrated vs dense-and-distributed, from
different pretraining domains. Two individually-predictive representations that
encode differently are the textbook case where concatenation beats either alone.
Nobody has tried it; AVES was closed on its solo number, which is a statement
about AVES *replacing* YAMNet, not *adding* to it. (IDEAS.md item C.)

Frame geometry: the existing yamnet (0.96 s) and aves (1.0 s) caches are on
different grids (hop = framelength_s * framehop_prop), so a plain row-wise concat
of the two caches is not clean -- per-chunk frame counts diverge. Instead, one
embedder on a single 1.0 s grid: full 1.0 s -> AVES, centre 0.96 s -> YAMNet,
concat -> 1792-d. Rows aligned by construction.

## Changes
- `embedders/yamnet_aves/embedder.py`: new embedder, framelength_s 1.0,
  n_embeddings 1792. YAMNet on the centred 15360 samples (patch_hop 0.96, one
  patch per crop), AVES batched on the full frame (its own torch/CUDA path,
  cuDNN off, as embedders/aves). TF GPU hidden under BUZZDETECT_NO_GPU=1.
- No change to 02_set / 03_train. Set config_extract unchanged
  (overlap_event_prop 0.2, framehop_prop 1) -> overlap_event_s 0.2 s, identical
  to the aves cache and ~4% off yamnet's 0.192 s.

## Protocol
- Comparator: `cv_baseline` (0.218), as `aves-probe` was logged -- the 1.0 s vs
  0.96 s label difference was negligible there (buzz rate unchanged to 3 dp).
- Probe config is YAMNet-tuned and the AVES block is dense/signed. Run plain
  first. If negative, sweep the readout offline on the cached 1792-d embeddings
  before logging a verdict (LOOP.md constraint), and check whether the AVES
  block is inert in the fold weights -> if so, rerun with per-block
  standardization paired with a fixed epoch budget.

## Results
| fold | baseline sens@fpr0.005 | this exp | delta | val frames |
|---|---|---|---|---|
_pending_

## Conclusion
_pending_
