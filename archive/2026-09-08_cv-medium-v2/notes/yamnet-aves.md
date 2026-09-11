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
28%, needing ~256 -- sparse-and-concentrated vs dense-and-distributed. Two
individually-predictive representations that encode differently are the textbook
case where concatenation beats either alone. Nobody had tried it; AVES was
closed on its solo number, which is a statement about AVES *replacing* YAMNet,
not *adding* to it. (IDEAS.md item C.)

## Changes
- `embedders/yamnet_aves/embedder.py`: new embedder, framelength_s 1.0,
  n_embeddings 1792 (1024 YAMNet + 768 AVES). Each frame feeds the full 1.0 s to
  AVES (batched, its own torch/CUDA path, cuDNN off as embedders/aves) and the
  centred 0.96 s to YAMNet. One shared frame grid, so rows align by
  construction -- the existing yamnet (0.96 s) and aves (1.0 s) caches are on
  different grids and cannot simply be concatenated. Raw concat only: no
  normalization, no projection, so the 1792-d cache is reusable for any readout.
  TF GPU hidden under BUZZDETECT_NO_GPU=1; default runs the YAMNet block on GPU
  via memory-growth, coexisting with AVES on the 4 GB card.
- No change to 02_set / 03_train. Set config unchanged (overlap_event_prop 0.2,
  framehop_prop 1) -> overlap_event_s 0.2 s, identical to the aves cache.

## Results
Comparator cv_baseline (0.218), as aves-probe was logged. Two independent draws
(no seed control): yamnet_aves on CPU, yamnet_aves_2 as the confirmation repeat.

| fold           | baseline | run 1 | run 2 | both up? |
|----------------|----------|-------|-------|----------|
| 1_29 (rich)    | 0.426    | 0.445 | 0.479 | yes      |
| 53 Marysville  | 0.425    | 0.452 | 0.433 | yes      |
| willard/1_11   | 0.180    | 0.230 | 0.193 | yes      |
| 1_95 (jet FP)  | 0.037    | 0.051 | 0.061 | yes      |
| 1_150 (low-SNR)| 0.021    | 0.021 | 0.007 | no (flat/down) |
| **mean**       | **0.218**| 0.240 | 0.235 |          |

- Shipped-epoch mean: +0.022 (run 1, 4up/0down/1flat), +0.017 (run 2, 4up/1down).
- honest_epoch (no leakage, both arms same rule), xfold-pooled:
  cv_baseline 0.219 -> run 1 0.243 (+0.024) -> run 2 0.246 (+0.027). The gain is
  NOT a stopping artifact -- stable at ~+0.025 under own-peak / shipped /
  xfold-median / xfold-pooled, unlike recorder-center.
- Reproducible direction: the same 4 folds up in both draws; 1_150 unmoved in
  both (as context-embedder found -- 1_150 needs the epoch rule, not a better
  representation). 1_95 up in both draws -- the FIRST intervention to move it
  (harmonic-comb, mech-margin, shared-trunk-head, context-embedder all flat or
  down there). Still near-chance.
- Run-to-run spread on the identical config is up to 0.037/fold (willard
  0.230->0.193, 1_29 0.445->0.479) -- the seedless training noise, consistent
  with the docs. The mean delta (~+0.020 shipped, ~+0.025 honest) sits right at
  probe-grid's ~0.027 single-run MDE, so magnitude is at the resolution limit;
  the two-draw agreement and the 4-up pattern are the signal, not the mean.
- For scale: context_embedder is +0.058 xfold-pooled. This is roughly half that,
  clean, on a plain concat with the YAMNet-tuned probe.

## Not done (leads)
- Per-block standardization. The AVES block is ~3.8x the YAMNet block in
  per-dim |mean| (0.306 vs 0.080) and 0% zeros vs 91.6%. Under one Adam LR the
  smaller block is likely underweighted. `--standardize` exists (off by
  default); IDEAS C flagged this. Pair with a fixed epoch budget (every prior
  normalisation moved best_epoch).
- Compose with context: yamnet_context (+0.058) and yamnet_aves attack
  different weaknesses (temporal contrast vs frequency coverage); a
  context+aves embedder is untried.

## Infra findings (not part of the verdict)
- GPU extraction: TF GPU + AVES torch/CUDA coexist on the 4 GB card via TF
  memory-growth. No OOM. Marginal speedup only -- AVES was already on CUDA, only
  the small YAMNet conv net moved off CPU. Full medium extraction ~17 min.
- GPU *training* works with `TF_GPU_ALLOCATOR=cuda_malloc_async` (the default
  BFC allocator hits a fatal `bfc_allocator.cc:802 Check failed` on this card,
  even for the tiny probe). ~1.6x faster per fold wall-clock (the win is the
  per-epoch val_sens callback, a batched forward pass; the training step itself
  is ~2 steps and dominated by launch/tf.data overhead). CV ~9 min CPU / ~5-6
  min GPU.
- jit_compile: NO measurable step speedup (111 vs 101 ms/step) -- nothing for
  XLA to fuse in a 2-step epoch, and the val_sens callback is a manual forward
  pass outside compile(). Not adopted. Kept CPU for the logged runs (matches
  every other era number).

## Conclusion
Positive, clean, reproducible, modest. ~+0.025 honest mean over two draws,
4/5 folds up both times, at the single-run MDE on magnitude. First movement ever
seen at 1_95. Worth building on -- per-block standardization is the obvious next
step given the 3.8x block-scale gap, and composing with yamnet_context is the
other. Not a headline win on its own (half of context_embedder), but the
"add a differently-encoding representation alongside YAMNet" thesis held.
