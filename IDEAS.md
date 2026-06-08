# Experiment Ideas

Candidate experiments for future LOOP agents. Each entry is a hypothesis ready to be taken through the standard experiment lifecycle in LOOP.md.

Before picking one up, check `experiments/log.jsonl` to confirm it hasn't already been tried.

---

## top-layer-unfreeze

**Hypothesis:** Fine-tuning the last few layers of the embedder backbone on this task will yield more discriminative embeddings than the frozen baseline, without the overfitting risk of full fine-tuning.

**What to do:** This requires an end-to-end training script that bypasses the pre-extracted pickle embeddings. Suggested approach:
1. Write a new training script (e.g. `03_train/train_finetune.py`) that loads raw audio snips, runs them through the embedder with gradients enabled for the top N layers only (bottom layers frozen), and jointly optimizes embedder tail + classifier head.
2. Start with N=1 (only the final projection/pooling layer), then try N=2–3.
3. Use a backbone LR ~50–100x smaller than the head LR (e.g. 1e-5 backbone, 5e-4 head).
4. After training, save the fine-tuned backbone as a new embedder (e.g. `embedders/yamnet_ft/`) and re-extract embeddings with it for a clean eval.

**Why it might help:** The pretrained embedder was not trained on buzz sounds specifically. Nudging the upper layers toward the buzz/non-buzz boundary should improve sensitivity without catastrophic forgetting of useful low-level features.

**Caveats:** Requires significant changes to the training pipeline (on-the-fly audio loading instead of pickle embeddings). Memory and compute cost will be higher. Start with YAMNet (lightest backbone) or Nighthawk (ResNet34, well-understood).

---

## lora-adapter

**Hypothesis:** Adding lightweight LoRA adapters inside the frozen embedder backbone and training only those adapters + the classifier head will improve embedding quality with minimal overfitting risk.

**What to do:**
1. Install `peft` (HuggingFace parameter-efficient fine-tuning library).
2. Wrap the embedder backbone with LoRA adapters on its linear/attention layers (rank 4–16).
3. Train adapters + head jointly on raw audio (same on-the-fly loading approach as `top-layer-unfreeze`).
4. Save the adapted backbone as a new embedder and re-extract for eval.

**Why it might help:** LoRA adds very few trainable parameters (~0.1–1% of backbone params), making overfitting on small datasets much less likely than full fine-tuning. It's the standard approach for adapting large pretrained models to narrow tasks with limited data.

**Caveats:** LoRA was developed for transformer architectures; applying it to CNN-based embedders (YAMNet, Nighthawk's ResNet) is less standard and requires manually inserting adapters into conv layers. Best first target is a transformer-based embedder (AVES or AST, once implemented).

---

## aves-intermediate-layer

**Hypothesis:** AVES embeddings from the last transformer layer are too bird-classification-specific to discriminate insect buzz. An intermediate layer (e.g. layer 6–9 of 12) carries more general acoustic features that may separate buzz from non-buzz more reliably.

**Background:** We trained `aves_lite` and found it performs at random-guess level on test data (precision ≈ base rate of 18%, decreasing with threshold — i.e. inverted). Investigation showed the pipeline is correct (same 242 buzz frames as YAMNet, same audio source). The problem is embedding discrimination: AVES embeddings have roughly half the mean absolute dimension-wise difference between buzz and non-buzz (0.13 vs 0.23 for YAMNet). YAMNet's non-buzz embeddings are sparse and non-negative (ReLU-activated), making the classes easy to linearly separate. AVES embeddings are symmetric around zero for both classes, so the buzz signal is weak.

The likely cause: AVES is a self-supervised wav2vec2 model fine-tuned on bird vocalizations. Its last transformer layer encodes bird-specific representations. Wav2vec2 transfer learning literature consistently shows that middle layers (~6–9 of 12) carry more general phonetic/acoustic features better suited to novel downstream tasks.

**What to do:**
1. In `embedders/aves/embedder.py`, change `layer_outputs[-1]` to `layer_outputs[N]` for N in {5, 7, 9, 11} (0-indexed, so layer 6, 8, 10, 12 of the 12-layer base model).
2. Re-extract aves embeddings for the lite set with each layer choice (or just try one — layer 8 is a reasonable first pick based on the literature).
3. Train a new model (e.g. `aves_lite_L8`) with the new embeddings.
4. Compare test metrics to `aves_lite` (last layer). Any precision > 18% at moderate recall would confirm the hypothesis.

**Implementation note:** `extract_features()` returns a list of 12 tensors, each shape `(1, T', 768)`. The index is 0-based, so `layer_outputs[11]` is the last layer (what we currently use) and `layer_outputs[7]` is layer 8. No other code changes needed — `n_embeddings=768` stays the same across all layers.

**Why it might help:** The last layer of wav2vec2 is the most task-specialized. For bird sounds, it likely encodes species/call-type features. Insect buzz has very different spectrotemporal structure, and the bird-specific features may be anti-correlated with buzz in out-of-domain (test) recordings — which is exactly what we observed (inverted predictions on test, model learned spurious training-set-specific pattern).

**Caveats:** Requires re-extracting embeddings for each layer tried (slow). Start with one layer (8 or 9) before sweeping. If no improvement across layers, the issue may be that AVES is simply not the right embedder for insect sounds regardless of layer choice — in which case consider a general-purpose audio embedder (PANNs CNN14, LAION-CLAP, AST).

---

## differential-lr

**Hypothesis:** Training the full embedder end-to-end with a very small backbone LR and normal head LR will outperform both the frozen baseline and aggressive fine-tuning.

**What to do:** Same on-the-fly training setup as `top-layer-unfreeze`, but unfreeze the entire backbone. Use a LR schedule where backbone LR = head LR / 100. Apply early stopping on validation loss to prevent catastrophic forgetting. Compare to `top-layer-unfreeze` result.

**Why it might help:** When data is sufficient, full backbone adaptation outperforms partial. This experiment establishes whether the dataset is large enough for this approach.

**Caveats:** High risk of overfitting with small datasets. Only worth trying after `top-layer-unfreeze` shows positive signal. If `top-layer-unfreeze` fails, skip this one.


---

## White noise
Add a little white noise samples to the training set; not augmenting with overlapping noise, just some  samples that are purely random samples. Easy to produce and could help refine the learned dimensions.


---

## supplementary-frequency-features

**Hypothesis:** Concatenating hand-crafted frequency features to the YAMNet embedding gives the linear probe a direct channel for buzz-specific acoustic properties that YAMNet's general embeddings don't emphasize.

**What to do:**
1. In `02_set/extract.py` (or a new extraction path), compute per-snip features from the raw audio alongside the YAMNet embedding: dominant frequency (peak of power spectrum), spectral centroid, zero-crossing rate, and autocorrelation energy in the 100–600 Hz band.
2. Concatenate these (e.g. 4–8 scalars) to the 1024-d YAMNet embedding to form a 1028–1032-d input vector.
3. Re-extract embeddings for the medium set under a new embedder name (e.g. `yamnet_freq`) and train as usual.

**Why it might help:** Insect buzz has characteristic wing-beat frequencies (bees: ~200–400 Hz) that produce periodic spectral signatures. YAMNet is trained on AudioSet with 521 classes; buzz is a minor category and the embedding doesn't need to highlight these frequency features to minimize AudioSet loss. A few targeted scalars give the classifier a direct handle that pure YAMNet probing lacks. This is the lowest-cost change that actually adds new information.

**Caveats:** Features must be computed at the same frame resolution as YAMNet embeddings (0.96s windows). Normalization of the new dimensions matters — scale them to ~unit variance before concatenation so they don't get drowned out by the 1024-d YAMNet component.

---

## mlp-head-repro

**Hypothesis:** The MLP head (Dense 128 → Dense N_classes) underperformed in deeper-std, but that experiment predated stable training methodology (null commit, no multi-run averaging, wrong set). With current defaults (low-delta, 5 runs, label smoothing 0.2), MLP may perform comparably or better than the linear probe.

**What to do:**
1. Add `Dense(128, relu)` between Dropout and output in `train.py`, matching deeper-std architecture.
2. Run the standard 5-run pipeline on the medium set.
3. Compare to the low-delta baseline (0.224, CI [0.207, 0.242]).

**Why it might help:** The buzz manifold in YAMNet space may be non-convex or entangled with mechanical hums in ways a linear probe can't untangle. A single hidden layer is the minimum nonlinearity to test this. With proper methodology we haven't actually confirmed the linear ceiling.

**Caveats:** More parameters = more overfitting risk with 236 buzz training examples. Watch whether validation loss diverges from training loss. If so, try smaller hidden size (64) or increase dropout.

---