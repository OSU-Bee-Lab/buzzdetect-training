# Experiment Ideas

Candidate experiments for future LOOP agents. Each entry is a hypothesis ready to be taken through the standard experiment lifecycle in LOOP.md.

Before picking one up, check `experiments/log.jsonl` to confirm it hasn't already been tried.

---

## negative-audit

**Hypothesis:** Unlabeled buzz events are leaking into the negative training class, degrading the decision boundary.

**What to do:** This is a data audit, not a model change. Before running any training:
1. Load the negative-class snips for the current set (audio files tagged as non-buzz)
2. Run the best current model on them and flag any that score above ~0.5
3. Listen to a sample of the high-scoring negatives — are any of them actual buzz?

If contamination is confirmed, filter those clips from the negative class and retrain the best current config to see the isolated effect. Record the contamination rate in notes.

**Why it might help:** A "No Free Lunch" benchmark study (arxiv:2508.10230) found that in detection tasks, models systematically fail to separate target sounds from backgrounds when training negatives contain unlabeled instances of the target. This is independent of embedder choice.

**Caveats:** Requires listening time. If contamination is low (<5%), impact is probably small.

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

## differential-lr

**Hypothesis:** Training the full embedder end-to-end with a very small backbone LR and normal head LR will outperform both the frozen baseline and aggressive fine-tuning.

**What to do:** Same on-the-fly training setup as `top-layer-unfreeze`, but unfreeze the entire backbone. Use a LR schedule where backbone LR = head LR / 100. Apply early stopping on validation loss to prevent catastrophic forgetting. Compare to `top-layer-unfreeze` result.

**Why it might help:** When data is sufficient, full backbone adaptation outperforms partial. This experiment establishes whether the dataset is large enough for this approach.

**Caveats:** High risk of overfitting with small datasets. Only worth trying after `top-layer-unfreeze` shows positive signal. If `top-layer-unfreeze` fails, skip this one.
