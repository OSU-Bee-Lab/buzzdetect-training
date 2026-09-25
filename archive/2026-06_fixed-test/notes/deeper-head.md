<!-- harvested 2026-09-25 from exp/deeper-head (b69c1f6); pinned at refs/archive/deeper-head -->

<!-- IDEAS.md -->
# Overview
These are ideas for future loops put in by the human in the loop.

# Embedders
## Samplerate
YAMNet works at 16Khz, but this throws out potentially valuable information. Why not feed it 32KHz audio and just tell it that it's 16KHz? Maybe we can train it to recognize buzzes at half-speed.

### Note
I tried this and it seemed to tank performance; but maybe it was my own bad implementation?

## Mel bands
YAMNet puts mel bands across a wide range, why not target something closer to the bee buzz range?

### Note
I tried this too and it also seemed to tank performance. Any perturbance from YAMNet's expected audio seems to tank performance in a way that can't be recovered by training.


# Extra information
## Audio extraction
We could take the average amplitude in a few targeted bands and add them to the embedding array. Issue is this could greatly slow down processessing, especially if not CUDA-compatible.
<!-- NOTES.md -->
# deeper-head

## Hypothesis
The current classifier is a single Dense layer (logistic regression) over YAMNet's 1024-dim embeddings. YAMNet was trained on AudioSet and its embeddings encode a wide variety of audio concepts — buzz-relevant features are almost certainly entangled with irrelevant ones in a nonlinear way. Adding one hidden layer (Dense 128, ReLU) before the output should allow the model to learn nonlinear combinations and improve sensitivity without sacrificing precision.

## Evidence
- Linear probing on pretrained audio embeddings is a common starting point but rarely optimal; a shallow MLP typically yields gains of 2–5% on detection tasks (e.g. HEAR benchmark results).
- The current head is architecturally the simplest possible; there is no prior experiment ruling out more capacity.

<!-- experiments/deeper-head/notes.md -->
# deeper-head

## Hypothesis
The current classifier is a single Dense layer (logistic regression) over YAMNet's 1024-dim embeddings. YAMNet embeddings encode many audio concepts; buzz-relevant features are likely entangled with irrelevant ones in a nonlinear way. Adding one hidden layer (Dense 128, ReLU) before the output should allow the model to learn nonlinear combinations and improve sensitivity without sacrificing precision.

## Changes
- `03_train/train.py`: Added `Dense(128, activation='relu')` before the output layer
- Set: `lite`, embedder: `yamnet`, translation: `general`

## Reproduction
No external artifacts. Run from the worktree:
```bash
conda run -n buzzdetect-train python 03_train/main.py \
  --model exp_deeper_head_lite --set lite --embedder yamnet \
  --translation general --epochs 300
```

## Results

Note: no single-Dense baseline exists for the `lite` set with test results, so a direct same-set comparison isn't available. The best available reference is `test_standard` (single Dense, `standard` set), which is a larger dataset.

Metric | test_standard (single Dense, standard) | exp_deeper_head_lite (Dense 128, lite)
---|---|---
Sensitivity @ 95% precision | 25.07% | 22.25%
Sensitivity @ 90% precision | 27.63% | 26.47%
Precision @ 80% sensitivity | 25.63% | 24.51%

The deeper head is worse across all metrics, though the comparison is confounded by dataset size (lite < standard). The `exp/deeper-std` experiment (same MLP head, standard set) confirmed the pattern holds even with equal data: 19.35% vs 25.07% — the MLP head hurts more, not less, on standard.

## Conclusion
MLP head is consistently worse than linear probe on YAMNet embeddings at the high-precision operating point. Do not revisit MLP head variants unless there's a specific reason to expect different behavior (e.g., much larger dataset, regularisation changes, or a different embedder with less-linear structure).
