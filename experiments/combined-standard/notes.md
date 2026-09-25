# combined-standard

## Hypothesis
The yamnet_combined embedder produces a 1545-dim feature vector (vs standard 1024-dim YAMNet) by combining multiple internal YAMNet representations. More dimensions could provide additional discriminative signal for buzz vs. non-buzz, particularly if the extra features capture spectral or temporal structure that the top-level embedding misses.

Prior evidence: the lite_combined model was trained but never evaluated, so this is effectively a first real test of the approach. No contradictory evidence exists.

## Changes
- Fixed `embeddername = "yamnet"` → `"yamnet_combined"` in `embedders/yamnet_combined/embedder.py` to prevent path collision with standard yamnet embeddings.

## Reproduction
- No external artifacts; yamnet_combined.keras is bundled with the embedder.
- Stage 2: `conda run -n buzzdetect-train python 02_set/main.py --set standard --embedder yamnet_combined`
- Stage 3: `conda run -n buzzdetect-train python 03_train/main.py --model exp_combined_std_v1 --set standard --embedder yamnet_combined --translation general --epochs 300`

## Results
(to be filled in after eval)

## Conclusion
(to be filled in after eval)
