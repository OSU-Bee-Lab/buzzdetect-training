# context-width

## Hypothesis

`exp/context-embedder` (+0.022, clean) stacked each frame's YAMNet embedding
with its real audio neighbours `[t-1, t, t+1]` (3072-d) and shipped. Its notes
flagged "sweep k=2,3" as the untested next step, and never claimed k=1 is the
right width — only that some context beats none.

This experiment runs k=2 only: `[t-2, t-1, t, t+1, t+2]`, 5 frames, 5120-d.
Same mechanism, same code path (`context_frames` in `02_set/extract.py`,
`stack_context` in the embedder), only the width changes.

A concurrent diagnostic (`exp/deployment-forensics`) produced a mechanistic
prediction to test explicitly: `Lily Adam - One Hive/recorders/willard/
2024-08-07/1_11` regressed under k=1 context (-0.074 in context-stack, -0.059
in context-embedder) because it has the highest fraction of short (<1s: 68%)
and isolated (>5s gap: 68%) buzz events of any fold — stacking dilutes a brief
isolated buzz with silent neighbours.

**Prediction: widening to k=2 should hurt willard MORE than k=1 did (more
silent frames folded in around a <1s event), and should help folds with long
sustained buzz more (more context to average over, same signal throughout).**
If that holds, context width should be adaptive/event-length-dependent rather
than a single fixed k — the natural next experiment. If it doesn't hold
(willard flat or improved, no clean split by buzz duration), the mechanistic
story from deployment-forensics needs revision.

Prior: dose-response could go either way past k=1 — more context could keep
helping (more duration evidence) or the willard-style dilution could start
costing more than the sustained-buzz folds gain, making k=1 a local optimum.

## Changes

`embedders/yamnet_context_k2/embedder.py` (new): copy of `exp/context-embedder`'s
`embedders/yamnet_context/embedder.py`, `CONTEXT_FRAMES = 1 -> 2`, class/embeddername
renamed to `yamnet_context_k2` / `EmbedderYamnetContextK2`. No other lines changed.
`n_embeddings` derives automatically (`1024 * (2*2+1) = 5120`).

`02_set/extract.py` unmodified from main — its `context_frames` handling
(added for context-embedder) is generic over k and needed no changes.

New embeddings dir: `02_set/sets/medium/embeddings/yamnet_context_k2/` (fresh
name; did not touch `yamnet_context`'s existing cache under `medium/embeddings/`).
Note: `embedders/yamnet_context/embedder.py` itself isn't present in main's
working tree (only committed on `exp/context-embedder`), so it was pulled via
`git show exp/context-embedder:embedders/yamnet_context/embedder.py` as the
base to copy from, per the task instructions.

## Results

(filled in after training)

## Conclusion

(filled in after training)
