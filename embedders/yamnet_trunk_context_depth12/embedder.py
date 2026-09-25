"""yamnet_trunk_context at the depth-12 cut: layer-11 maps, fine-tunable
layers 12-14, with one frame of temporal context either side.

Era cv-medium-v4 grid (2026-09-25): every combination of {context, pitch-shift,
dense head} on the retrained trunk. `trunk-ft-v3` found context did not stack
with plain fine-tuning at the layer-12 cut (+0.002); this reruns it at the
cut the previous era's best (`trunk-pitchshift-depth12`) used, on the Hard
Negatives data.

Pure recombination: yamnet_trunk_context's framing/stacking methods over
yamnet_trunk_depth12's cut and build_head (MRO: context first, then depth12).
Each frame is concat(t-1, t, t+1) = 3 x 12288 float16.
"""
import importlib

_ctx = importlib.import_module('embedders.yamnet_trunk_context.embedder')
_trunk12 = importlib.import_module('embedders.yamnet_trunk_depth12.embedder')


class EmbedderYamnetTrunkContextDepth12(_ctx.EmbedderYamnetTrunkContext,
                                       _trunk12.EmbedderYamnetTrunkDepth12):
    embeddername = "yamnet_trunk_context_depth12"
    context_frames = _ctx.CONTEXT_FRAMES
    n_ctx = 2 * _ctx.CONTEXT_FRAMES + 1
    n_embeddings = _trunk12.EmbedderYamnetTrunkDepth12.n_embeddings * n_ctx
