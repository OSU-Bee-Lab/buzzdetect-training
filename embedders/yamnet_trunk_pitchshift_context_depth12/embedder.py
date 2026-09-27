"""yamnet_trunk_pitchshift_depth12 plus one frame of temporal context either
side: layer-11 maps of the plain frame and its octave-up twin, for t-1, t, t+1,
all through ONE shared fine-tunable tail (layers 12-14).

Era cv-medium-v4 grid (2026-09-25): every combination of {context, pitch-shift,
dense head} on the retrained trunk; this is the context + pitch-shift cell.

Pure recombination (MRO: context, then pitchshift_depth12): its recipe runs
[plain, octave-up] per frame and widens both with their neighbours, so a row is
[t-1 plain, t-1 up, t plain, t up, t+1 plain, t+1 up] = 6 x 12288 float16.
build_head()'s n_ctx path reshapes to those 6 views in the same order.
"""
import importlib

from embedders.recipe import Branch, Context, Keras, Recipe, UP_OCTAVE

_ctx = importlib.import_module('embedders.yamnet_trunk_context.embedder')
_psd12 = importlib.import_module('embedders.yamnet_trunk_pitchshift_depth12.embedder')


class EmbedderYamnetTrunkPitchshiftContextDepth12(_ctx.EmbedderYamnetTrunkContext,
                                                  _psd12.EmbedderYamnetTrunkPitchshiftDepth12):
    embeddername = "yamnet_trunk_pitchshift_context_depth12"
    context_frames = _ctx.CONTEXT_FRAMES
    recipe = Recipe(branches=(Branch(Keras()),
                              Branch(Keras(), UP_OCTAVE)),
                    context=Context(k=_ctx.CONTEXT_FRAMES), chunk=True, dtype='float16')
    n_ctx = 2 * (2 * _ctx.CONTEXT_FRAMES + 1)  # 2 views x 3 frames
    n_embeddings = _psd12.EmbedderYamnetTrunkPitchshiftDepth12.n_embeddings * (2 * _ctx.CONTEXT_FRAMES + 1)
