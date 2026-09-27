"""yamnet_trunk (layer-12 maps, fine-tunable layers 13-14), plus the same
octave-up pitch-shifted twin `yamnet_pitchshift` concatenates -- but run
through the SAME fine-tuned trunk tail via the shared-weight TimeDistributed
mechanism `yamnet_trunk_context` already built for temporal neighbours.

IDEAS proposal, new (2026-09-21): this era's two largest confirmed single
levers are `yamnet-pitchshift` (concat with an octave-up twin, +0.069, frozen
probe) and `trunk-ft-v3` (unfreezing layers 13-14, +0.077, plain audio). They
have never been combined. `trunk-ft-v3`'s own context variant
(`yamnet_trunk_context`) showed *temporal* context does not stack with
fine-tuning (+0.002 over trunk3-ft-1e5) -- but pitch-shift is a different
axis (same timestamp, transformed audio), already shown independently
additive with AVES depth (`pitchshift-aves-mid` beat both parents). This is
the first test of whether it is additive with backbone fine-tuning too.

Each frame's trunk (layer12) features are computed TWICE per frame -- once
on the plain 0.96 s audio, once on its octave-up resample+tile twin (same
mechanism as `yamnet_pitchshift`, ported unchanged) -- and cached as one
2 x 12288 float16 row. `build_head()`'s existing `n_ctx` path (a single
shared copy of layers 13-14, `TimeDistributed`, inherited unchanged from
`yamnet_trunk`) then fine-tunes ONE trunk tail jointly across both views
before concatenating their 1024-d codes into the Dense head. No temporal
neighbour is read; only the pitch-shift mechanism from `yamnet_pitchshift`
and the shared-tail mechanism from `yamnet_trunk_context`, combined.
"""
import importlib

from embedders.recipe import Branch, Keras, Recipe, RecipeEmbedder, UP_OCTAVE

_trunk = importlib.import_module('embedders.yamnet_trunk.embedder')


class EmbedderYamnetTrunkPitchshift(RecipeEmbedder, _trunk.EmbedderYamnetTrunk):
    embeddername = "yamnet_trunk_pitchshift"
    # [plain, octave-up], each through the layer-12 trunk
    recipe = Recipe(branches=(Branch(Keras()),
                              Branch(Keras(), UP_OCTAVE)),
                    dtype='float16')
    n_ctx = 2  # shared trunk tail, TimeDistributed in build_head
    n_embeddings = _trunk.EmbedderYamnetTrunk.n_embeddings * 2  # 24576
