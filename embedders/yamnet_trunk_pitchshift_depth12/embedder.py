"""yamnet_trunk_depth12 (layer-11 cut, fine-tunable layers 12-14), plus the
same octave-up pitch-shifted twin `yamnet_pitchshift`/`yamnet_trunk_pitchshift`
concatenate -- run through the SAME fine-tuned trunk tail via the
TimeDistributed mechanism, exactly as `yamnet_trunk_pitchshift` does for the
layer-12 cut.

IDEAS-style proposal (2026-09-23, run inline, not queued): `trunk-ft-pitchshift`
combined this era's two largest single levers (pitch-shift concat, trunk
fine-tuning) at the layer-12 cut and got a mixed hard-fold signature (1_95 up,
1_114 down) relative to pitchshift-contrast's clean 1_114 sweep in the frozen
regime. Separately, `trunk-depth-headtohead` found the layer-11 cut
(depth12) statistically equivalent to the layer-12 cut (13-14) on plain
fine-tuning alone (-0.020 +/- 0.011, inside the era's noise floor). Whether
that equivalence holds once pitch-shift concat is stacked on top is untested
-- the extra trainable block could interact differently with two co-tuned
views than with one. Comparator: `trunk-ft-pitchshift`'s
`yamnet_trunk_pitchshift` model (layer-12 cut + concat), same day/config
otherwise.

Mechanism is a straight recombination of two already-merged embedders: the
depth-12 cut/tail from `yamnet_trunk_depth12`, the octave-up dual-view +
shared-tail TimeDistributed concat from `yamnet_trunk_pitchshift`. No new
audio transform, no new architecture piece.
"""
import importlib

from embedders.recipe import Branch, Keras, Recipe, RecipeEmbedder, UP_OCTAVE

_trunk12 = importlib.import_module('embedders.yamnet_trunk_depth12.embedder')


class EmbedderYamnetTrunkPitchshiftDepth12(RecipeEmbedder, _trunk12.EmbedderYamnetTrunkDepth12):
    embeddername = "yamnet_trunk_pitchshift_depth12"
    # [plain, octave-up], each through the depth12 trunk
    recipe = Recipe(branches=(Branch(Keras()),
                              Branch(Keras(), UP_OCTAVE)),
                    dtype='float16')
    n_ctx = len(recipe.branches)  # one shared trunk tail per view, TimeDistributed in build_head
    n_embeddings = _trunk12.EmbedderYamnetTrunkDepth12.n_embeddings * n_ctx
