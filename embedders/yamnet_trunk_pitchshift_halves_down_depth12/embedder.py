"""[plain, octave-down of the frame's first half, octave-down of its second half]:
resample-down that sees the whole frame. `v4-ft-psd`'s down view sees only the
centre half; this adds the outer quarters back as a second down view.

Part of the 2026-09-26 pitch-shift method x direction grid (HANDOFF.md). Views
live in embedders/recipe.py. Comparators:
`v4-ft` (no shift, 0.375), `v4-ft-ps` (resample up, 0.452), `v4-ft-psud`
(resample up + centre down, 0.452), `v4-ft-psd` (resample centre down).

embed() and to_onnx() both come from the recipe (embedders/recipe.py).
"""
import importlib

from embedders.recipe import Branch, Keras, Recipe, RecipeEmbedder, DOWN_OCTAVE_FIRST, DOWN_OCTAVE_SECOND

_trunk12 = importlib.import_module('embedders.yamnet_trunk_depth12.embedder')


class EmbedderYamnetTrunkPitchshiftHalvesDownDepth12(RecipeEmbedder, _trunk12.EmbedderYamnetTrunkDepth12):
    embeddername = "yamnet_trunk_pitchshift_halves_down_depth12"
    # [plain, octave-down first half, second half], each through the depth12 trunk
    recipe = Recipe(branches=(Branch(Keras()),
                              Branch(Keras(), DOWN_OCTAVE_FIRST),
                              Branch(Keras(), DOWN_OCTAVE_SECOND)),
                    dtype='float16')
    n_ctx = len(recipe.branches)  # one shared trunk tail per view, TimeDistributed in build_head
    n_embeddings = _trunk12.EmbedderYamnetTrunkDepth12.n_embeddings * n_ctx
