"""[plain, phase-vocoder octave-up, phase-vocoder octave-down]: the vocoder
version of `v4-ft-psud`.

Part of the 2026-09-26 pitch-shift method x direction grid (HANDOFF.md). Views
live in embedders/recipe.py. Comparators:
`v4-ft` (no shift, 0.375), `v4-ft-ps` (resample up, 0.452), `v4-ft-psud`
(resample up + centre down, 0.452), `v4-ft-psd` (resample centre down).

embed() and to_onnx() both come from the recipe (embedders/recipe.py). The
vocoder has no ONNX form, so to_onnx() raises: CV only.
"""
import importlib

from embedders.recipe import Branch, Keras, Recipe, RecipeEmbedder, VOCODER_DOWN, VOCODER_UP

_trunk12 = importlib.import_module('embedders.yamnet_trunk_depth12.embedder')


class EmbedderYamnetTrunkVocoderUpdownDepth12(RecipeEmbedder, _trunk12.EmbedderYamnetTrunkDepth12):
    embeddername = "yamnet_trunk_vocoder_updown_depth12"
    # [plain, vocoder up, vocoder down], each through the depth12 trunk
    recipe = Recipe(branches=(Branch(Keras()),
                              Branch(Keras(), VOCODER_UP),
                              Branch(Keras(), VOCODER_DOWN)),
                    dtype='float16')
    n_ctx = len(recipe.branches)  # one shared trunk tail per view, TimeDistributed in build_head
    n_embeddings = _trunk12.EmbedderYamnetTrunkDepth12.n_embeddings * n_ctx
