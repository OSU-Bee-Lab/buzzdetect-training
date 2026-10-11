"""yamnet_trunk_depth10 plus the octave-up twin (ps-depth10); see yamnet_trunk_pitchshift_depth12."""
import importlib

from embedders.recipe import Branch, Keras, Recipe, RecipeEmbedder, UP_OCTAVE

_trunk12 = importlib.import_module('embedders.yamnet_trunk_depth10.embedder')


class EmbedderYamnetTrunkPitchshiftDepth10(RecipeEmbedder, _trunk12.EmbedderYamnetTrunkDepth10):
    embeddername = "yamnet_trunk_pitchshift_depth10"
    # [plain, octave-up], each through the depth10 trunk
    recipe = Recipe(branches=(Branch(Keras()),
                              Branch(Keras(), UP_OCTAVE)),
                    dtype='float16')
    n_ctx = len(recipe.branches)  # one shared trunk tail per view, TimeDistributed in build_head
    n_embeddings = _trunk12.EmbedderYamnetTrunkDepth10.n_embeddings * n_ctx
