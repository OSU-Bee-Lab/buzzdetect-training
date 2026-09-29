"""Isolation cell for `ps-fast` (2026-09-28): the up view read off the plain
STFT, as in yamnet_trunk_pitchshift_fast_depth12, but TILED to 96 frames as
v4-ft-ps's resampled view was. Both views give a (6, 4, 512) map through one
shared depth12 tail (TimeDistributed, n_ctx=2), exactly v4-ft-ps's head.

`ps-fast` changed two things at once and lost 0.174 on 1_114. The 2x2:

                     tiled                untiled
  resampled audio    v4-ft-ps (0.452)     yamnet_trunk_pitchshift_untiled_depth12
  shared STFT        this embedder        v4-ft-ps-fast (0.432)
"""
import importlib

import numpy as np

from embedders.recipe import Branch, Keras, Recipe, RecipeEmbedder

_trunk12 = importlib.import_module('embedders.yamnet_trunk_depth12.embedder')
_fast = importlib.import_module('embedders.yamnet_trunk_pitchshift_fast_depth12.embedder')


class EmbedderYamnetTrunkPitchshiftSharedtiledDepth12(RecipeEmbedder, _trunk12.EmbedderYamnetTrunkDepth12):
    embeddername = "yamnet_trunk_pitchshift_sharedtiled_depth12"
    recipe = Recipe(branches=(Branch(Keras()),), dtype='float16')
    n_ctx = 2  # [plain, up], one shared trunk tail per view, TimeDistributed in build_head
    n_embeddings = _trunk12.EmbedderYamnetTrunkDepth12.n_embeddings * n_ctx

    def initialize(self):
        import keras
        import tensorflow as tf

        full = self._load_full()
        names = [l.name for l in full.layers]
        cnn = full.layers[names.index(_fast._FIRST_LAYER):names.index(_trunk12._TRUNK_LAYER) + 1]
        wave = keras.Input(shape=(), dtype=tf.float32, name='waveform')
        plain, up = _fast._dual_features_layer(full.layers[1].params, tile=True)(wave)
        for layer in cnn:
            plain, up = layer(plain), layer(up)
        out = keras.layers.Concatenate()([keras.layers.Flatten()(plain),
                                          keras.layers.Flatten()(up)])
        self.model = keras.Model(wave, out, name='yamnet_trunk_l11_pssharedtiled')
        self.embed(np.zeros(int(self.framelength_s * self.samplerate), dtype=np.float32))
