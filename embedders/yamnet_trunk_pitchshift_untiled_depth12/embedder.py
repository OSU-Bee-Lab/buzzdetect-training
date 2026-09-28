"""Isolation cell for `ps-fast` (2026-09-28): v4-ft-ps's up view (librosa
resample 2:1, relabelled 16 kHz) but UNTILED: the 0.48 s of shifted audio gets
its own STFT and goes through the CNN as a 48-frame patch, (3, 4, 512) map,
with v4-ft-ps-fast's head (one shared depth12 tail on the (6,4,512) plain map
and the (3,4,512) up map).

                     tiled                untiled
  resampled audio    v4-ft-ps (0.452)     this embedder
  shared STFT        ..._sharedtiled      v4-ft-ps-fast (0.432)
"""
import importlib

import numpy as np

from embedders.recipe import Branch, Keras, Recipe, Resample

_trunk12 = importlib.import_module('embedders.yamnet_trunk_depth12.embedder')
_fast = importlib.import_module('embedders.yamnet_trunk_pitchshift_fast_depth12.embedder')


class EmbedderYamnetTrunkPitchshiftUntiledDepth12(_fast.EmbedderYamnetTrunkPitchshiftFastDepth12):
    embeddername = "yamnet_trunk_pitchshift_untiled_depth12"
    # [plain, octave-up untiled]; the up branch's model frames 0.48 s patches
    recipe = Recipe(branches=(Branch(Keras()),
                              Branch(Keras('model_half'), (Resample('half'),))),
                    dtype='float16')

    def initialize(self):
        import copy

        import keras
        import tensorflow as tf

        full = self._load_full()
        # plain: exactly yamnet_trunk_depth12's model
        self.model = keras.Model(inputs=full.input,
                                 outputs=full.get_layer(_trunk12._TRUNK_LAYER).output,
                                 name='yamnet_trunk_l11')
        names = [l.name for l in full.layers]
        cnn = full.layers[names.index(_fast._FIRST_LAYER):names.index(_trunk12._TRUNK_LAYER) + 1]
        params = copy.copy(full.layers[1].params)
        params.patch_window_seconds = params.patch_hop_seconds = self.framelength_s / 2
        features = type(full.layers[1])(params, name='waveform_features_half')
        wave = keras.Input(shape=(), dtype=tf.float32, name='waveform')
        _, x = features(wave)
        x = keras.layers.Reshape((48, 64, 1))(x)
        for layer in cnn:
            x = layer(x)
        self.model_half = keras.Model(wave, x, name='yamnet_trunk_l11_half')
        self.embed(np.zeros(int(self.framelength_s * self.samplerate), dtype=np.float32))
