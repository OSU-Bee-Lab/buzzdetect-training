import importlib

import numpy as np

# Imported through the module (not `from ... import`) so load_embedder()'s scan
# for a BaseEmbedder subclass in this namespace finds only EmbedderYamnetGain4.
_yamnet = importlib.import_module('embedders.yamnet.embedder')

# Linear gain applied to the waveform before YAMNet's log-mel front end. YAMNet
# takes tf.abs(stft), so a 4x waveform is a 4x magnitude spectrogram, i.e. a
# +log(4) ~= +1.386 offset on log_mel wherever mel >> log_offset (0.001), and a
# softer lift near the noise floor. The frozen BatchNorm / ReLU stack downstream
# makes that non-linear, which is the only reason this is worth extracting.
GAIN = 4.0


class EmbedderYamnetGain4(_yamnet.EmbedderYamnet):
    embeddername = "yamnet_gain4"

    def embed(self, audio):
        return self.model(np.asarray(audio, dtype=np.float32) * GAIN)
