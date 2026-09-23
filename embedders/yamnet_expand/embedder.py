"""YAMNet on 2x time-expanded audio: 0.48 s frames, one YAMNet patch each.

YAMNet's own frontend with the STFT hop halved (10 ms -> 5 ms; same 25 ms
window, 512-point FFT and 64 mel bands), so 0.48 s of audio fills the
96-frame patch YAMNet expects. The model hears the audio 2x slower at the
same pitch. The finer timing is real (every 5 ms is its own STFT frame),
not an interpolated spectrogram. The inverse of `yamnet_triple_squeeze`.

Costs 2x plain `yamnet` per second of audio: one trunk pass per 0.48 s.

self.model is a single Keras model (waveform -> 1024), so embed() is one call
on-device and BaseEmbedder.to_onnx() exports it as-is.
"""
import os

import numpy as np

from embedders.embedding import BaseEmbedder

EXPANSION = 2
_GAP_LAYER = 'global_average_pooling2d'


class EmbedderYamnetExpand(BaseEmbedder):
    embeddername = "yamnet_expand"
    framelength_s = 0.96 / EXPANSION  # 0.48
    digits_time = 2
    samplerate = 16000
    n_embeddings = 1024
    dtype_in = 'float32'

    def initialize(self):
        import copy

        import keras
        import tensorflow as tf  # noqa: F401 -- deferred, see yamnet/embedder.py
        from embedders.yamnet.yamnet import WaveformFeatures  # also registers it for loading

        yamnet_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'yamnet')
        full = keras.models.load_model(os.path.join(yamnet_dir, 'yamnet.keras'), compile=False)
        # Conv stack from YAMNet's 96x64 patch input to the 1024-d GAP.
        core = keras.Model(inputs=full.layers[1].output[1],
                           outputs=full.get_layer(_GAP_LAYER).output)

        params = copy.copy(full.layers[1].params)
        params.stft_hop_seconds = params.stft_hop_seconds / EXPANSION
        params.patch_window_seconds = self.framelength_s
        params.patch_hop_seconds = self.framehop_s
        # loaded params are a SimpleNamespace, without Params.patch_frames
        patch_frames = round(params.patch_window_seconds / params.stft_hop_seconds)
        assert patch_frames == 96, patch_frames

        waveform = keras.layers.Input(shape=(), dtype='float32', name='waveform')
        _, patches = WaveformFeatures(params, name='expand_features')(waveform)
        self.model = keras.Model(waveform, core(patches), name='yamnet_expand')
        # warm the TF thread pool before librosa/Accelerate claims threads;
        # also the call Keras needs before it will export
        self.embed(np.zeros(int(self.framelength_s * self.samplerate), dtype=np.float32))

    def embed(self, audio):
        return self.model(audio)
