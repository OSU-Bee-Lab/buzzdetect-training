"""YAMNet over a 3x-long frame in one trunk pass, for `yamnet_triple_overlay`
and `yamnet_triple_squeeze`.

YAMNet's frontend computes a 288x64 log-mel patch for each 2.88 s frame (same
STFT and mel params as stock YAMNet); a reduction brings it down to 96x64, the
one patch YAMNet's conv stack expects; the stack runs once per frame. That's
~1/3 the trunk work of plain `yamnet`, which is the point.

Reductions work on mel magnitude (log undone, then redone), treating it as
power so mixing is incoherent:
- overlay: the three 0.96 s thirds stacked, power summed. Roughly the three
  clips playing at once: real timescale, but they can mask each other.
- squeeze: adjacent STFT frames pooled in threes, mean power. Time order kept,
  but everything is 3x faster (pitch unchanged); YAMNet never saw that.

self.model is a single Keras model (waveform -> 1024), so embed() is one call
on-device and BaseEmbedder.to_onnx() exports it as-is.
"""
import os

import numpy as np

from embedders.embedding import BaseEmbedder

SUBFRAMES = 3
_GAP_LAYER = 'global_average_pooling2d'


def _reduce_patches(features, mode, log_offset):
    """(n, 288, 64) log-mel -> (n, 96, 64) log-mel."""
    import tensorflow as tf
    power = tf.square(tf.exp(features) - log_offset)
    n, bands = tf.shape(power)[0], power.shape[-1]
    frames = power.shape[1] // SUBFRAMES
    if mode == 'overlay':
        # (n, 3, 96, 64): thirds of the frame, summed
        power = tf.reduce_sum(tf.reshape(power, [n, SUBFRAMES, frames, bands]), axis=1)
    elif mode == 'squeeze':
        # (n, 96, 3, 64): adjacent STFT frames, averaged
        power = tf.reduce_mean(tf.reshape(power, [n, frames, SUBFRAMES, bands]), axis=2)
    else:
        raise ValueError(f'unknown mode {mode!r}')
    return tf.math.log(tf.sqrt(power) + log_offset)


def _make_layer_class():
    import keras
    import tensorflow as tf
    from embedders.yamnet import features as features_lib

    class PooledWaveformFeatures(keras.layers.Layer):
        """Waveform -> (n_frames, 96, 64) reduced log-mel patches."""

        def __init__(self, params, mode, **kwargs):
            super().__init__(**kwargs)
            self.params = params
            self.mode = mode

        def call(self, waveform):
            padded = features_lib.pad_waveform(waveform, self.params)
            _, features = features_lib.waveform_to_log_mel_spectrogram_patches(padded, self.params)
            return _reduce_patches(features, self.mode, self.params.log_offset)

    return PooledWaveformFeatures


class EmbedderYamnetPooled(BaseEmbedder):
    mode: str = None
    framelength_s = 0.96 * SUBFRAMES  # 2.88
    digits_time = 2
    samplerate = 16000
    n_embeddings = 1024
    dtype_in = 'float32'

    def initialize(self):
        import copy

        import keras
        import tensorflow as tf  # noqa: F401 -- deferred, see yamnet/embedder.py
        from embedders.yamnet.yamnet import WaveformFeatures  # registers custom layer
        _ = WaveformFeatures.dtype

        yamnet_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yamnet')
        full = keras.models.load_model(os.path.join(yamnet_dir, 'yamnet.keras'), compile=False)
        # Conv stack from YAMNet's 96x64 patch input to the 1024-d GAP.
        core = keras.Model(inputs=full.layers[1].output[1],
                           outputs=full.get_layer(_GAP_LAYER).output)

        params = copy.copy(full.layers[1].params)
        params.patch_window_seconds = self.framelength_s
        params.patch_hop_seconds = self.framehop_s

        waveform = keras.layers.Input(shape=(), dtype='float32', name='waveform')
        patches = _make_layer_class()(params, self.mode, name=f'{self.mode}_features')(waveform)
        self.model = keras.Model(waveform, core(patches), name=f'yamnet_triple_{self.mode}')
        # warm the TF thread pool before librosa/Accelerate claims threads;
        # also the call Keras needs before it will export
        self.embed(np.zeros(int(self.framelength_s * self.samplerate), dtype=np.float32))

    def embed(self, audio):
        return self.model(audio)
