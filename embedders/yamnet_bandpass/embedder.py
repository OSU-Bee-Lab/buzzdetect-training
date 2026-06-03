import os

import numpy as np

from embedders.embedding import BaseEmbedder


class EmbedderYamnetBandpass(BaseEmbedder):
    embeddername = "yamnet_bandpass"
    framelength_s = 0.96
    digits_time = 2
    samplerate = 16000
    n_embeddings = 1024
    dtype_in = 'float32'

    LOWCUT_HZ  = 100
    HIGHCUT_HZ = 3000
    NUM_TAPS   = 201  # odd → linear phase, group delay = (N-1)/2 = 100 samples

    def initialize(self):
        import tensorflow as tf
        from embedders.yamnet.yamnet import WaveformFeatures  # needed for model loading
        _ = WaveformFeatures.dtype
        curdir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(curdir, '..', 'yamnet', 'yamnet.keras')
        model = tf.keras.models.load_model(model_path, compile=False)
        model.layers[1].params.patch_hop_seconds = self.framehop_s
        self.model = model
        self._filter_kernel = self._build_filter_kernel()

    def _build_filter_kernel(self):
        """Hamming-windowed sinc bandpass FIR, returned as a tf constant."""
        import tensorflow as tf
        sr   = self.samplerate
        low  = self.LOWCUT_HZ
        high = self.HIGHCUT_HZ
        n    = self.NUM_TAPS

        # Centered sample indices
        t = np.arange(n) - (n - 1) / 2.0

        # Bandpass = LP(high) − LP(low); np.sinc is normalized (sin(πx)/(πx))
        h = (2 * high / sr) * np.sinc(2 * high / sr * t) \
          - (2 * low  / sr) * np.sinc(2 * low  / sr * t)
        h *= np.hamming(n)

        # Normalize so passband gain ≈ 1 at center frequency
        center = (low + high) / 2.0
        response_at_center = np.dot(h, np.exp(2j * np.pi * center / sr * np.arange(n)))
        h /= np.abs(response_at_center)

        # Shape expected by tf.nn.conv1d: [filter_width, in_channels, out_channels]
        return tf.constant(h.astype(np.float32).reshape(n, 1, 1), dtype=tf.float32)

    def embed(self, audio):
        import tensorflow as tf
        x = tf.cast(audio, tf.float32)
        # conv1d wants [batch, width, channels]
        x = x[tf.newaxis, :, tf.newaxis]
        filtered = tf.nn.conv1d(x, self._filter_kernel, stride=1, padding='SAME')
        filtered = tf.squeeze(filtered, axis=[0, 2])
        return self.model(filtered)
