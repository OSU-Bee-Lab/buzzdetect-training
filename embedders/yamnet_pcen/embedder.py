import os

import numpy as np

from embedders.embedding import BaseEmbedder

"""IDEAS.md item 15: PCEN as a parallel block.

Per-Channel Energy Normalization (Lostanlen et al., arXiv:1911.00417) is an
adaptive per-channel gain whose time constant defines "background" -- exactly
the failure mode behind this era's three worst hard-fold thresholds (`1_95`
aircraft, `1_114` orthopteran trill, `1_37` "background"), all of which sit at
a 1-2 s+ time scale that a ~1 s buzz does not share.

Only defensible as a parallel block: replacing YAMNet's own frontend would
corrupt the input distribution it was trained on, so this computes an
independent, hand-built PCEN feature (mean-pooled per mel band over the frame)
and concatenates it with the unmodified YAMNet embedding, same frame grid
(0.96 s) as plain `yamnet` so this is a clean paired comparison against it.

Per frame: [yamnet(t) 1024-d, pcen_meanpool(t) n_mels-d].
"""

_N_MELS = 64
_N_FFT = 512          # 32 ms at 16 kHz
_HOP_LENGTH = 160      # 10 ms at 16 kHz


class EmbedderYamnetPcen(BaseEmbedder):
    embeddername = "yamnet_pcen"
    framelength_s = 0.96
    digits_time = 2
    samplerate = 16000
    n_embeddings = 1024 + _N_MELS
    dtype_in = 'float32'

    def initialize(self):
        import tensorflow as tf
        from embedders.yamnet.yamnet import WaveformFeatures  # registers custom layer
        _ = WaveformFeatures.dtype

        if os.environ.get('BUZZDETECT_NO_GPU'):
            tf.config.set_visible_devices([], 'GPU')
        else:
            for gpu in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)

        here = os.path.dirname(os.path.realpath(__file__))
        yamnet_keras = os.path.join(here, '..', 'yamnet', 'yamnet.keras')
        self.yamnet = tf.keras.models.load_model(yamnet_keras, compile=False)
        self.yamnet.layers[1].params.patch_hop_seconds = self.framelength_s

        import librosa
        self._librosa = librosa

        self.embed(np.zeros(int(self.framelength_s * self.samplerate), dtype=np.float32))

    def _pcen_pooled(self, crop):
        S = self._librosa.feature.melspectrogram(
            y=crop.astype(np.float64), sr=self.samplerate, n_fft=_N_FFT,
            hop_length=_HOP_LENGTH, n_mels=_N_MELS, power=1.0,
        )
        # librosa.pcen expects a magnitude (power=1) mel spectrogram scaled by
        # its sample rate / hop, per its own docstring convention.
        pcen = self._librosa.pcen(S * (2 ** 31), sr=self.samplerate, hop_length=_HOP_LENGTH)
        return pcen.mean(axis=1).astype(np.float32)

    def embed(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        n_samples_frame = int(self.framelength_s * self.samplerate)
        n = len(audio) // n_samples_frame
        if n == 0:
            return np.empty((0, self.n_embeddings), dtype=np.float32)
        frames = audio[:n * n_samples_frame].reshape(n, n_samples_frame)

        yam = np.asarray(self.yamnet(frames.reshape(-1)), dtype=np.float32)
        if len(yam) != n:
            raise ValueError(
                f'yamnet_pcen: YAMNet returned {len(yam)} frames for {n} crops'
            )

        pcen = np.stack([self._pcen_pooled(f) for f in frames])

        return np.concatenate([yam, pcen], axis=1)
