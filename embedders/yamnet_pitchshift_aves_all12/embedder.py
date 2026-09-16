import os

import numpy as np

from embedders.embedding import BaseEmbedder

"""IDEAS.md item 20a: all 12 AVES transformer layers, mean-pooled, alongside
the lead's YAMNet blocks. This is a diagnostic cache, not a trained model on
its own -- offline, for each layer, a linear readout is scored against
`03_train/metrics.py:sens_at_fpr` on the cached embeddings (see
`diagnostics/.../layer_profile.py`) to map which AVES depth carries buzz,
before committing to a specific 3-layer concat via a real Keras CV.

`aves-mid` (layers 6, 9, 12) was picked by argument; nobody has looked at
whether a different depth carries more of the signal. `extract_features`
already computes every layer at no extra AVES forward-pass cost -- this just
keeps all twelve instead of a chosen three.

Per frame: [yamnet(t), yamnet(pitch_up_octave(t)), aves_layer1..12(t)] =
1024 + 1024 + 12*768 = 11264-d. Same pitch-shift block as
`yamnet_pitchshift_aves_mid`, so the two share the YAMNet-side signal
mechanistically and differ only in how much of AVES is kept.
"""

_YAMNET_SAMPLES = 15360   # 0.96 s at 16 kHz, centred in the 1.0 s frame
_CROP0 = (16000 - _YAMNET_SAMPLES) // 2   # 320
_N_LAYERS = 12


class EmbedderYamnetPitchshiftAvesAll12(BaseEmbedder):
    embeddername = "yamnet_pitchshift_aves_all12"
    framelength_s = 1.0
    digits_time = 2
    samplerate = 16000
    n_embeddings = 1024 + 1024 + 768 * _N_LAYERS
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
        self.yamnet.layers[1].params.patch_hop_seconds = 0.96

        import librosa
        self._librosa = librosa
        self._half_sr = _YAMNET_SAMPLES // 2  # frame samples halved -> octave up on playback

        import json
        import torch
        import torchaudio

        aves_dir = os.path.join(here, '..', 'aves')
        with open(os.path.join(aves_dir, 'model_config.json')) as f:
            config = json.load(f)
        self.aves = torchaudio.models.wav2vec2_model(**config, aux_num_out=None)
        state = torch.load(os.path.join(aves_dir, 'aves-base-bio.pt'),
                           map_location='cpu', weights_only=True)
        self.aves.load_state_dict(state)
        self.aves.eval()

        device = os.environ.get('BUZZDETECT_AVES_DEVICE', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.batch_size = int(os.environ.get('BUZZDETECT_AVES_BATCH', '64'))
        if device == 'cuda':
            torch.backends.cudnn.enabled = False
        self.aves.to(device)

        self.embed(np.zeros(self.samplerate, dtype=np.float32))

    def _pitch_up_octave(self, crop):
        shifted = self._librosa.resample(
            crop, orig_sr=16000, target_sr=self._half_sr,
        ).astype(np.float32)
        tiled = np.tile(shifted, 2)
        if len(tiled) < _YAMNET_SAMPLES:
            tiled = np.pad(tiled, (0, _YAMNET_SAMPLES - len(tiled)))
        return tiled[:_YAMNET_SAMPLES]

    def embed(self, audio):
        import torch

        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio) // self.samplerate
        if n == 0:
            return np.empty((0, self.n_embeddings), dtype=np.float32)
        frames = audio[:n * self.samplerate].reshape(n, self.samplerate)

        crops = frames[:, _CROP0:_CROP0 + _YAMNET_SAMPLES]
        unshifted = np.asarray(self.yamnet(crops.reshape(-1)), dtype=np.float32)
        if len(unshifted) != n:
            raise ValueError(
                f'yamnet_pitchshift_aves_all12: YAMNet returned {len(unshifted)} '
                f'frames for {n} crops (unshifted)'
            )

        shifted_crops = np.stack([self._pitch_up_octave(c) for c in crops])
        shifted = np.asarray(self.yamnet(shifted_crops.reshape(-1)), dtype=np.float32)
        if len(shifted) != n:
            raise ValueError(
                f'yamnet_pitchshift_aves_all12: YAMNet returned {len(shifted)} '
                f'frames for {n} crops (shifted)'
            )

        out = []
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                stop = min(start + self.batch_size, n)
                wav = torch.from_numpy(frames[start:stop]).to(self.device)
                layer_outputs, _ = self.aves.extract_features(wav)
                if len(layer_outputs) != _N_LAYERS:
                    raise ValueError(
                        f'yamnet_pitchshift_aves_all12: extract_features returned '
                        f'{len(layer_outputs)} layers, expected {_N_LAYERS}'
                    )
                pooled = torch.cat(
                    [layer.mean(dim=1) for layer in layer_outputs],
                    dim=1,
                )
                out.append(pooled.cpu().numpy())
        av = np.concatenate(out, axis=0).astype(np.float32)

        return np.concatenate([unshifted, shifted, av], axis=1)
