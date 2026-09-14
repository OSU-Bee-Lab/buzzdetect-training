import os

import numpy as np

from embedders.embedding import BaseEmbedder

"""Pitch-shifted YAMNet + AVES middle layers, on one shared 1.0 s frame grid.

IDEAS.md item 17. This era's two strongest, mechanistically distinct
confirmed levers: `yamnet_pitchshift` (+0.069, 8/8 folds, twice confirmed --
YAMNet's frequency register) and `aves-mid` (+0.026 to +0.032 headline, both
draws, with `1_150`/`1_29`/willard/`1_95` all up both draws -- AVES's
representation depth). Unlike item 16 (context stacked onto pitch-shift,
which traded rich-fold gain for hard-fold cost), `aves-mid`'s own gain is
*itself* concentrated on the hard folds, so this pairing has a real chance of
being additive on the folds that matter rather than the era's usual
rich/hard trade-off.

The two source embedders live on different native frame grids
(`yamnet_pitchshift`: 0.96 s: `yamnet_aves_mid`: 1.0 s), so they cannot be
concatenated by reading their existing caches -- exactly the problem
`yamnet_aves`'s own docstring names. This embedder resolves it the same way
`yamnet_aves` did: declare `framelength_s = 1.0`, centre-crop 0.96 s of each
1.0 s frame for every YAMNet pass, and feed the full 1.0 s to AVES.

Per frame: `[yamnet(t), yamnet(pitch_up_octave(t)), aves_layer6(t),
aves_layer9(t), aves_layer12(t)]` = 1024 + 1024 + 3*768 = 4352-d. The AVES
side is layers only (no plain-YAMNet block, unlike `yamnet_aves` /
`yamnet_aves_mid`) -- the unshifted YAMNet block above already supplies that
representation once, and concatenating it twice would be redundant capacity,
not a new signal.
"""

_YAMNET_SAMPLES = 15360   # 0.96 s at 16 kHz, centred in the 1.0 s frame
_CROP0 = (16000 - _YAMNET_SAMPLES) // 2   # 320
_LAYER_INDICES = (5, 8, 11)   # 0-indexed: AVES layers 6, 9, 12


class EmbedderYamnetPitchshiftAvesMid(BaseEmbedder):
    embeddername = "yamnet_pitchshift_aves_mid"
    framelength_s = 1.0
    digits_time = 2
    samplerate = 16000
    n_embeddings = 1024 + 1024 + 768 * len(_LAYER_INDICES)  # unshifted + shifted YAMNet + AVES layers
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
        """One 0.96 s YAMNet crop -> its octave-shifted, re-tiled twin. Same
        mechanism as embedders/yamnet_pitchshift/embedder.py."""
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
                f'yamnet_pitchshift_aves_mid: YAMNet returned {len(unshifted)} '
                f'frames for {n} crops (unshifted)'
            )

        shifted_crops = np.stack([self._pitch_up_octave(c) for c in crops])
        shifted = np.asarray(self.yamnet(shifted_crops.reshape(-1)), dtype=np.float32)
        if len(shifted) != n:
            raise ValueError(
                f'yamnet_pitchshift_aves_mid: YAMNet returned {len(shifted)} '
                f'frames for {n} crops (shifted)'
            )

        out = []
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                stop = min(start + self.batch_size, n)
                wav = torch.from_numpy(frames[start:stop]).to(self.device)
                layer_outputs, _ = self.aves.extract_features(wav)
                if len(layer_outputs) <= max(_LAYER_INDICES):
                    raise ValueError(
                        f'yamnet_pitchshift_aves_mid: extract_features returned '
                        f'{len(layer_outputs)} layers, need index '
                        f'{max(_LAYER_INDICES)}'
                    )
                pooled = torch.cat(
                    [layer_outputs[i].mean(dim=1) for i in _LAYER_INDICES],
                    dim=1,
                )
                out.append(pooled.cpu().numpy())
        av = np.concatenate(out, axis=0).astype(np.float32)

        return np.concatenate([unshifted, shifted, av], axis=1)
