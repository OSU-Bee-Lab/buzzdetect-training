import os

import numpy as np

from embedders.embedding import BaseEmbedder

"""YAMNet (unmodified) + AVES fed a decimated octave-up input instead of the
raw frame.

IDEAS.md item 22. `yamnet-pitchshift` (+0.069, 8/8 folds) is attributed to
YAMNet's thin mel coverage near ~230 Hz -- a front-end-specific explanation.
AVES has no mel filterbank (it reads raw waveform through a conv front end),
so if the octave-up shift *also* helps AVES, the win is about where the
pretraining data's spectral energy sat, not about YAMNet's front end
specifically, and shifting becomes a lever worth trying on every encoder
(Perch, a future bioacoustic SSL model). If AVES is flat or down, the win is
YAMNet-specific and nobody should spend an extraction shifting other
encoders.

One variable against `yamnet_aves_mid`: the YAMNet block is byte-identical
(same `_CROP0` 0.96 s crop of the untouched frame); the AVES block reads the
**decimated, seamless** up-shift instead of the raw 1.0 s frame. Per the
seamless-shift mechanism (`yamnet_pitchshift_decimate`'s docstring), 30720
real, contiguous samples starting at frame t's own start -- t's full 16000
samples plus the leading 14720 samples of its real successor t+1, supplied by
`context_frames = 1` -- are resampled down to 16000 samples (AVES's native
1.0 s frame width), a ~1.92x compression that raises all frequencies by
roughly (not exactly) an octave. A frame at a true recording edge has no real
successor and clamps to itself, degrading gracefully to a tiled-style shift
for that one row only.

1024 YAMNet + 3*768 AVES(shifted) = 3328-d, same width as `yamnet_aves_mid`.
"""

_YAMNET_SAMPLES = 15360   # 0.96 s at 16 kHz, centred in the 1.0 s frame
_CROP0 = (16000 - _YAMNET_SAMPLES) // 2   # 320
_AVES_INPUT_SAMPLES = 16000   # AVES's native 1.0 s frame width
_DECIMATE_INPUT_SAMPLES = 2 * _AVES_INPUT_SAMPLES  # 30720: frame t's start through frame t+1's end
_LAYER_INDICES = (5, 8, 11)   # 0-indexed: AVES layers 6, 9, 12


class EmbedderYamnetAvesMidAvesshift(BaseEmbedder):
    embeddername = "yamnet_aves_mid_avesshift"
    framelength_s = 1.0
    context_frames = 1
    digits_time = 2
    samplerate = 16000
    n_embeddings = 1024 + 768 * len(_LAYER_INDICES)  # YAMNet (untouched) + AVES layers (shifted input)
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
        # resample 30720 real samples down to exactly 16000: a ~1.92x
        # compression, not a clean octave, because AVES's frame is 1.0 s wide
        # while the borrowed window is built from two 1.0 s frames.
        self._target_sr = self.samplerate * _AVES_INPUT_SAMPLES / _DECIMATE_INPUT_SAMPLES

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

        self.embed(np.zeros(self.samplerate * 3, dtype=np.float32))

    def _decimate_pair(self, samples_30720):
        """30720 real, contiguous samples (frame t's start through frame
        t+1's end) -> one 16000-sample buffer, AVES's native frame width."""
        decimated = self._librosa.resample(
            samples_30720, orig_sr=self.samplerate, target_sr=self._target_sr,
        ).astype(np.float32)
        if len(decimated) < _AVES_INPUT_SAMPLES:
            decimated = np.pad(decimated, (0, _AVES_INPUT_SAMPLES - len(decimated)))
        return decimated[:_AVES_INPUT_SAMPLES]

    def embed(self, audio):
        import torch

        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio) // self.samplerate
        if n == 0:
            return np.empty((0, self.n_embeddings), dtype=np.float32)
        frames = audio[:n * self.samplerate].reshape(n, self.samplerate)

        crops = frames[:, _CROP0:_CROP0 + _YAMNET_SAMPLES]
        yam = np.asarray(self.yamnet(crops.reshape(-1)), dtype=np.float32)
        if len(yam) != n:
            raise ValueError(
                f'{self.embeddername}: YAMNet returned {len(yam)} frames for {n} crops'
            )

        # Real successor per row; clamps to itself at a true buffer edge (see
        # module docstring and yamnet_pitchshift_decimate's own clamp).
        next_frames = np.concatenate([frames[1:], frames[-1:]], axis=0)
        n_head = _DECIMATE_INPUT_SAMPLES - self.samplerate  # 14720
        pairs = np.concatenate([frames, next_frames[:, :n_head]], axis=1)  # (n, 30720)
        shifted_frames = np.stack([self._decimate_pair(p) for p in pairs])

        out = []
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                stop = min(start + self.batch_size, n)
                wav = torch.from_numpy(shifted_frames[start:stop]).to(self.device)
                layer_outputs, _ = self.aves.extract_features(wav)
                if len(layer_outputs) <= max(_LAYER_INDICES):
                    raise ValueError(
                        f'{self.embeddername}: extract_features returned '
                        f'{len(layer_outputs)} layers, need index '
                        f'{max(_LAYER_INDICES)}'
                    )
                pooled = torch.cat(
                    [layer_outputs[i].mean(dim=1) for i in _LAYER_INDICES],
                    dim=1,
                )
                out.append(pooled.cpu().numpy())
        av = np.concatenate(out, axis=0).astype(np.float32)

        return np.concatenate([yam, av], axis=1)
