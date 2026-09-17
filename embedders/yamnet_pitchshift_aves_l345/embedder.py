import numpy as np

from embedders.embedding import BaseEmbedder

"""IDEAS.md item 20a's follow-up. `aves-mid` picked AVES layers 6, 9, 12 by
argument; the offline per-layer sklearn sweep in
`diagnostics/2026-09-16_aves-layer-profile/` (all 12 layers, real
leave-one-fold-out rotation, headline = mean sens@fpr0.005 over the 8 rotating
folds) found the opposite depth profile: layer 4 was the single best layer on
the headline (0.294) and on every named hard fold (`1_114` 0.355, `1_150`
0.234, `1_37` 0.227), with layers 3 and 5 close behind (0.281, 0.283); layers
9 and 12 -- two of `aves-mid`'s three picks -- were among the *worst*
(0.259, 0.253). `1_95` stayed flat and low across every layer (0.055-0.088),
consistent with it being an aircraft-representation problem AVES depth alone
doesn't touch (see item 2).

This embedder swaps `aves-mid`'s layers {6, 9, 12} for {3, 4, 5} at the same
width (4352-d), so it is a one-variable comparison. Its cache is built by
`tools/slice_layers.py` from the already-extracted
`yamnet_pitchshift_aves_all12` cache (a column slice, same frame grid, no
re-extraction) rather than by this class's own `embed()` -- same pattern as
`yamnet_pitchshift_aves_mid_perch`. `embed()` is kept faithful to the direct
audio path anyway (mirroring `yamnet_pitchshift_aves_mid`) so the embedder is
usable standalone if the derived cache is ever regenerated from scratch.
"""

import os

_YAMNET_SAMPLES = 15360   # 0.96 s at 16 kHz, centred in the 1.0 s frame
_CROP0 = (16000 - _YAMNET_SAMPLES) // 2   # 320
_LAYER_INDICES = (2, 3, 4)   # 0-indexed: AVES layers 3, 4, 5


class EmbedderYamnetPitchshiftAvesL345(BaseEmbedder):
    embeddername = "yamnet_pitchshift_aves_l345"
    framelength_s = 1.0
    digits_time = 2
    samplerate = 16000
    n_embeddings = 1024 + 1024 + 768 * len(_LAYER_INDICES)
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
        self._half_sr = _YAMNET_SAMPLES // 2

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
                f'yamnet_pitchshift_aves_l345: YAMNet returned {len(unshifted)} '
                f'frames for {n} crops (unshifted)'
            )

        shifted_crops = np.stack([self._pitch_up_octave(c) for c in crops])
        shifted = np.asarray(self.yamnet(shifted_crops.reshape(-1)), dtype=np.float32)
        if len(shifted) != n:
            raise ValueError(
                f'yamnet_pitchshift_aves_l345: YAMNet returned {len(shifted)} '
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
                        f'yamnet_pitchshift_aves_l345: extract_features returned '
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
