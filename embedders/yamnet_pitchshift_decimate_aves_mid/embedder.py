import os

import numpy as np

from embedders.embedding import BaseEmbedder

"""Decimated (seamless) pitch-shifted YAMNet + AVES middle layers.

IDEAS.md item 19. `pitchshift-aves-mid` (era lead, 0.422/0.411 on two draws)
uses the **tiled** octave shift: 0.48 s of real audio played twice, with a hard
seam at the midpoint. `pitchshift-decimate-up` showed the seamless
**decimated** shift (real, contiguous audio compressed 2x, no repeat) moves
`1_95` and `1_114` -- two of the lead's three FP-limited folds -- while
trading away rich-fold sensitivity elsewhere; `aves-mid` supplies exactly that
rich-fold sensitivity back. This embedder swaps the lead's shifted block for
the decimated form, one variable against `yamnet_pitchshift_aves_mid`.

Per frame: `[yamnet(crop_t), yamnet(decimate(crop_t .. crop_t+30720)),
aves_layer6(frame_t), aves_layer9(frame_t), aves_layer12(frame_t)]`
= 1024 + 1024 + 3*768 = 4352-d, same width and AVES layers as the lead.

Frame grid: `framelength_s = 1.0` (AVES needs the full second), YAMNet reads a
centred 0.96 s crop of each 1.0 s frame -- the same `_CROP0` mechanism
`yamnet_pitchshift_aves_mid` uses to put both encoders on one grid. The
decimated block needs 30720 real samples starting at the crop's own start
position: 15680 of them are the tail of frame t (from `_CROP0` to the frame's
end), and the remaining 15040 are the head of frame t+1, supplied by
`context_frames = 1`'s real-audio buffer (never read from cached rows -- see
`yamnet_pitchshift_decimate`'s docstring for why that distinction matters). A
frame at a true recording edge has no real successor; it clamps to itself
(pads with its own tail), degrading gracefully to a tiled-style shift for that
one row, exactly as `yamnet_pitchshift_decimate` does.
"""

_YAMNET_SAMPLES = 15360   # 0.96 s at 16 kHz, centred in the 1.0 s frame
_CROP0 = (16000 - _YAMNET_SAMPLES) // 2   # 320
_DECIMATE_INPUT_SAMPLES = 2 * _YAMNET_SAMPLES  # 30720: crop_t's start to crop_t+1's end
_LAYER_INDICES = (5, 8, 11)   # 0-indexed: AVES layers 6, 9, 12


class EmbedderYamnetPitchshiftDecimateAvesMid(BaseEmbedder):
    embeddername = "yamnet_pitchshift_decimate_aves_mid"
    framelength_s = 1.0
    context_frames = 1
    digits_time = 2
    samplerate = 16000
    n_embeddings = 1024 + 1024 + 768 * len(_LAYER_INDICES)  # unshifted + decimated-shifted YAMNet + AVES layers
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
        self._half_sr = _YAMNET_SAMPLES // 2  # decimated-frame samples halved -> octave up on playback

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
        """30720 real, contiguous samples (crop_t's start through crop_t+1's
        end) -> one 15360-sample buffer: a genuine 1.92 s of real audio
        compressed 2x, no tile, no seam."""
        decimated = self._librosa.resample(
            samples_30720, orig_sr=self.samplerate, target_sr=self._half_sr,
        ).astype(np.float32)
        if len(decimated) < _YAMNET_SAMPLES:
            decimated = np.pad(decimated, (0, _YAMNET_SAMPLES - len(decimated)))
        return decimated[:_YAMNET_SAMPLES]

    def embed(self, audio):
        """
        Args:
            audio: numpy array of audio samples at self.samplerate, a whole
                number of self.framelength_s frames, in time order and
                contiguous -- extraction hands this a buffer padded by
                context_frames real frames on each side (see
                02_set/extract.py's `_embed_with_context`); framehop_prop must
                be 1.

        Returns:
            numpy array of embeddings, (n_frames, 4352)
        """
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
                f'{self.embeddername}: YAMNet returned {len(unshifted)} '
                f'frames for {n} crops (unshifted)'
            )

        # Real successor per row; clamps to itself at a true buffer edge (see
        # module docstring and yamnet_pitchshift_decimate's own clamp).
        next_frames = np.concatenate([frames[1:], frames[-1:]], axis=0)
        tail = frames[:, _CROP0:]  # (n, 15680): crop_t's start to frame t's end
        n_head = _DECIMATE_INPUT_SAMPLES - tail.shape[1]  # 15040
        head = next_frames[:, :n_head]
        pairs = np.concatenate([tail, head], axis=1)  # (n, 30720)
        decimated_frames = np.stack([self._decimate_pair(p) for p in pairs])
        shifted = np.asarray(self.yamnet(decimated_frames.reshape(-1)), dtype=np.float32)
        if len(shifted) != n:
            raise ValueError(
                f'{self.embeddername}: YAMNet returned {len(shifted)} '
                f'frames for {n} crops (decimated-shifted)'
            )

        out = []
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                stop = min(start + self.batch_size, n)
                wav = torch.from_numpy(frames[start:stop]).to(self.device)
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

        return np.concatenate([unshifted, shifted, av], axis=1)
