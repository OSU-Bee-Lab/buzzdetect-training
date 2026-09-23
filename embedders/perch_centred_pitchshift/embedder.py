"""perch_centred's real 5.0 s centred Perch window, concatenated with
yamnet_pitchshift's dual-view [plain, octave-up] YAMNet concat -- both frozen,
no fine-tuning.

Follow-up to `perch-centred` (2026-09-23): centred Perch clears its
pre-registered falsifier decisively on both `1_95` (jet, +0.180 +/- 0.048)
and `1_114` (trill, +0.294 +/- 0.101), the two hard folds it was aimed at --
but three rich folds (`1_29`, `1_143`, `1_11`) collapse, netting a flat
headline. This is the exact inverse of the era's context/averaging prior
(context levers flatter rich folds, leave hard folds flat). If the two
mechanisms are genuinely separable (targeted hard-fold signal from Perch,
general sensitivity from YAMNet + pitch-shift's confirmed +0.069 lever),
concatenating them should keep both folds' gains rather than trading them.

Tests the *frozen* stack first (matching perch-centred's own frozen regime,
and yamnet_pitchshift's confirmed frozen lever) -- fine-tuning the trunk on
top is a separate, more expensive follow-up if this pays.

Two sample rates share one embed() call: Perch's own 32 kHz (native,
`context_frames=3` for the real 5.0 s window) and YAMNet's 16 kHz (each
0.96 s frame downsampled from the 32 kHz buffer, then the existing octave-up
mechanism from `yamnet_pitchshift`/`yamnet_trunk_pitchshift` runs at 16 kHz
as before). No context is read for the YAMNet half -- only the frame's own
audio, downsampled -- so the pitch-shift mechanism is exactly as confirmed
elsewhere, not a new variant.

Composition, not inheritance: `EmbedderPerchCentred.embed()` sizes its output
array from `self.n_embeddings`, which would resolve to *this* class's wider
value if this class subclassed it and called `super().embed()`. A held inner
instance keeps that binding correct.
"""
import importlib
import os

import numpy as np

from embedders.embedding import BaseEmbedder

_pc = importlib.import_module('embedders.perch_centred.embedder')

_YAMNET_SR = 16000
_YAMNET_FRAME_SAMPLES = int(round(0.96 * _YAMNET_SR))  # 15360


class EmbedderPerchCentredPitchshift(BaseEmbedder):
    embeddername = "perch_centred_pitchshift"
    framelength_s = 0.96
    digits_time = 2
    samplerate = 32000  # Perch's native rate; the YAMNet half downsamples per-frame
    # 1536 perch + 1024 yamnet (plain) + 1024 yamnet (octave-up)
    n_embeddings = _pc.EmbedderPerchCentred.n_embeddings + 2048
    dtype_in = 'float32'
    context_frames = _pc.EmbedderPerchCentred.context_frames

    def initialize(self):
        self._perch = _pc.EmbedderPerchCentred(framehop_prop=self.framehop_prop)
        self._perch.initialize()
        self.frame_samples = self._perch.frame_samples  # 30720 @ 32 kHz

        import tensorflow as tf
        from embedders.yamnet.yamnet import WaveformFeatures  # registers custom layer
        _ = WaveformFeatures.dtype

        here = os.path.dirname(os.path.realpath(__file__))
        yamnet_keras = os.path.join(here, '..', 'yamnet', 'yamnet.keras')
        yamnet = tf.keras.models.load_model(yamnet_keras, compile=False)
        yamnet.layers[1].params.patch_hop_seconds = 0.96
        self.yamnet = yamnet

        import librosa
        self._librosa = librosa

        self.embed(np.zeros(self.frame_samples * (2 * self.context_frames + 1), dtype=np.float32))

    def _to_yamnet_rate(self, frame_32k):
        """One 0.96 s frame at 32 kHz -> the same audio at YAMNet's native 16 kHz."""
        resampled = self._librosa.resample(
            frame_32k, orig_sr=self.samplerate, target_sr=_YAMNET_SR,
        ).astype(np.float32)
        if len(resampled) < _YAMNET_FRAME_SAMPLES:
            resampled = np.pad(resampled, (0, _YAMNET_FRAME_SAMPLES - len(resampled)))
        return resampled[:_YAMNET_FRAME_SAMPLES]

    def _pitch_up_octave(self, frame_16k):
        """Identical mechanism to yamnet_pitchshift._pitch_up_octave, at 16 kHz."""
        shifted = self._librosa.resample(
            frame_16k, orig_sr=_YAMNET_SR, target_sr=_YAMNET_SR // 2,
        ).astype(np.float32)
        tiled = np.tile(shifted, 2)
        if len(tiled) < _YAMNET_FRAME_SAMPLES:
            tiled = np.pad(tiled, (0, _YAMNET_FRAME_SAMPLES - len(tiled)))
        return tiled[:_YAMNET_FRAME_SAMPLES]

    def embed(self, audio):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        n = len(audio) // self.frame_samples
        if n == 0:
            return np.zeros((0, self.n_embeddings), dtype=np.float32)

        perch = self._perch.embed(audio)  # (n, 1536), real centred 5.0 s windows
        if len(perch) != n:
            raise ValueError(
                f'{self.embeddername}: perch_centred returned {len(perch)} frames for {n} input frames'
            )

        frames_32k = audio[:n * self.frame_samples].reshape(n, self.frame_samples)
        frames_16k = np.stack([self._to_yamnet_rate(f) for f in frames_32k])

        unshifted = np.asarray(self.yamnet(frames_16k.reshape(-1)), dtype=np.float32)
        if len(unshifted) != n:
            raise ValueError(
                f'{self.embeddername}: yamnet returned {len(unshifted)} frames for {n} input frames'
            )
        shifted_frames = np.stack([self._pitch_up_octave(f) for f in frames_16k])
        shifted = np.asarray(self.yamnet(shifted_frames.reshape(-1)), dtype=np.float32)

        return np.concatenate([unshifted, shifted, perch], axis=1).astype(np.float32)
