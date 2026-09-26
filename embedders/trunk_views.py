"""Shared base for depth12 trunk embedders that add transformed views of each
frame: [plain, view_1, ..., view_k], every view through ONE shared fine-tuned
tail (TimeDistributed, n_ctx = k + 1), codes concatenated into the head.

Not an embedder itself (no directory, never loaded by name). Subclasses set
`views` (method names below, in order) and `n_ctx`/`n_embeddings` to match.

Built 2026-09-26 for the pitch-shift method x direction grid (see HANDOFF.md):
resample (the v4-ft-ps mechanism), resample over both halves of the frame, and
phase vocoder (librosa.effects.pitch_shift, duration-preserving).

No to_onnx(): CV only. Add an export before deploying any subclass.
"""
import importlib

import numpy as np

_trunk12 = importlib.import_module('embedders.yamnet_trunk_depth12.embedder')

VIEW_WIDTH = _trunk12.EmbedderYamnetTrunkDepth12.n_embeddings  # 12288


class TrunkViewsDepth12(_trunk12.EmbedderYamnetTrunkDepth12):
    views = ()

    def initialize(self):
        import librosa
        self._librosa = librosa
        self._frame_samples = int(round(self.framelength_s * self.samplerate))
        super().initialize()  # loads self.model = trunk through layer11, warms it

    # --- views: (n, S) float32 frames -> (n, S) float32 ---------------------

    def _fit(self, x):
        """Pad or crop a 1-D signal to one frame."""
        S = self._frame_samples
        if len(x) < S:
            x = np.pad(x, (0, S - len(x)))
        return x[:S].astype(np.float32)

    def _up2(self, x):
        # resample to half rate, relabel at full rate: one octave up, half length
        return self._librosa.resample(x, orig_sr=self.samplerate, target_sr=self.samplerate // 2)

    def _down2(self, x):
        # resample to double rate, relabel at full rate: one octave down, double length
        return self._librosa.resample(x, orig_sr=self.samplerate, target_sr=self.samplerate * 2)

    def resample_up(self, frames):
        """v4-ft-ps's up view: whole frame an octave up, tiled twice to fill."""
        return np.stack([self._fit(np.tile(self._up2(f), 2)) for f in frames])

    def resample_down_first(self, frames):
        """First half of the frame (0-0.48 s), an octave down, filling the frame."""
        h = self._frame_samples // 2
        return np.stack([self._fit(self._down2(f[:h])) for f in frames])

    def resample_down_second(self, frames):
        """Second half of the frame (0.48-0.96 s), an octave down, filling the frame."""
        h = self._frame_samples // 2
        return np.stack([self._fit(self._down2(f[h:2 * h])) for f in frames])

    def vocoder_up(self, frames):
        """Phase vocoder, +12 semitones, same duration. Batched over frames."""
        return self._librosa.effects.pitch_shift(
            frames, sr=self.samplerate, n_steps=12).astype(np.float32)

    def vocoder_down(self, frames):
        """Phase vocoder, -12 semitones, same duration. Batched over frames."""
        return self._librosa.effects.pitch_shift(
            frames, sr=self.samplerate, n_steps=-12).astype(np.float32)

    # --- embed ----------------------------------------------------------------

    def embed(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio) // self._frame_samples
        if n == 0:
            return np.empty((0, self.n_embeddings), dtype=np.float16)
        usable = audio[:n * self._frame_samples]

        trunk_embed = _trunk12.EmbedderYamnetTrunkDepth12.embed
        plain = trunk_embed(self, usable)  # (n, 12288) float16, layer11 features
        if len(plain) != n:
            raise ValueError(
                f'{self.embeddername}: trunk returned {len(plain)} frames for {n} input frames'
            )

        frames = usable.reshape(n, self._frame_samples)
        out = [plain]
        for v in self.views:
            shifted = getattr(self, v)(frames)
            out.append(trunk_embed(self, shifted.reshape(-1)))

        return np.concatenate(out, axis=1).astype(np.float16)
