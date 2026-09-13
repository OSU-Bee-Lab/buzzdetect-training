import importlib

import numpy as np

from embedders.embedding import BaseEmbedder  # noqa: F401  -- load_embedder's scan

"""YAMNet, its octave-shifted twin, and a second two-octave-shifted twin.

IDEAS.md item 5, x4 rung. `yamnet_pitchshift` (x2: resample 16k->8k, tile 2x)
was the era's largest clean gain (+0.069, 8/8 folds). This is its untested
follow-up: a third 1024-d block from resampling 16k->4k and tiling 4x, landing
honey-bee harmonics at 920/1840/2760 Hz instead of 460/920/1380. Concatenated
onto the existing [unshifted, x2] pair -> 3072-d.

Imported through the module (not `from ... import EmbedderYamnetPitchshift`),
so load_embedder()'s scan for a BaseEmbedder subclass in this module's
namespace can only find EmbedderYamnetPitchshiftX4 -- same pattern as
embedders/yamnet_context_aves/embedder.py.
"""

_yps = importlib.import_module('embedders.yamnet_pitchshift.embedder')


class EmbedderYamnetPitchshiftX4(_yps.EmbedderYamnetPitchshift):
    embeddername = "yamnet_pitchshift_x4"
    n_embeddings = 3072  # 1024 unshifted + 1024 x2-shifted + 1024 x4-shifted

    def initialize(self):
        self._quarter_sr = self.samplerate // 4
        super().initialize()  # parent's warm-up call reaches embed() below

    def _pitch_up_two_octaves(self, frame):
        """One 0.96 s frame -> its two-octave-shifted, re-tiled quadruplet."""
        shifted = self._librosa.resample(
            frame, orig_sr=self.samplerate, target_sr=self._quarter_sr,
        ).astype(np.float32)
        tiled = np.tile(shifted, 4)
        if len(tiled) < self._frame_samples:
            tiled = np.pad(tiled, (0, self._frame_samples - len(tiled)))
        return tiled[:self._frame_samples]

    def embed(self, audio):
        """[unshifted, x2-shifted, x4-shifted], (n_frames, 3072)."""
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio) // self._frame_samples
        if n == 0:
            return np.empty((0, self.n_embeddings), dtype=np.float32)
        usable = audio[:n * self._frame_samples]

        unshifted = np.asarray(self.model(usable), dtype=np.float32)
        if len(unshifted) != n:
            raise ValueError(
                f'yamnet_pitchshift_x4: YAMNet returned {len(unshifted)} '
                f'frames for {n} input frames'
            )

        frames = usable.reshape(n, self._frame_samples)
        x2_frames = np.stack([self._pitch_up_octave(f) for f in frames])
        x2 = np.asarray(self.model(x2_frames.reshape(-1)), dtype=np.float32)

        x4_frames = np.stack([self._pitch_up_two_octaves(f) for f in frames])
        x4 = np.asarray(self.model(x4_frames.reshape(-1)), dtype=np.float32)

        return np.concatenate([unshifted, x2, x4], axis=1)
