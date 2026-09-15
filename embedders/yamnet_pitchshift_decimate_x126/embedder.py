import importlib

import numpy as np

from embedders.embedding import BaseEmbedder

_yamnet = importlib.import_module('embedders.yamnet.embedder')

"""YAMNet, plus a real-audio x1.26-shifted copy of each frame, concatenated.

IDEAS.md item 24: the shift-ratio ladder below one octave. `yamnet_pitchshift`
(tiled x2, +0.069) and `yamnet_pitchshift_decimate` (seamless x2, headline
flat but `1_95`/`1_114` up both draws) bracket ratio 2; nothing between 1 and
2 has been tried, and bee flight fundamentals (~130-250 Hz across taxa) may
be over- or under-shot by a fixed octave. This is the x1.26 rung (a minor
third, 2**(4/12)), built in the same seamless decimated form as
`yamnet_pitchshift_decimate`: no tiling, no repeated content, no seam.

Ratio r needs `r * framelength_s` seconds of real, contiguous audio per row.
`context_frames = 1` supplies frame t plus its true successor t+1 (1.92 s
total), of which only the leading `round(r * frame_samples)` samples are
used -- r=1.26 needs ~1.21 s, comfortably inside the 1.92 s available. Those
samples are resampled down to one frame's width (`samplerate / r`), i.e.
compressed by r, raising every frequency by that same ratio. A frame at a
true recording edge (no real successor) clamps: it decimates its own
truncated single frame against itself, i.e. is short by `(r-1) *
frame_samples` samples of real content for that one row (still lengthened by
zero-padding, never by tiling).

Frame spacing matches the standard 0.96 s grid (frame counts and labels line
up 1:1 with `yamnet_pitchshift_decimate`'s x2 rung, for a clean comparison).
2048-d: unshifted YAMNet block + x1.26-shifted YAMNet block.
"""

RATIO = 2 ** (4 / 12)  # 1.2599..., a minor third
CONTEXT_FRAMES = 1  # one real frame of look-ahead covers any ratio <= 2


class EmbedderYamnetPitchshiftDecimateX126(_yamnet.EmbedderYamnet):
    embeddername = "yamnet_pitchshift_decimate_x126"
    context_frames = CONTEXT_FRAMES
    n_embeddings = _yamnet.EmbedderYamnet.n_embeddings * 2  # unshifted + real x1.26-shifted

    def initialize(self):
        # Set before super().initialize(), which warms the model up with a
        # call to self.embed() that needs these already in place.
        import librosa
        self._librosa = librosa
        self._frame_samples = int(round(self.framelength_s * self.samplerate))
        self._real_samples = int(round(RATIO * self._frame_samples))
        self._shift_sr = int(round(self.samplerate / RATIO))
        super().initialize()

    def _decimate(self, real_audio):
        """`self._real_samples` real, contiguous samples -> one
        `self._frame_samples`-sample buffer, compressed by RATIO. See module
        docstring."""
        decimated = self._librosa.resample(
            real_audio, orig_sr=self.samplerate, target_sr=self._shift_sr,
        ).astype(np.float32)
        # resample's output length can be off by a sample or two from the
        # exact target; pad or trim back to precisely one frame so the
        # concatenated multi-frame buffer stays on YAMNet's exact patch grid.
        if len(decimated) < self._frame_samples:
            decimated = np.pad(decimated, (0, self._frame_samples - len(decimated)))
        return decimated[:self._frame_samples]

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
            numpy array of embeddings, (n_frames, 2048)
        """
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio) // self._frame_samples
        if n == 0:
            return np.empty((0, self.n_embeddings), dtype=np.float32)
        usable = audio[:n * self._frame_samples]
        frames = usable.reshape(n, self._frame_samples)

        unshifted = np.asarray(self.model(usable), dtype=np.float32)
        if len(unshifted) != n:
            raise ValueError(
                f'{self.embeddername}: YAMNet returned {len(unshifted)} frames '
                f'for {n} input frames (unshifted block)'
            )

        # Real successor per row; the very last row of the buffer has none within
        # this call, so it clamps to itself. extract.py's context padding means
        # that only happens at a true edge of the recording -- everywhere else,
        # the "last row" here is a pad frame extract.py appended and trims away.
        next_frames = np.concatenate([frames[1:], frames[-1:]], axis=0)
        pairs = np.concatenate([frames, next_frames], axis=1)  # (n, 2*frame_samples)
        real_windows = pairs[:, :self._real_samples]
        shifted_frames = np.stack([self._decimate(w) for w in real_windows])
        shifted = np.asarray(self.model(shifted_frames.reshape(-1)), dtype=np.float32)
        if len(shifted) != n:
            raise ValueError(
                f'{self.embeddername}: YAMNet returned {len(shifted)} frames '
                f'for {n} input frames (shifted block)'
            )

        return np.concatenate([unshifted, shifted], axis=1)
