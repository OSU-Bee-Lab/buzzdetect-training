import importlib

import numpy as np

from embedders.embedding import BaseEmbedder

_yamnet = importlib.import_module('embedders.yamnet.embedder')

"""YAMNet, plus a real-audio up-shifted copy of each frame, concatenated.

IDEAS.md item 5b. `yamnet_pitchshift`'s x2 rung (+0.069, 8/8 folds, the era's
largest clean gain) shifts a frame up an octave by resampling its own 15360
samples down to 7680 and then TILING that back to 15360 -- so the shifted
block for frame t is really 0.48 s of real audio played twice with a hard
seam at the midpoint, not 0.96 s of anything that happened. This embedder
removes the tile: it decimates 30720 real, contiguous samples -- frame t and
its true temporal successor, frame t+1 -- down to 15360, so YAMNet is handed a
buffer that is genuinely two frames' worth of real content compressed 2x
(every frequency doubled, precisely as before) with no repeated content and no
seam.

**The row this produces is not "frame t, wider" -- it is honestly 1.92 s.**
Row spacing still matches the standard 0.96 s grid (frame counts and labels
line up 1:1 with the unshifted block and with `yamnet_pitchshift`, for a clean
comparison), but a buzz sitting entirely in the borrowed second half shapes
this row's shifted-block vector without being that row's own label -- the same
dilution concern `framelength-changes-labels` raised for a literal wider frame,
just paid on the shifted block only rather than the whole row. Say so in any
write-up; do not read this as a strict improvement on `yamnet_pitchshift` just
because the tile artifact is gone.

Context comes from `context_frames`, the same mechanism `yamnet_context` uses:
02_set/extract.py hands embed() a buffer of real contiguous audio (padded by
one true frame of audio on each side, at chunk boundaries, from the snip) and
this class reads its raw SAMPLES directly out of that buffer -- not
post-embedding vectors the way `yamnet_context` stacks. Frames at a true edge
of a recording (no real successor available) clamp: they decimate frame t
against itself, i.e. degrade gracefully to `yamnet_pitchshift`'s own tile for
that one row only.
"""

CONTEXT_FRAMES = 1  # one real frame of look-ahead is all the decimation needs


class EmbedderYamnetPitchshiftDecimate(_yamnet.EmbedderYamnet):
    embeddername = "yamnet_pitchshift_decimate"
    context_frames = CONTEXT_FRAMES
    n_embeddings = _yamnet.EmbedderYamnet.n_embeddings * 2  # unshifted + real up-shifted

    def initialize(self):
        # Set before super().initialize(), which warms the model up with a
        # call to self.embed() that needs these already in place.
        import librosa
        self._librosa = librosa
        self._frame_samples = int(round(self.framelength_s * self.samplerate))
        self._half_sr = self.samplerate // 2
        super().initialize()

    def _decimate_pair(self, pair):
        """Two contiguous 0.96 s frames (30720 real samples) -> one 15360-sample
        buffer: a genuine 1.92 s of real audio compressed 2x. See module
        docstring."""
        decimated = self._librosa.resample(
            pair, orig_sr=self.samplerate, target_sr=self._half_sr,
        ).astype(np.float32)
        # resample's output length can be off by a sample or two from the
        # exact half; pad or trim back to precisely one frame so the
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
        decimated_frames = np.stack([self._decimate_pair(p) for p in pairs])
        shifted = np.asarray(self.model(decimated_frames.reshape(-1)), dtype=np.float32)
        if len(shifted) != n:
            raise ValueError(
                f'{self.embeddername}: YAMNet returned {len(shifted)} frames '
                f'for {n} input frames (shifted block)'
            )

        return np.concatenate([unshifted, shifted], axis=1)
