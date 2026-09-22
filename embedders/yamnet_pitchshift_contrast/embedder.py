import importlib

import numpy as np

# Through the module, so load_embedder()'s scan finds only this class.
_ps = importlib.import_module('embedders.yamnet_pitchshift.embedder')

"""YAMNet, plus the signed contrast against its own octave-shifted twin.

New proposal, not from IDEAS.md's queue -- combines two confirmed levers this
era: `yamnet-pitchshift` (concat `[e_t, e_shifted(t)]`, +0.069, this era's
largest single-lever gain) and `asym-context-yamnet` (a linear readout of a
raw concat cannot compute a contrast; computing it explicitly --
`[e_t, e_t - mean(neighbours)]` -- beat plain concatenation, +0.046 vs
context-frames-fix's +0.030 at the same budget).

This applies the SAME mechanism along the pitch axis instead of the time
axis: `[e_t, e_t - e_shifted(t)]`, 2048-d. There is exactly one "neighbour"
(the octave-up embedding of the identical frame, not an adjacent frame in
time), so this does not touch LOOP.md's context/averaging prior -- no
temporal neighbour is read, and no cached row from another frame is involved.
The two towers still see independent audio: e_shifted(t) is re-embedded from
resampled audio for frame t, exactly as `yamnet_pitchshift` computes it; only
the assembly changes from concat to concat-plus-contrast.

Frame grid, samplerate and pitch-shift mechanism are all inherited unchanged
from `yamnet_pitchshift` -- see that module's docstring for why the shift is
safe (holds the 0.96 s patch grid fixed; resample+tile, not a formant-
preserving shifter).
"""


class EmbedderYamnetPitchshiftContrast(_ps.EmbedderYamnetPitchshift):
    embeddername = "yamnet_pitchshift_contrast"
    n_embeddings = 2048  # 1024 unshifted + 1024 signed contrast (unshifted - shifted)

    def embed(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio) // self._frame_samples
        if n == 0:
            return np.empty((0, self.n_embeddings), dtype=np.float32)
        usable = audio[:n * self._frame_samples]

        unshifted = np.asarray(self.model(usable), dtype=np.float32)
        if len(unshifted) != n:
            raise ValueError(
                f'{self.embeddername}: YAMNet returned {len(unshifted)} frames '
                f'for {n} input frames'
            )

        frames = usable.reshape(n, self._frame_samples)
        shifted_frames = np.stack([self._pitch_up_octave(f) for f in frames])
        shifted = np.asarray(self.model(shifted_frames.reshape(-1)), dtype=np.float32)

        return np.concatenate([unshifted, unshifted - shifted], axis=1)
