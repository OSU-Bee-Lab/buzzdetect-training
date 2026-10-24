"""yamnet_trunk_pitchshift_depth8 with its octave-up view made seam-free
(pssf-d8).

ps-depth8's octave-up view resamples the 0.96 s frame 2:1 and tiles it 2x:
one seam mid-frame and only 0.96 s of audio. ps2-seamfree-d8 showed that a
two-octave view built from real, contiguous neighbouring audio instead of
tiles took 1_114 from 0.41 to 0.67. Here the octave-up view itself gets the
same treatment: its source is 1.92 s of real audio centred on the frame
(half a frame either side), resampled 2:1 to one frame. It comes from
`context_frames` (02_set/extract.py's `_embed_with_context`), so it reads
neighbouring audio, never neighbouring cached rows. Where the recording ends
inside the window, the window is zero-padded, never tiled.

The plain view is the parent recipe's, over the same buffer.

No ONNX form: a research embedder.
"""
import importlib

import numpy as np

from embedders.recipe import Fit, Keras, Resample, _chunk_frames

_ps8 = importlib.import_module('embedders.yamnet_trunk_pitchshift_depth8.embedder')


def seamfree_view(e, audio, n, span):
    """One frame-length view per row from `span` frames of real audio centred on
    the row, resampled 2:1 log2(span) times. Zero outside the buffer."""
    c = e.recipe.ctx(e)
    L = c.frame
    lo = (span * L - L) // 2
    padded = np.concatenate([np.zeros(lo, np.float32), audio[:n * L],
                             np.zeros(span * L, np.float32)])
    ops = (Resample('half'),) * int(np.log2(span)) + (Fit(),)
    step = _chunk_frames()
    out = []
    for a in range(0, n, step):
        x = np.stack([padded[t * L: t * L + span * L] for t in range(a, min(n, a + step))])
        for op in ops:
            x = op.numpy(x, c)
        out.append(Keras().numpy(e, x))
    return np.concatenate(out, axis=0)


class EmbedderYamnetTrunkPitchshiftsfDepth8(_ps8.EmbedderYamnetTrunkPitchshiftDepth8):
    embeddername = "yamnet_trunk_pitchshiftsf_depth8"
    context_frames = 1   # covers the half frame of real audio either side
    n_ctx = 2  # [plain, octave-up seam-free], shared tail
    n_embeddings = _ps8.EmbedderYamnetTrunkPitchshiftDepth8.n_embeddings

    def embed(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio) // self.recipe.ctx(self).frame
        if n == 0:
            return np.zeros((0, self.n_embeddings), dtype=self.recipe.dtype)
        width = self.n_embeddings // self.n_ctx
        plain = self.recipe.embed_frames(self, audio)[:, :width]
        up = seamfree_view(self, audio, n, span=2)
        out = np.concatenate([plain, up], axis=1).astype(self.recipe.dtype)
        if out.shape[1] != self.n_embeddings:
            raise ValueError(f'{self.embeddername}: {out.shape[1]} columns, '
                             f'n_embeddings says {self.n_embeddings}')
        return out

    def to_onnx(self, opset=17):
        raise NotImplementedError(f'{self.embeddername} is a research embedder with no ONNX form')
