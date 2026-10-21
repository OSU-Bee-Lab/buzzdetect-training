"""yamnet_trunk_pitchshift_depth8 plus a seam-free two-octave-up third view
(ps2-seamfree-d8, IDEAS #36).

`yamnet_trunk_pitchshift2_depth8` built its two-octave view by resampling the
0.96 s frame 4:1 and tiling it 4x: four seams and only 0.24 s of audio. Here
the third view's source is 3.84 s of real, contiguous audio centred on the
frame (1.5 frames either side), resampled 2:1 twice down to one frame. It
comes from `context_frames` (02_set/extract.py's `_embed_with_context`), so it
reads neighbouring audio, never neighbouring cached rows. Where the recording
ends inside the window, the window is zero-padded, never tiled.

The first two views are the parent recipe's (plain, octave-up) over the same
buffer. The buffer now carries pad frames, so chunk-edge frames can differ
from the parent's cache in the last bits (CLAUDE.md, BUZZDETECT_CHUNK_FRAMES).

No ONNX form: a research embedder. Shipping it needs a windowed FrameOp in
embedders/recipe.py first.
"""
import importlib

import numpy as np

from embedders.recipe import Fit, Keras, Resample

_ps8 = importlib.import_module('embedders.yamnet_trunk_pitchshift_depth8.embedder')

CONTEXT_FRAMES = 2   # covers the 1.5 frames of real audio either side
SPAN = 4             # source window, in frames; two 2:1 resamples bring it to one
UP_TWO_OCTAVES_SEAMFREE = (Resample('half'), Resample('half'), Fit())


class EmbedderYamnetTrunkPitchshift2sfDepth8(_ps8.EmbedderYamnetTrunkPitchshiftDepth8):
    embeddername = "yamnet_trunk_pitchshift2sf_depth8"
    context_frames = CONTEXT_FRAMES
    n_ctx = 3  # [plain, octave-up, two-octaves-up seam-free], shared tail
    n_embeddings = _ps8.EmbedderYamnetTrunkPitchshiftDepth8.n_embeddings // 2 * n_ctx

    def embed(self, audio):
        c = self.recipe.ctx(self)
        L = c.frame
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio) // L
        if n == 0:
            return np.zeros((0, self.n_embeddings), dtype=self.recipe.dtype)
        two = self.recipe.embed_frames(self, audio)  # (n, 2 x 12288) float32

        # row t's window: [t L - 1.5 L, t L + 2.5 L), zero outside the buffer
        lo = (SPAN * L - L) // 2
        padded = np.concatenate([np.zeros(lo, np.float32), audio[:n * L],
                                 np.zeros(SPAN * L, np.float32)])
        windows = np.stack([padded[t * L: t * L + SPAN * L] for t in range(n)])
        x = windows
        for op in UP_TWO_OCTAVES_SEAMFREE:
            x = op.numpy(x, c)
        third = Keras().numpy(self, x)

        out = np.concatenate([two, third], axis=1).astype(self.recipe.dtype)
        if out.shape[1] != self.n_embeddings:
            raise ValueError(f'{self.embeddername}: {out.shape[1]} columns, '
                             f'n_embeddings says {self.n_embeddings}')
        return out

    def to_onnx(self, opset=17):
        raise NotImplementedError(f'{self.embeddername} is a research embedder with no ONNX form')
