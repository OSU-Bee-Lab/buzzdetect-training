"""Plain plus seam-free octave-up and two-octave-up views (ps12sf-d8).

Both shifted views are built from real, contiguous audio centred on the frame
(1.92 s and 3.84 s), resampled 2:1 once and twice: the octave-up view of
`yamnet_trunk_pitchshiftsf_depth8` beside the third view of
`yamnet_trunk_pitchshift2sf_depth8`, with no tiled view left. Context comes
from `context_frames`, so it reads neighbouring audio, never cached rows.

No ONNX form: a research embedder.
"""
import importlib

import numpy as np

_ps8 = importlib.import_module('embedders.yamnet_trunk_pitchshift_depth8.embedder')
_sf = importlib.import_module('embedders.yamnet_trunk_pitchshiftsf_depth8.embedder')


class EmbedderYamnetTrunkPitchshift12sfDepth8(_ps8.EmbedderYamnetTrunkPitchshiftDepth8):
    embeddername = "yamnet_trunk_pitchshift12sf_depth8"
    context_frames = 2   # covers the 1.5 frames of real audio either side
    n_ctx = 3  # [plain, octave-up sf, two-octaves-up sf], shared tail
    n_embeddings = _ps8.EmbedderYamnetTrunkPitchshiftDepth8.n_embeddings // 2 * n_ctx

    def embed(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio) // self.recipe.ctx(self).frame
        if n == 0:
            return np.zeros((0, self.n_embeddings), dtype=self.recipe.dtype)
        width = self.n_embeddings // self.n_ctx
        plain = self.recipe.embed_frames(self, audio)[:, :width]
        up = _sf.seamfree_view(self, audio, n, span=2)
        up2 = _sf.seamfree_view(self, audio, n, span=4)
        out = np.concatenate([plain, up, up2], axis=1).astype(self.recipe.dtype)
        if out.shape[1] != self.n_embeddings:
            raise ValueError(f'{self.embeddername}: {out.shape[1]} columns, '
                             f'n_embeddings says {self.n_embeddings}')
        return out

    def to_onnx(self, opset=17):
        raise NotImplementedError(f'{self.embeddername} is a research embedder with no ONNX form')
