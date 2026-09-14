import importlib

import numpy as np

import config as cfg
from embedders.embedding import BaseEmbedder  # noqa: F401  -- load_embedder's scan

_pitchshift = importlib.import_module('embedders.yamnet_pitchshift.embedder')

"""IDEAS.md item 16: stack context onto the pitch-shift block.

This era's two best-confirmed levers, composed the way `yamnet_context_aves`
composed context + AVES: widen only ONE named half with temporal context and
leave the other as-is, so the run tests one new combination, not two new
levers at once.

`yamnet_pitchshift` (+0.069, 8/8 folds, twice confirmed) is
`[unshifted(t), shifted_up_octave(t)]`, 2048-d, both halves reading only frame
t's own audio. `yamnet_context` (+0.030, honest) widens plain YAMNet with its
real temporal neighbours. Widening the UNSHIFTED half here (not the shifted
one) mirrors `yamnet_context_aves`'s choice to widen the plain-YAMNet block
and leave AVES's already-temporally-integrated block alone: the pitch-shift
transform is a per-frame audio operation, not obviously suited to being
computed on someone else's neighbour, whereas the unshifted half is exactly
`yamnet_context`'s own input.

Per frame: `[yam(t-1), yam(t), yam(t+1), shifted(t)]` = 4096-d.

Context comes from `context_frames`, the same mechanism `yamnet_context` and
`yamnet_pitchshift_decimate` use: 02_set/extract.py hands embed() a buffer of
real contiguous audio, padded by one true frame on each side at chunk
boundaries, and discards the pad rows after.
"""

CONTEXT_FRAMES = 1
_YAMNET_DIMS = 1024


class EmbedderYamnetPitchshiftContext(_pitchshift.EmbedderYamnetPitchshift):
    embeddername = "yamnet_pitchshift_context"
    context_frames = CONTEXT_FRAMES
    n_embeddings = _YAMNET_DIMS * (2 * CONTEXT_FRAMES + 1) + _YAMNET_DIMS  # widened unshifted + shifted as-is

    def embed_frames(self, audio):
        """Plain per-frame [unshifted | shifted] for a contiguous buffer, (n, 2048).

        Chunked at cfg.CHUNK_FRAMES for the same memory reason 02_set chunks,
        but on frame boundaries and here rather than in the caller -- the
        caller must not split the buffer itself or frames at each split lose
        their neighbours.
        """
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio) // self._frame_samples
        if n == 0:
            return np.empty((0, _pitchshift.EmbedderYamnetPitchshift.n_embeddings), dtype=np.float32)

        out = []
        for lo in range(0, n, cfg.CHUNK_FRAMES):
            hi = min(lo + cfg.CHUNK_FRAMES, n)
            piece = audio[lo * self._frame_samples: hi * self._frame_samples]
            out.append(super().embed(piece))
        return np.concatenate(out, axis=0)

    def stack_context(self, embeddings):
        """(n, 2048) -> (n, 4096): unshifted block widened, shifted block as-is."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        k = self.context_frames
        n = len(embeddings)
        if n == 0:
            return np.zeros((0, self.n_embeddings), dtype=np.float32)
        unshifted, shifted = embeddings[:, :_YAMNET_DIMS], embeddings[:, _YAMNET_DIMS:]
        if k == 0:
            return np.concatenate([unshifted, shifted], axis=1)
        idx = np.arange(n)
        blocks = [unshifted[np.clip(idx + offset, 0, n - 1)] for offset in range(-k, k + 1)]
        return np.concatenate(blocks + [shifted], axis=1)

    def embed(self, audio):
        """Contiguous audio in, one context-widened embedding per frame out."""
        return self.stack_context(self.embed_frames(audio))
