import importlib

import numpy as np

import config as cfg

# Imported through the module rather than `from ... import EmbedderYamnet`, so that
# load_embedder()'s scan for a BaseEmbedder subclass in this module's namespace can
# only find EmbedderYamnetContextK2. Two candidates and it picks whichever sorts first.
_yamnet = importlib.import_module('embedders.yamnet.embedder')

# Frames of real audio either side of the target frame folded into its embedding.
# k=2 dose-response step from exp/context-embedder's k=1 ([t-1,t,t+1], 3072-d):
# widen to [t-2..t+2], 5 frames, 5120-d. See exp/context-width's notes.md.
CONTEXT_FRAMES = 2


class EmbedderYamnetContextK2(_yamnet.EmbedderYamnet):
    """YAMNet, with each frame's embedding widened by its temporal neighbours.

    Identical mechanism to exp/context-embedder's EmbedderYamnetContext, just a
    wider CONTEXT_FRAMES (2 instead of 1) — this class exists only to test the
    dose-response of context width, not to change the approach.

    embed() takes a buffer of *contiguous* audio, embeds every frame in it with
    YAMNet, and returns concat(frame t-2, t-1, t, t+1, t+2) per frame — so the
    context a frame carries is the audio that actually sat either side of it.
    Frames at the two ends of a buffer have fewer real neighbours on one side and
    repeat themselves (edge clamping); that is the same thing an inference buffer's
    edge does, and it costs 2*k frames per buffer.

    Because the context lives in the embedder, the shipped model.py works
    unchanged: predict() calls embed() and gets n_embeddings-wide vectors.

    Extraction must hand embed() contiguous audio for any of this to be true.
    02_set/extract.py checks `context_frames` and does exactly that; embedding a
    buffer of frames that were first grouped by label would make each frame's
    "neighbours" other frames with its own label, which is the artifact this
    embedder exists to avoid.
    """

    embeddername = "yamnet_context_k2"
    context_frames = CONTEXT_FRAMES
    n_embeddings = _yamnet.EmbedderYamnet.n_embeddings * (2 * CONTEXT_FRAMES + 1)

    def n_frames(self, n_samples):
        """Frames YAMNet will emit for a buffer of n_samples, at this framehop."""
        length = int(round(self.framelength_s * self.samplerate))
        hop = int(round(self.framehop_s * self.samplerate))
        if n_samples < length:
            return 0
        return (n_samples - length) // hop + 1

    def embed_frames(self, audio):
        """Plain per-frame YAMNet embeddings for a contiguous buffer, (n, 1024).

        Chunked at cfg.CHUNK_FRAMES for the same memory reason 02_set chunks, but
        on frame boundaries and here rather than in the caller — the caller must
        not split the buffer itself or frames at each split lose their neighbours.
        """
        audio = np.asarray(audio, dtype=np.float32)
        length = int(round(self.framelength_s * self.samplerate))
        hop = int(round(self.framehop_s * self.samplerate))
        n = self.n_frames(len(audio))
        if n == 0:
            return np.zeros((0, _yamnet.EmbedderYamnet.n_embeddings), dtype=np.float32)

        out = []
        for lo in range(0, n, cfg.CHUNK_FRAMES):
            hi = min(lo + cfg.CHUNK_FRAMES, n)
            piece = audio[lo * hop: (hi - 1) * hop + length]
            embedded = np.asarray(self.model(piece), dtype=np.float32)
            if len(embedded) != hi - lo:
                raise ValueError(
                    f'{self.embeddername}: YAMNet returned {len(embedded)} frames for '
                    f'{len(piece)} samples, expected {hi - lo}'
                )
            out.append(embedded)
        return np.concatenate(out, axis=0)

    def stack_context(self, embeddings):
        """(n, 1024) -> (n, 1024 * (2k+1)), frames in time order, edges clamped."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        k = self.context_frames
        if k == 0:
            return embeddings
        n = len(embeddings)
        if n == 0:
            return np.zeros((0, self.n_embeddings), dtype=np.float32)
        idx = np.arange(n)
        return np.concatenate(
            [embeddings[np.clip(idx + offset, 0, n - 1)] for offset in range(-k, k + 1)],
            axis=1,
        )

    def embed(self, audio):
        """Contiguous audio in, one context-widened embedding per frame out."""
        return self.stack_context(self.embed_frames(audio))
