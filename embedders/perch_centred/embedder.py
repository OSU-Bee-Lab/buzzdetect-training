"""Perch v2, re-centred on the era's own 0.96 s YAMNet frame grid instead of
its native 5.0 s non-overlapping frame.

IDEAS.md item 2. `perch-probe` (E3) put Perch on its native 5.0 s frame and
found the whole apparent gain tracked frame-length dilution of the labels
(`overlap_event_s` scales with framelength, so a 5 s frame needs 1.0 s of
label overlap against YAMNet's 0.192 s) -- inconclusive, not a verdict on the
embedder. `perch-broadcast-join`, an attempt to reuse Perch's existing 5 s
cache by broadcasting one row across many 0.96 s frames, was closed unfinished
and is forbidden by LOOP.md's "no artificial frame/embedding cache" rule: it
would duplicate one Perch row across every frame near it and zero-fill frames
with no nearby real row -- the same inflation mechanism as `context-stack`'s
leak.

This is the compliant version: a genuine new extraction. Each output row
still sits on the lead's 0.96 s frame grid (same frame count, same labels,
same `overlap_event_s` as `yamnet`/`yamnet_trunk`/etc.), but the audio handed
to Perch for that row is a real, contiguous 5.0 s window **centred** on the
0.96 s frame, read from the actual recording via `02_set/extract.py`'s
`context_frames` mechanism (the same contiguous-buffer machinery
`yamnet_context` and `yamnet_trunk_context` use, not a cache join). No row is
synthesized, duplicated or zero-filled from another embedder's cache; every
row is Perch's own forward pass over real audio nobody else has embedded.

`context_frames = 3` (padding = 3 x 0.96 s = 2.88 s each side) is chosen so a
full centred 5.0 s window (2.5 s either side of the 0.96 s frame, i.e. 2.02 s
of padding beyond the frame itself) is always available from real audio, with
margin, for every frame `extract.py` hands this embedder -- including the
first and last frame of a chunk, which is exactly the case the padding exists
to cover. Only a frame that sits at the true edge of a source recording
(no real audio on that side at all) falls back to zero-padding the window
there, the same case `yamnet_context`'s docstring already carries: the one
place a live deployment would have no neighbour either.

This is item 2's `1_95` falsifier: a headline gain whose delta sits inside
`1_95`'s own SD is a rich-fold gain, not the "separates the jet" result the
item is testing for.
"""
import os

import numpy as np

from embedders.embedding import BaseEmbedder

_CPU_SUBDIR = 'perch_v2_cpu'
_GPU_SUBDIR = 'perch_v2_gpu'
_TARGET_PEAK = 0.25  # perch_hoplite's zoo_interface.EmbeddingModel.normalize_audio

_WINDOW_S = 5.0  # Perch's native, fixed SavedModel input width
_BATCH_BUCKET = 16  # see embed(): windows per fixed-shape model call


class EmbedderPerchCentred(BaseEmbedder):
    embeddername = "perch_centred"
    framelength_s = 0.96          # the era's YAMNet-native frame grid, not Perch's own
    digits_time = 2
    samplerate = 32000             # Perch's native rate; avoids a second resample in embed()
    n_embeddings = 1536
    dtype_in = 'float32'
    context_frames = 3             # >= ceil((2.5 - 0.96/2) / 0.96) = 3, see module docstring

    def initialize(self):
        import tensorflow as tf  # deferred: must not import TF before fork, see embedders/perch

        # Calling this SavedModel's signature repeatedly under XLA JIT (its
        # default) retraces/recompiles per call in a way that never releases
        # the old compiled graph -- measured as unbounded host RSS growth
        # across a real extraction (10+ GB within single-digit minutes),
        # even with embed()'s fixed bucketed batch shape. Disabling JIT here
        # gives flat RSS across repeated calls (measured: ~2.7-2.9 GB steady
        # over 20+ calls) at the SAME per-window throughput perch-probe
        # already established (~0.4 s / 5 s window) -- JIT was not buying
        # real speed on this workload, only leaking memory. Must be set
        # before the model loads.
        tf.config.optimizer.set_jit(False)

        curdir = os.path.dirname(os.path.realpath(__file__))
        perch_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'perch')
        has_gpu = bool(tf.test.gpu_device_name())
        subdir = _GPU_SUBDIR if has_gpu else _CPU_SUBDIR
        model_path = os.path.join(perch_dir, subdir)

        self.frame_samples = int(round(self.framelength_s * self.samplerate))
        self.window_samples = int(round(_WINDOW_S * self.samplerate))
        self._offset_samples = (self.window_samples - self.frame_samples) // 2

        loaded = tf.saved_model.load(model_path)
        self.model = loaded.signatures['serving_default']
        self._loaded = loaded  # keep alive -- the signature doesn't own the captured resources

        self.embed(np.zeros(self.frame_samples * (2 * self.context_frames + 1), dtype=np.float32))

    def embed(self, audio):
        """A flat, contiguous buffer of whole 0.96 s frames at 32 kHz (real
        neighbouring audio either side, courtesy of extract.py's
        context_frames padding -- see module docstring). Returns one 1536-d
        Perch embedding per frame, each computed from a real 5.0 s window
        centred on that frame, read from THIS buffer (never another
        embedder's cache).
        """
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        n = len(audio) // self.frame_samples
        if n == 0:
            return np.zeros((0, self.n_embeddings), dtype=np.float32)

        # One fixed-shape model call per _BATCH_BUCKET windows. extract.py's
        # context path hands this embedder a whole annotation chunk at once
        # (BUZZDETECT_CHUNK_FRAMES never reaches it), up to a full 360 s snip:
        # ~375 windows, which Perch turned into a single 6 GB activation
        # allocation, and every distinct chunk length into a fresh retrace.
        # A constant (_BATCH_BUCKET, 160000) input keeps both bounded; the
        # last batch is zero-padded and its pad rows dropped.
        out = np.empty((n, self.n_embeddings), dtype=np.float32)
        for b0 in range(0, n, _BATCH_BUCKET):
            windows = np.zeros((_BATCH_BUCKET, self.window_samples), dtype=np.float32)
            for j, i in enumerate(range(b0, min(b0 + _BATCH_BUCKET, n))):
                lo = i * self.frame_samples - self._offset_samples
                hi = lo + self.window_samples
                src_lo, src_hi = max(lo, 0), min(hi, len(audio))
                if src_hi > src_lo:
                    windows[j, src_lo - lo:src_hi - lo] = audio[src_lo:src_hi]

            # Perch's normalize_audio: DC-remove, then scale each window's peak to _TARGET_PEAK.
            windows = windows - windows.mean(axis=-1, keepdims=True)
            peak = np.max(np.abs(windows), axis=-1, keepdims=True)
            windows = np.divide(windows, peak, out=np.zeros_like(windows), where=(peak > 0.0)) * _TARGET_PEAK

            m = min(_BATCH_BUCKET, n - b0)
            out[b0:b0 + m] = self.model(inputs=windows)['embedding'].numpy()[:m]
        return out
