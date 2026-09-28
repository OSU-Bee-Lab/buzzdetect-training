"""Embedder recipes: one description of how an embedder turns audio into
embeddings, which drives both its numpy `embed()` (extraction, training) and
its ONNX graph (`to_onnx()`, what buzzdetect runs).

Why this exists. buzzdetect runs a single ONNX graph, waveform in, embeddings
out. Extraction runs the embedder's own Python. Before recipes, every embedder
that did more than "call one Keras model" carried a second, hand-built ONNX
copy of its embed(), and the two could drift apart (several did). Now an
embedder declares *what* it does as data, and this module owns *how* each step
is done, once, in both forms side by side:

    class EmbedderYamnetTrunkPitchshiftDepth12(RecipeEmbedder, EmbedderYamnetTrunkDepth12):
        recipe = Recipe(
            branches=(Branch(Keras('model')),
                      Branch(Keras('model'), UP_OCTAVE)),
            dtype='float16')

Reading a recipe, top to bottom, is reading the pipeline:

  1. `prefilter`: an FIR over the whole waveform, before framing (bandpass).
  2. framing: the waveform is cut into whole `framelength_s` frames and a
     ragged tail is dropped (framing='native' instead hands the waveform to a
     single model that frames it itself, the way plain YAMNet does).
  3. `branches`: each branch takes every frame through its `ops` (slice,
     resample, tile...) and then through its `model`, giving (frames, d).
     Every branch sees the same frames; each branch's frames go through the
     model in one call of their own, so YAMNet's STFT never looks across from
     one branch's audio into another's.
  4. `combine`: branch outputs side by side ('concat'), or [b0, b0 - b1]
     ('contrast').
  5. `context`: each frame widened by its neighbours, t-k..t+k, edges clamped
     (the first `context.branches` branches' columns; the rest appended as-is).
  6. `dtype`: the stored precision.

Adding a new kind of step means adding one class here with both a `numpy()`
and an `onnx()` method. An embedder assembled from existing steps needs no
export code at all, and 04_deploy's parity check (embed() against the graph)
tests the pair on every export.

Invariants the numpy side keeps, because cached embeddings on disk were made
by the code this replaced and must stay bit-identical to what extraction
produces now: per-frame loops where the old code looped per frame (librosa
resample), YAMNet called on one concatenated buffer per branch, AVES batched
from the start of each call, context embedders chunked at cfg.CHUNK_FRAMES.
"""
import os
from dataclasses import dataclass

import numpy as np

from embedders.embedding import BaseEmbedder


# ============================================================================
# ONNX graph builder
# ============================================================================

class Graph:
    """Accumulates nodes in construction order, which is topological order
    as long as every node is added after the nodes that feed it."""

    def __init__(self, opset):
        self.opset = opset
        self.nodes, self.inits, self.value_info = [], [], []
        self.opsets = {'': opset}
        self.ir_version = 8
        self._n = 0

    def name(self, hint):
        self._n += 1
        return f'r{self._n}_{hint}'

    def const(self, values, hint='c', dtype='int64'):
        from onnx import numpy_helper
        name = self.name(hint)
        self.inits.append(numpy_helper.from_array(np.asarray(values, dtype=dtype), name))
        return name

    def op(self, op_type, inputs, hint=None, **attrs):
        from onnx import helper
        out = self.name(hint or op_type.lower())
        self.nodes.append(helper.make_node(op_type, list(inputs), [out], name=out, **attrs))
        return out

    def embed_model(self, proto, input_name, prefix):
        """Splice a whole sub-model in, reading `input_name`; returns its output."""
        from onnx import compose, helper
        proto = compose.add_prefix(proto, prefix=prefix)
        g = proto.graph
        self.nodes.append(helper.make_node('Identity', [input_name], [g.input[0].name],
                                           name=f'{prefix}in'))
        self.nodes.extend(g.node)
        self.inits.extend(g.initializer)
        self.value_info.extend(g.value_info)
        for imp in proto.opset_import:
            self.opsets[imp.domain] = max(self.opsets.get(imp.domain, 0), imp.version)
        self.ir_version = max(self.ir_version, proto.ir_version)
        return g.output[0].name

    def model(self, input_name, output_name, graph_name):
        from onnx import TensorProto, helper
        graph = helper.make_graph(
            self.nodes, graph_name,
            [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, ['samples'])],
            [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, None)],
            initializer=self.inits, value_info=self.value_info)
        return helper.make_model(
            graph, producer_name='buzzdetect-training',
            opset_imports=[helper.make_opsetid(d, v) for d, v in self.opsets.items()],
            ir_version=self.ir_version)


# ============================================================================
# Per-frame ops: (n, L) frames -> (n, L') frames.
# In ONNX, frames are (n, 1, L), so Conv applies directly.
# ============================================================================

@dataclass(frozen=True)
class Ctx:
    sr: int     # sample rate
    frame: int  # the recipe's frame length, samples


class FrameOp:
    def length(self, L, c):
        """Output length for an input frame of L samples."""
        raise NotImplementedError

    def numpy(self, frames, c):
        raise NotImplementedError

    def onnx(self, g, x, L, c):
        raise NotImplementedError(
            f'{type(self).__name__} has no ONNX form yet: add an onnx() to it in '
            f'embedders/recipe.py to export an embedder that uses it')


@dataclass(frozen=True)
class Slice(FrameOp):
    """Samples [start, stop) of each frame; floats are fractions of the frame."""
    start: float
    stop: float

    def bounds(self, L):
        f = lambda v: int(round(v * L)) if isinstance(v, float) else int(v)
        return f(self.start), f(self.stop)

    def length(self, L, c):
        a, b = self.bounds(L)
        return b - a

    def numpy(self, frames, c):
        a, b = self.bounds(frames.shape[1])
        return frames[:, a:b]

    def onnx(self, g, x, L, c):
        a, b = self.bounds(L)
        return g.op('Slice', [x, g.const([a]), g.const([b]), g.const([2])], 'slice')


@dataclass(frozen=True)
class Resample(FrameOp):
    """librosa.resample by 2:1, relabelled at the original rate.

    to='half' (sr -> sr/2) plays back an octave up, at half the length;
    to='double' (sr -> 2 sr) an octave down, at twice the length.

    In ONNX: librosa's resampler is linear and, at a 2:1 ratio, time-invariant
    up to the stride with zero padding at the edges, so it is one fixed FIR,
    read off librosa itself by resampling two impulses. 'half' is a stride-2
    Conv; 'double' is zero-stuffing then a Conv. Agrees with librosa to float32
    rounding (~1e-6).
    """
    to: str  # 'half' | 'double'

    TRIM = 1e-7  # taps below this fraction of the peak are dropped from the ONNX FIR

    def target(self, sr):
        return sr // 2 if self.to == 'half' else sr * 2

    def length(self, L, c):
        return L // 2 if self.to == 'half' else L * 2

    def numpy(self, frames, c):
        import librosa
        # one frame at a time, as the code this replaced did
        return np.stack([
            librosa.resample(f, orig_sr=c.sr, target_sr=self.target(c.sr)).astype(np.float32)
            for f in frames])

    def kernel(self, L, sr):
        """(Conv weight, pad_left, pad_right) reproducing librosa on L samples."""
        import librosa
        taps = {}
        for k0 in (L // 2, L // 2 + 1):
            impulse = np.zeros(L, dtype=np.float32)
            impulse[k0] = 1
            out = librosa.resample(impulse, orig_sr=sr, target_sr=self.target(sr))
            for m in np.nonzero(out)[0]:
                # 'half': y[m] = sum_k h(2m - k) x[k];  'double': y[m] = sum_k h(m - 2k) x[k]
                d = 2 * int(m) - k0 if self.to == 'half' else int(m) - 2 * k0
                taps[d] = out[m]
        # soxr's impulse response is ~1660 taps, but all outside the central ~380
        # sum to ~3e-6 (L1): drop them. Kept whole, this one Conv cost more than
        # both YAMNets together in the exported graph.
        peak = max(abs(v) for v in taps.values())
        kept = [d for d, v in taps.items() if abs(v) > self.TRIM * peak]
        lo, hi = min(kept), max(kept)
        taps = {d: v for d, v in taps.items() if lo <= d <= hi}
        w = np.zeros(hi - lo + 1, dtype=np.float32)
        for d, v in taps.items():
            w[hi - d] = v  # Conv is cross-correlation: w[t] = h(pad_left - t)
        return w, hi, len(w) - 1 - hi

    def onnx(self, g, x, L, c):
        w, pad_l, pad_r = self.kernel(L, c.sr)
        kernel = g.const(w.reshape(1, 1, -1), 'fir', dtype='float32')
        conv = dict(pads=[pad_l, pad_r], kernel_shape=[len(w)])
        if self.to == 'half':
            return g.op('Conv', [x, kernel], 'resample_half', strides=[2], **conv)
        # zero-stuff: x[k] -> u[2k], u[2k + 1] = 0
        col = g.op('Unsqueeze', [x, g.const([3])])
        zeros = g.op('Sub', [col, col])
        pairs = g.op('Concat', [col, zeros], axis=3)
        stuffed = g.op('Reshape', [pairs, g.const([-1, 1, 2 * L])])
        return g.op('Conv', [stuffed, kernel], 'resample_double', strides=[1], **conv)


@dataclass(frozen=True)
class Tile(FrameOp):
    """The frame repeated end to end, `reps` times."""
    reps: int

    def length(self, L, c):
        return L * self.reps

    def numpy(self, frames, c):
        return np.tile(frames, (1, self.reps))

    def onnx(self, g, x, L, c):
        return g.op('Concat', [x] * self.reps, 'tile', axis=2)


@dataclass(frozen=True)
class Fit(FrameOp):
    """Zero-pad or crop to exactly `samples` (default: the recipe's frame)."""
    samples: int = None

    def n(self, c):
        return c.frame if self.samples is None else self.samples

    def length(self, L, c):
        return self.n(c)

    def numpy(self, frames, c):
        n = self.n(c)
        if frames.shape[1] < n:
            frames = np.pad(frames, ((0, 0), (0, n - frames.shape[1])))
        return frames[:, :n].astype(np.float32)

    def onnx(self, g, x, L, c):
        n = self.n(c)
        if L < n:
            x = g.op('Pad', [x, g.const([0, 0, 0, 0, 0, n - L])], 'fit_pad')
        elif L > n:
            x = g.op('Slice', [x, g.const([0]), g.const([n]), g.const([2])], 'fit_crop')
        return x


@dataclass(frozen=True)
class Vocoder(FrameOp):
    """librosa.effects.pitch_shift (phase vocoder), duration kept. No ONNX form."""
    n_steps: float

    def length(self, L, c):
        return L

    def numpy(self, frames, c):
        import librosa
        # batched over frames, as the code this replaced did
        return librosa.effects.pitch_shift(frames, sr=c.sr, n_steps=self.n_steps).astype(np.float32)


# The named views the pitch-shift embedders are built from.
UP_OCTAVE = (Resample('half'), Tile(2), Fit())                       # whole frame, tiled to fill
DOWN_OCTAVE_CENTRE = (Slice(0.25, 0.75), Resample('double'), Fit())  # centre half, stretched
DOWN_OCTAVE_FIRST = (Slice(0.0, 0.5), Resample('double'), Fit())     # first half
DOWN_OCTAVE_SECOND = (Slice(0.5, 1.0), Resample('double'), Fit())    # second half
VOCODER_UP = (Vocoder(12),)
VOCODER_DOWN = (Vocoder(-12),)


# ============================================================================
# Models: (n, L) frames -> (n, d) embeddings
# ============================================================================

@dataclass(frozen=True)
class Keras:
    """A Keras model held on the embedder as `attr` that takes a flat waveform
    and frames it itself (YAMNet and its trunk cuts). A branch's frames go in
    laid end to end, as one call; the model returns `patches` rows per frame
    (3 for yamnet_triple's 2.88 s frames), flattened to one row per frame."""
    attr: str = 'model'
    patches: int = 1

    def width(self, e):
        shape = getattr(e, self.attr).output.shape[1:]
        return self.patches * int(np.prod(shape))

    def numpy(self, e, frames):
        n = len(frames)
        out = np.asarray(getattr(e, self.attr)(frames.reshape(-1)), dtype=np.float32)
        if len(out) != n * self.patches:
            raise ValueError(f'{e.embeddername}: {self.attr} returned {len(out)} rows '
                             f'for {n} frames, expected {n * self.patches}')
        return out.reshape(n, -1)

    def export(self, e, opset):
        import tempfile

        import onnx
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'model.onnx')
            # Keras refuses to export a model it has never seen called;
            # every embedder's initialize() calls it once.
            getattr(e, self.attr).export(path, format='onnx', verbose=False, opset_version=opset)
            return onnx.load(path)

    def onnx(self, g, e, x, proto, prefix):
        flat = g.op('Reshape', [x, g.const([-1])], 'flat')
        out = g.embed_model(proto, flat, prefix)
        return g.op('Reshape', [out, g.const([-1, self.width(e)])], 'rows')


def _aves_pool(pool, layers, layer_outputs):
    """AVES token features -> one vector per clip. Written once, used by both
    the numpy path and the exported torch module, so they cannot disagree.
    pool='mean': token mean of each layer in `layers`, side by side.
    pool='stats': [mean, max, std] over the tokens of the last layer."""
    import torch
    if pool == 'mean':
        return torch.cat([layer_outputs[i].mean(dim=1) for i in layers], dim=1)
    if pool == 'stats':
        tokens = layer_outputs[-1]
        return torch.cat([tokens.mean(dim=1), tokens.amax(dim=1), tokens.std(dim=1)], dim=1)
    raise ValueError(f'unknown AVES pool {pool!r}')


@dataclass(frozen=True)
class Aves:
    """AVES (wav2vec2, PyTorch) held on the embedder as `e.aves`, one clip per
    frame, pooled over its tokens (see _aves_pool). numpy runs in batches of
    e.batch_size on e.device from the start of each call."""
    pool: str = 'mean'
    layers: tuple = (-1,)  # 'mean' only; 0-indexed, -1 = last

    def width(self, e):
        return 768 * (len(self.layers) if self.pool == 'mean' else 3)

    def numpy(self, e, frames):
        import torch
        out = []
        with torch.no_grad():
            for start in range(0, len(frames), e.batch_size):
                wav = torch.from_numpy(np.ascontiguousarray(frames[start:start + e.batch_size])).to(e.device)
                layer_outputs, _ = e.aves.extract_features(wav)
                out.append(_aves_pool(self.pool, self.layers, layer_outputs).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def export(self, e, opset):
        import tempfile

        import onnx
        import torch
        pool, layers = self.pool, self.layers

        class Pooled(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, waveform):
                layer_outputs, _ = self.model.extract_features(waveform)
                return _aves_pool(pool, layers, layer_outputs)

        # trace on CPU, then put AVES back where embed() runs it
        wrapper = Pooled(e.aves.cpu()).eval()
        try:
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, 'aves.onnx')
                with torch.no_grad():
                    torch.onnx.export(
                        wrapper, torch.zeros(2, e.samplerate), path,
                        input_names=['frames'], output_names=['embedding'],
                        dynamic_axes={'frames': {0: 'batch'}, 'embedding': {0: 'batch'}},
                        opset_version=opset, dynamo=False)
                return onnx.load(path)
        finally:
            e.aves.to(e.device)

    def onnx(self, g, e, x, proto, prefix):
        rows = g.op('Squeeze', [x, g.const([1])], 'clips')  # (n, 1, L) -> (n, L)
        return g.embed_model(proto, rows, prefix)


# ============================================================================
# Recipe
# ============================================================================

@dataclass(frozen=True)
class Branch:
    model: object          # Keras(...) | Aves(...)
    ops: tuple = ()        # FrameOps applied to every frame first


@dataclass(frozen=True)
class Context:
    """Widen each frame with frames t-k..t+k (edges clamped). Only the first
    `branches` blocks are widened (None: all); the rest are appended as-is."""
    k: int = 1
    branches: int = None


@dataclass(frozen=True)
class FIR:
    """A 'SAME'-padded FIR over the whole waveform, before framing. The taps
    are the embedder's `attr` (a tf constant, shaped (taps, 1, 1))."""
    attr: str

    def numpy(self, e, audio):
        import tensorflow as tf
        x = tf.cast(audio, tf.float32)[tf.newaxis, :, tf.newaxis]
        y = tf.nn.conv1d(x, getattr(e, self.attr), stride=1, padding='SAME')
        return tf.squeeze(y, axis=[0, 2])

    def onnx(self, g, e, x):
        taps = np.asarray(getattr(e, self.attr)).reshape(-1).astype(np.float32)
        if len(taps) % 2 == 0:
            raise NotImplementedError('even-length FIR: TF SAME padding is asymmetric')
        half = len(taps) // 2
        # Conv refuses a zero-length signal; one sample of silence instead
        # gives what numpy gives there (YAMNet pads either to a frame of zeros)
        short = g.op('Max', [g.op('Sub', [g.const([1]), g.op('Shape', [x])]), g.const([0])])
        x = g.op('Pad', [x, g.op('Concat', [g.const([0]), short], axis=0)], 'prefilter_pad')
        x = g.op('Reshape', [x, g.const([1, 1, -1])])
        x = g.op('Conv', [x, g.const(taps.reshape(1, 1, -1), 'fir', dtype='float32')],
                 'prefilter', pads=[half, half], strides=[1], kernel_shape=[len(taps)])
        return g.op('Reshape', [x, g.const([-1])])


@dataclass(frozen=True)
class Recipe:
    branches: tuple
    combine: str = 'concat'     # 'concat' | 'contrast' ([b0, b0 - b1])
    context: Context = None
    chunk: bool = False         # numpy: run frames through in cfg.CHUNK_FRAMES blocks
    prefilter: FIR = None
    framing: str = 'whole'      # 'whole' frames, ragged tail dropped | 'native'
    dtype: str = 'float32'

    # ---- numpy ---------------------------------------------------------------

    def ctx(self, e):
        return Ctx(e.samplerate, int(round(e.framelength_s * e.samplerate)))

    def branch_numpy(self, e, frames, c):
        blocks = []
        for b in self.branches:
            x = frames
            for op in b.ops:
                x = op.numpy(x, c)
            blocks.append(b.model.numpy(e, x))
        if self.combine == 'contrast':
            blocks = [blocks[0], blocks[0] - blocks[1]]
        return blocks

    def widths(self, e):
        w = [b.model.width(e) for b in self.branches]
        return [w[0], w[0]] if self.combine == 'contrast' else w

    def embed_frames(self, e, audio):
        """Per-frame embeddings before any context, (n, sum of widths), float32."""
        c = self.ctx(e)
        audio = np.asarray(audio, dtype=np.float32)
        if self.prefilter is not None:
            audio = self.prefilter.numpy(e, audio)
        if self.framing == 'native':
            (b,) = self.branches
            return np.asarray(getattr(e, b.model.attr)(audio), dtype=np.float32)
        n = len(audio) // c.frame
        if n == 0:
            return np.zeros((0, sum(self.widths(e))), dtype=np.float32)
        frames = audio[:n * c.frame].reshape(n, c.frame)
        step = _chunk_frames() if self.chunk else n
        rows = []
        for lo in range(0, n, step):
            rows.append(np.concatenate(self.branch_numpy(e, frames[lo:lo + step], c), axis=1))
        return np.concatenate(rows, axis=0)

    def stack_context(self, e, emb):
        if self.context is None:
            return emb
        widths = self.widths(e)
        nb = len(widths) if self.context.branches is None else self.context.branches
        split = sum(widths[:nb])
        n = len(emb)
        if n == 0:
            return np.zeros((0, split * (2 * self.context.k + 1) + emb.shape[1] - split), emb.dtype)
        idx = np.arange(n)
        wide, rest = emb[:, :split], emb[:, split:]
        blocks = [wide[np.clip(idx + o, 0, n - 1)] for o in range(-self.context.k, self.context.k + 1)]
        return np.concatenate(blocks + ([rest] if rest.shape[1] else []), axis=1)

    def numpy(self, e, audio):
        out = self.stack_context(e, self.embed_frames(e, audio)).astype(self.dtype)
        if out.shape[1] != e.n_embeddings:
            raise ValueError(f'{e.embeddername}: recipe gives {out.shape[1]} columns, '
                             f'n_embeddings says {e.n_embeddings}')
        return out

    # ---- ONNX ----------------------------------------------------------------

    def onnx(self, e, opset=17):
        c = self.ctx(e)
        g = Graph(opset)
        x = 'samples'
        if self.prefilter is not None:
            x = self.prefilter.onnx(g, e, x)

        protos = {}  # one export per distinct model, spliced in once per branch

        def proto(model):
            if model not in protos:
                protos[model] = model.export(e, opset).SerializeToString()
            import onnx
            return onnx.ModelProto.FromString(protos[model])

        if self.framing == 'native':
            (b,) = self.branches
            out = g.embed_model(proto(b.model), x, 'b0_')
            return g.model('samples', out, e.embeddername)

        # Whole frames, ragged tail dropped. The branches always run at least
        # one frame (silence when the input is shorter than a frame): AVES's
        # graph cannot take a zero-row batch, and YAMNet pads a short clip up
        # to a frame by itself. The output is trimmed back to n rows, 0 included.
        S = c.frame
        total = g.op('Shape', [x], 'len')
        n = g.op('Div', [total, g.const([S])], 'n_frames')
        n_safe = g.op('Max', [n, g.const([1])], 'n_run')
        usable = g.op('Mul', [n_safe, g.const([S])], 'usable')
        pad = g.op('Max', [g.op('Sub', [usable, total]), g.const([0])], 'pad')
        padded = g.op('Pad', [x, g.op('Concat', [g.const([0]), pad], axis=0)], 'padded')
        cropped = g.op('Slice', [padded, g.const([0]), usable, g.const([0])], 'cropped')
        frames = g.op('Reshape', [cropped, g.const([-1, 1, S])], 'frames')

        blocks = []
        for i, b in enumerate(self.branches):
            y, L = frames, S
            for op in b.ops:
                y = op.onnx(g, y, L, c)
                L = op.length(L, c)
            blocks.append(b.model.onnx(g, e, y, proto(b.model), f'b{i}_'))
        if self.combine == 'contrast':
            blocks = [blocks[0], g.op('Sub', [blocks[0], blocks[1]], 'contrast')]
        emb = blocks[0] if len(blocks) == 1 else g.op('Concat', blocks, 'embeddings', axis=1)

        if self.context is not None:
            emb = self._context_onnx(g, e, emb, n_safe)
        out = g.op('Slice', [emb, g.const([0]), n, g.const([0])], 'embedding')
        return g.model('samples', out, e.embeddername)

    def _context_onnx(self, g, e, emb, n):
        widths = self.widths(e)
        nb = len(widths) if self.context.branches is None else self.context.branches
        split, total = sum(widths[:nb]), sum(widths)
        wide = g.op('Slice', [emb, g.const([0]), g.const([split]), g.const([1])], 'ctx_wide')
        idx = g.op('Range', [g.const(0), g.op('Squeeze', [n, g.const([0])]), g.const(1)], 'ctx_idx')
        last = g.op('Sub', [n, g.const([1])], 'ctx_last')
        blocks = []
        for o in range(-self.context.k, self.context.k + 1):
            shifted = g.op('Add', [idx, g.const(o)])
            clamped = g.op('Max', [g.op('Min', [shifted, last]), g.const([0])])
            blocks.append(g.op('Gather', [wide, clamped], f'ctx_{o:+d}', axis=0))
        if total > split:
            blocks.append(g.op('Slice', [emb, g.const([split]), g.const([total]), g.const([1])], 'ctx_rest'))
        return g.op('Concat', blocks, 'context', axis=1)


def _chunk_frames():
    import config as cfg
    return cfg.CHUNK_FRAMES


# ============================================================================
# The embedder side
# ============================================================================

class RecipeEmbedder(BaseEmbedder):
    """Mixin: embed() and to_onnx() from the class's `recipe`. List it first
    in the bases, so it comes before any parent's own embed():

        class EmbedderX(RecipeEmbedder, EmbedderYamnetTrunkDepth12):
            recipe = Recipe(...)

    The parent still owns initialize() (loading the models the recipe names)
    and build_head()."""
    recipe: Recipe = None

    def embed(self, audio):
        return self.recipe.numpy(self, audio)

    def embed_frames(self, audio):
        return self.recipe.embed_frames(self, audio)

    def stack_context(self, embeddings):
        return self.recipe.stack_context(self, embeddings)

    def to_onnx(self, opset=17):
        return self.recipe.onnx(self, opset)
