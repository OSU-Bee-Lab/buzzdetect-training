"""Keras builder for the distillation student: YAMNet, narrower.

The student is YAMNet's exact structure -- WaveformFeatures front end, the
14-layer MobileNetV1 stack from `_YAMNET_LAYER_DEFS`, global average pool,
linear Dense head -- with every layer's filter count replaced by a list you
choose. `widths_for(alpha)` gives round(alpha * base) per layer; pass any list
(shorter than 14 to truncate depth, or non-uniform) as `filters`.

Layer names match YAMNet's (`layer{i}_conv`, `layer{i}_pointwise_conv`, ...,
the same Keras layer names yamnet_frames_model uses), so a later step can load
YAMNet weights by channel selection: for layer i, keep output channels
`sel[i]` of the pointwise/conv kernel and BN vectors, and input channels
`sel[i-1]` of layer i+1's pointwise kernel (its depthwise kernel and BN follow
`sel[i]` too). Nothing here loads weights; init is random.

Output is waveform -> logits, (num_samples,) -> (frames, n_out), patch hop
welded to the 0.96 s window, as the deployed graph is.

Needs `import tensorflow` first in any entry point (see CLAUDE.md).
"""

import os
import re
import sys

import tensorflow as tf  # noqa: F401  (import-order guard)
from tensorflow.keras import Model, layers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from embedders.yamnet.params import Params  # noqa: E402
from embedders.yamnet.yamnet import _YAMNET_LAYER_DEFS, WaveformFeatures, _conv  # noqa: E402
import frontends as fes  # noqa: E402

BASE_FILTERS = [d[3] for d in _YAMNET_LAYER_DEFS]
KERNELS = [d[1] for d in _YAMNET_LAYER_DEFS]
STRIDES = [d[2] for d in _YAMNET_LAYER_DEFS]


def widths_for(alpha, n_layers=14, minimum=1):
    """Per-layer filter counts round(alpha * base), first n_layers layers."""
    return [max(minimum, int(round(alpha * b))) for b in BASE_FILTERS[:n_layers]]


def parse_arch(name):
    """'a<alpha>[_d<depth>][_f<k>]' -> (alpha, depth, fused_from): 'a0.50' is (0.5, 14, 0), 'a0.375_d8' is
    (0.375, 8, 0), 'a0.50_f8' is (0.5, 14, 8): layers 8 on are fused blocks (see `_stack`)."""
    m = re.fullmatch(r'a(\d+(?:\.\d+)?)(?:_d(\d+))?(?:_f(\d+))?', name)
    depth = int(m.group(2) or 14) if m else 0
    fused = int(m.group(3) or 0) if m else 0
    if not m or not 0 < depth <= 14 or (fused and not 2 <= fused <= depth):
        raise ValueError(f'arch {name!r} is not a<alpha>[_d<depth 1-14>][_f<first fused layer 2-depth>]')
    return float(m.group(1)), depth, fused


def fused_from_of(model):
    """First fused layer (1-based) of a built student, 0 if every block is separable."""
    names = {l.name for l in model.layers}
    return next((i for i in range(2, len(_YAMNET_LAYER_DEFS) + 1) if f'layer{i}_conv' in names), 0)


def student_params(n_out):
    p = Params()
    p.num_classes = n_out
    p.patch_hop_seconds = p.patch_window_seconds  # hop welded to window
    return p


def main_checkout():
    """The main checkout's root. A worktree lacks the gitignored data (YAMNet
    weights, sets, models), so data reads go through here."""
    root = ROOT
    for d in ('.claude', '.local'):
        root = root.split(os.sep + d + os.sep + 'worktrees' + os.sep)[0]
    return root


def _stack(net, filters, params, fused_from=0):
    """fused_from: 1-based index of the first fused layer (0 = none). A fused layer is one full conv with the
    layer's kernel and stride in place of depthwise + pointwise (`layer{i}_conv`, as layer 1 is): on the
    CUDA path a depthwise conv costs by its channel count, not its MACs, so a block is one conv, not two."""
    for i, f in enumerate(filters):
        fun, kernel, stride, _ = _YAMNET_LAYER_DEFS[i]
        if fused_from and i + 1 >= fused_from:
            fun = _conv
        net = fun(f'layer{i + 1}', kernel, stride, int(f), params)(net)
    return net


def build_student(filters, n_out=15, stop_at_features=False, name='student',
                  input_type='waveform', expose_code=False, frontend='yamnet', fe_impl='stft', fused_from=0):
    """The student, waveform-in (deploy graph) or mel-in (training graph).

    filters: list of per-layer filter counts, len 1..14 (layer i uses
        _YAMNET_LAYER_DEFS[i]'s kernel/stride and filters[i]).
    input_type: 'waveform' -> input (num_samples,), output (frames, n_out),
        YAMNet front end inside. 'mel' -> input (96, 64) log-mel patches, a
        batch of frames, no front end (it is not learned, so training skips
        it). Conv/BN/Dense layers carry the same names either way, so weights
        move between the two graphs by layer name (`copy_weights`).
    expose_code: also output the GAP code (the head's input) as a second
        output: outputs [logits, code].
    stop_at_features: waveform only; return the log-mel patches instead --
        the front end alone, for timing its floor.
    fe_impl: 'stft' | 'conv' | 'mm' | 'mmt' | 'mmtz', how a non-YAMNet front end's graph is built (frontends.make_features_layer).
    fused_from: first fused layer, 1-based (0 = every block separable); see `_stack`.
    frontend: name in frontends.FRONTENDS. 'yamnet' (default) is YAMNet's own layer,
        exactly as before; any other spec has its own STFT/mel channels, and the
        mel input becomes (96, bands, channels).
    """
    assert 1 <= len(filters) <= len(_YAMNET_LAYER_DEFS)
    params = student_params(n_out)
    fe = None if frontend == 'yamnet' else fes.get(frontend)
    if input_type == 'mel':
        assert not stop_at_features
        shape = (params.patch_frames, params.patch_bands) if fe is None else \
            (fe.frames, fe.bands, fe.n_channels)
        inp = layers.Input(shape=shape, name='mel')
        feats = inp
    else:
        assert input_type == 'waveform'
        inp = layers.Input(shape=(), dtype=tf.float32, name='waveform')
        if fe is None and fe_impl != 'stft':     # YAMNet's log-mel from the 'yamnet' spec's graph (conv: equal to 3e-5)
            feats = fes.make_features_layer(fes.get('yamnet'), fe_impl)(inp)[..., 0]
        elif fe is None:
            _, feats = WaveformFeatures(params)(inp)
        else:
            feats = fes.make_features_layer(fe, fe_impl)(inp)
        if stop_at_features:
            return Model(name=name + '_frontend', inputs=inp, outputs=feats)
    net = layers.Reshape((params.patch_frames, params.patch_bands, 1))(feats) if fe is None else feats
    net = _stack(net, filters, params, fused_from)
    emb = layers.GlobalAveragePooling2D(name='gap')(net)
    logits = layers.Dense(units=n_out, use_bias=True, name='logits')(emb)
    return Model(name=name, inputs=inp, outputs=[logits, emb] if expose_code else logits)


def copy_weights(src, dst):
    """Copy every same-named weighted layer src -> dst (mel <-> waveform)."""
    n = 0
    for layer in dst.layers:
        if layer.weights:
            try:
                s = src.get_layer(layer.name)
            except ValueError:
                continue
            layer.set_weights(s.get_weights())
            n += 1
    return n


def macs_per_frame(filters, n_out=15, h=96, w=64, in_channels=1, fused_from=0):
    """Multiply-adds of the conv stack + head for one 0.96 s patch (front end excluded)."""
    total, c_in = 0, in_channels
    for i, f in enumerate(filters):
        k = KERNELS[i][0] * KERNELS[i][1]
        s = STRIDES[i]
        h, w = -(-h // s), -(-w // s)  # 'same' padding
        if i == 0 or (fused_from and i + 1 >= fused_from):
            total += h * w * k * c_in * f
        else:
            total += h * w * k * c_in  # depthwise
            total += h * w * c_in * f  # pointwise
        c_in = f
    return total + c_in * n_out
