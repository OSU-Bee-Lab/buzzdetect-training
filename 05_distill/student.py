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
import sys

import tensorflow as tf  # noqa: F401  (import-order guard)
from tensorflow.keras import Model, layers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from embedders.yamnet.params import Params  # noqa: E402
from embedders.yamnet.yamnet import _YAMNET_LAYER_DEFS, WaveformFeatures  # noqa: E402

BASE_FILTERS = [d[3] for d in _YAMNET_LAYER_DEFS]
KERNELS = [d[1] for d in _YAMNET_LAYER_DEFS]
STRIDES = [d[2] for d in _YAMNET_LAYER_DEFS]


def widths_for(alpha, n_layers=14, minimum=1):
    """Per-layer filter counts round(alpha * base), first n_layers layers."""
    return [max(minimum, int(round(alpha * b))) for b in BASE_FILTERS[:n_layers]]


def student_params(n_out):
    p = Params()
    p.num_classes = n_out
    p.patch_hop_seconds = p.patch_window_seconds  # hop welded to window
    return p


def build_student(filters, n_out=15, stop_at_features=False, name='student'):
    """waveform -> logits (frames, n_out).

    filters: list of per-layer filter counts, len 1..14 (layer i uses
        _YAMNET_LAYER_DEFS[i]'s kernel/stride and filters[i]).
    stop_at_features: return the log-mel patches (frames, 96, 64) instead --
        the front end alone, for timing its floor.
    """
    assert 1 <= len(filters) <= len(_YAMNET_LAYER_DEFS)
    params = student_params(n_out)
    waveform = layers.Input(shape=(), dtype=tf.float32, name='waveform')
    _, feats = WaveformFeatures(params)(waveform)
    if stop_at_features:
        return Model(name=name + '_frontend', inputs=waveform, outputs=feats)
    net = layers.Reshape((params.patch_frames, params.patch_bands, 1))(feats)
    for i, f in enumerate(filters):
        fun, kernel, stride, _ = _YAMNET_LAYER_DEFS[i]
        net = fun(f'layer{i + 1}', kernel, stride, int(f), params)(net)
    emb = layers.GlobalAveragePooling2D()(net)
    logits = layers.Dense(units=n_out, use_bias=True, name='logits')(emb)
    return Model(name=name, inputs=waveform, outputs=logits)


def macs_per_frame(filters, n_out=15, h=96, w=64):
    """Multiply-adds of the conv stack + head for one 0.96 s patch (front end excluded)."""
    total, c_in = 0, 1
    for i, f in enumerate(filters):
        k = KERNELS[i][0] * KERNELS[i][1]
        s = STRIDES[i]
        h, w = -(-h // s), -(-w // s)  # 'same' padding
        if i == 0:
            total += h * w * k * c_in * f
        else:
            total += h * w * k * c_in  # depthwise
            total += h * w * c_in * f  # pointwise
        c_in = f
    return total + c_in * n_out
