"""Two graph rewrites applied to a freshly exported ONNX model.

Both are exactly the kind of thing an exporter should do and a runtime cannot:
they need to happen once, to a build artifact, not on every session creation.
Neither changes what the model computes -- `fold_batchnorm` is an algebraic
identity and `fuse_conv_relu` only changes which kernel applies the `max()`.

Depends on `onnx` alone: no TensorFlow, no onnxruntime, nothing from this repo.
The pair also runs as a CLI, for a graph that came from somewhere else:

    python tools/onnx_passes.py model.onnx model.onnx

`fuse_conv_relu` is vendored from buzzdetect's investigation of why the ONNX
path was slower than TensorFlow on CUDA (`benchmarks/onnx-vs-tf/RESULTS.md` §8
on the `bench/onnx-vs-tf` branch, and `HANDOFF.md`). `fold_batchnorm` is
adapted from `fold_depthwise_bn.py` in the same directory.
"""

import argparse

import numpy as np
import onnx
from onnx import numpy_helper


def fold_batchnorm(model):
    """Fold a Conv -> Mul -> Add chain back into the convolution's weights.

    tf2onnx emits each depthwise layer's batch normalisation as a standalone
    per-channel Mul and Add. The pointwise convolutions arrive already folded
    (weights plus a bias), so this is an export artifact rather than anything
    inherent to the model.

    It costs twice: two extra full passes over an activation tensor per layer,
    and -- because the pair sits between the Conv and its Relu -- it blocks
    `fuse_conv_relu` below from ever seeing an adjacent pair. Run this first.

    The rewrite is the identity ((X*W) * s + b) == X * (W*s) + b for a
    per-channel s, so it is exact in real arithmetic; in float32 it moves the
    last bits, which is why the export re-checks parity against Keras after it.

    Returns (model, n_folded). Modified in place and also returned.
    """
    graph = model.graph
    initializer = {i.name: i for i in graph.initializer}
    values = {k: numpy_helper.to_array(v) for k, v in initializer.items()}
    producer = {out: n for n in graph.node for out in n.output}
    consumers = {}
    for node in graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    drop, n_folded = set(), 0
    for mul in list(graph.node):
        if mul.op_type != 'Mul' or len(mul.output) != 1:
            continue
        conv = producer.get(mul.input[0])
        name_scale = mul.input[1]
        if conv is None or conv.op_type != 'Conv' or name_scale not in values:
            continue
        # Both intermediates must be private to this chain, for the same
        # reason fuse_conv_relu checks it: anything else reading them would
        # lose its input.
        if len(consumers.get(conv.output[0], [])) != 1:
            continue
        if len(consumers.get(mul.output[0], [])) != 1:
            continue
        add = consumers[mul.output[0]][0]
        if add.op_type != 'Add':
            continue
        name_shift = add.input[1]
        if name_shift not in values:
            continue

        name_w = conv.input[1]
        w = values[name_w]
        scale = values[name_scale].reshape(-1)
        shift = values[name_shift].reshape(-1)
        # Per-output-channel only. Anything else is not a batchnorm and folding
        # it into the kernel would be wrong.
        if scale.shape[0] != w.shape[0] or shift.shape[0] != w.shape[0]:
            continue

        w_folded = (w * scale.reshape(-1, 1, 1, 1)).astype(np.float32)
        initializer[name_w].CopyFrom(numpy_helper.from_array(w_folded, name_w))
        name_bias = name_w + '_folded_bias'
        graph.initializer.append(
            numpy_helper.from_array(shift.astype(np.float32), name_bias))
        if len(conv.input) >= 3:
            conv.input[2] = name_bias
        else:
            conv.input.append(name_bias)
        # The Conv now produces what the Add produced, so both go away.
        conv.output[0] = add.output[0]
        drop.add(id(mul))
        drop.add(id(add))
        n_folded += 1

    for node in [n for n in graph.node if id(n) in drop]:
        graph.node.remove(node)

    return model, n_folded


def fuse_conv_relu(model):
    """Rewrite every Conv -> Relu pair into a single com.microsoft.FusedConv.

    tf2onnx emits every Relu as its own node, and onnxruntime's
    ConvActivationFusion pass is registered for the CPU execution provider but
    not for CUDA. So on a GPU each Relu is a separate kernel that reads a whole
    activation tensor out of memory and writes it back just to apply a max().
    Handing onnxruntime the pairs already fused lets the convolution kernel
    apply the activation to data it is still holding.

    Worth 1.38x end to end on CUDA and bit-exact there
    (`benchmarks/onnx-vs-tf/RESULTS.md` §8). Free on CoreML and on the CPU
    provider, which already does this fusion itself (`COREML.md`).

    A pair is only fused when the Conv's output feeds nothing but that one
    Relu. If anything else reads it -- a skip connection, a second branch, a
    graph output -- fusing would delete a value that node still needs. YAMNet
    has no such case, but MobileNet-v2/v3 and ResNet backbones do, so the check
    is not optional if this is ever pointed at another backbone.

    Returns (model, n_fused). Modified in place and also returned.
    """
    graph = model.graph
    producer = {out: n for n in graph.node for out in n.output}
    consumers = {}
    for node in graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)
    outputs = {o.name for o in graph.output}

    drop, n_fused = set(), 0
    for relu in list(graph.node):
        if relu.op_type != 'Relu':
            continue
        conv = producer.get(relu.input[0])
        if conv is None or conv.op_type != 'Conv':
            continue
        if len(consumers.get(conv.output[0], [])) != 1:
            continue
        if conv.output[0] in outputs:
            continue

        conv.op_type = 'FusedConv'
        conv.domain = 'com.microsoft'
        conv.attribute.append(onnx.helper.make_attribute('activation', 'Relu'))
        conv.output[0] = relu.output[0]
        drop.add(id(relu))
        n_fused += 1

    for node in [n for n in graph.node if id(n) in drop]:
        graph.node.remove(node)

    if n_fused and not any(o.domain == 'com.microsoft' for o in model.opset_import):
        model.opset_import.append(onnx.helper.make_opsetid('com.microsoft', 1))

    return model, n_fused


def drop_orphan_initializers(model):
    """Remove initializers no node reads any more.

    fold_batchnorm leaves the batchnorm's scale and shift behind, along with
    whatever tf2onnx precomputed to build them. onnxruntime prints a warning
    per orphan on every session creation -- around thirty lines before a single
    chunk is analysed -- and the export is the right place to make that stop.

    Returns (model, n_dropped).
    """
    graph = model.graph
    used = {i for n in graph.node for i in n.input}
    used |= {o.name for o in graph.output}
    orphans = [i for i in graph.initializer if i.name not in used]
    for i in orphans:
        graph.initializer.remove(i)
    # Graph inputs that shadow an initializer go too, or the checker complains
    # about an input with no producer and no value.
    names = {i.name for i in orphans}
    for value in [v for v in graph.input if v.name in names]:
        graph.input.remove(value)
    return model, len(orphans)


def optimize(model):
    """The passes, in the only order that works.

    fold_batchnorm has to run before fuse_conv_relu: the Mul and Add it removes
    sit between the Conv and its Relu, and while they are there only the
    already-folded pointwise convolutions can fuse -- 14 of 27 on YAMNet.

    Returns (model, n_folded, n_fused, n_dropped).
    """
    model, n_folded = fold_batchnorm(model)
    model, n_fused = fuse_conv_relu(model)
    model, n_dropped = drop_orphan_initializers(model)
    return model, n_folded, n_fused, n_dropped


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('src')
    parser.add_argument('dst')
    args = parser.parse_args()

    model = onnx.load(args.src)
    before = len(model.graph.node)
    model, n_folded, n_fused, n_dropped = optimize(model)
    onnx.checker.check_model(model)
    onnx.save(model, args.dst)
    print(f'{args.src}: {before} -> {len(model.graph.node)} nodes, '
          f'{n_folded} batchnorms folded, {n_fused} Conv+Relu pairs fused, '
          f'{n_dropped} orphaned initializers dropped')
    print(f'wrote {args.dst}')


if __name__ == '__main__':
    main()
