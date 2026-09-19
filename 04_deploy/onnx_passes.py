"""Two graph rewrites applied to a freshly exported ONNX model.

Both are exactly the kind of thing an exporter should do and a runtime cannot:
they need to happen once, to a build artifact, not on every session creation.
Neither changes what the model computes -- `fold_batchnorm` is an algebraic
identity and `fuse_conv_relu` only changes which kernel applies the `max()`.

Depends on `onnx` alone: no TensorFlow, no onnxruntime, nothing from this repo.
The pair also runs as a CLI, for a graph that came from somewhere else:

    python 04_deploy/onnx_passes.py model.onnx model.onnx

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


_SCALAR_ELEMENTWISE = ('Add', 'Sub', 'Mul', 'Div', 'Max', 'Min', 'Ceil', 'Floor',
                       'Cast', 'Neg', 'Abs')


def promote_scalars(model):
    """Rewrite rank-0 shape arithmetic as rank-1, in place; return the number
    of tensors promoted.

    Frameworks compute things like "how many samples to pad" as a chain of
    rank-0 tensors: a Squeeze of a length, a few elementwise ops against rank-0
    constants, then an Unsqueeze into a Pad's paddings. CoreML's MLProgram
    format types a rank-0 result as [1] and rejects the graph ("Output '0'
    has unexpected type 'ios18.mul'"), and this only surfaces when the chain
    survives constant folding, i.e. when its length isn't known until runtime.
    A rank-1 [1] tensor holds the same number, so: the Squeeze that opens the
    chain and the Unsqueeze that closes it become Identity, and the rank-0
    constants in between are replaced by [1] copies. Every tensor of the chain
    must be consumed only by the chain (or its closing Unsqueeze); if anything else
    reads one, that chain is left alone rather than half-converted.
    """
    g = model.graph
    init = {t.name: t for t in g.initializer}
    rank = {t.name: len(t.dims) for t in g.initializer}
    for v in list(onnx.shape_inference.infer_shapes(model).graph.value_info):
        if v.type.HasField('tensor_type') and v.type.tensor_type.HasField('shape'):
            rank.setdefault(v.name, len(v.type.tensor_type.shape.dim))
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    def opens(n):   # Squeeze of a [1] into a scalar
        return n.op_type == 'Squeeze' and len(n.output) == 1 and rank.get(n.output[0]) == 0

    promoted, opened = set(), []
    for n in g.node:
        if opens(n):
            promoted.add(n.output[0])
            opened.append(n)
    chain, closers = [], []
    changed = True
    while changed:
        changed = False
        for n in g.node:
            if n in chain or n in closers or not n.output:
                continue
            ins = [i for i in n.input if i]
            data = ins[:1] if n.op_type == 'Cast' else ins
            if (n.op_type in _SCALAR_ELEMENTWISE
                    and any(i in promoted for i in data)
                    and all(i in promoted or (i in init and rank[i] == 0) for i in data)):
                promoted.add(n.output[0])
                chain.append(n)
                changed = True
            elif (n.op_type == 'Unsqueeze' and ins and ins[0] in promoted
                    and rank.get(n.output[0]) == 1):
                closers.append(n)
                changed = True

    # Group the rewritten nodes into connected chains, then keep only chains
    # that are entirely self-contained: a promoted tensor read by anything
    # outside its chain (a Range that needs a true scalar, say), or a constant
    # shared with such a node, disqualifies that chain and no other.
    members = opened + chain + closers
    ids = {id(n): k for k, n in enumerate(members)}
    parent = list(range(len(members)))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    producer = {n.output[0]: n for n in opened + chain}
    for n in chain + closers:
        for i in n.input:
            if i in producer:
                parent[find(ids[id(n)])] = find(ids[id(producer[i])])
    bad = set()
    for n in opened + chain:
        for c in consumers.get(n.output[0], []):
            if id(c) not in ids:
                bad.add(find(ids[id(n)]))
    keep = {id(n) for n in members if find(ids[id(n)]) not in bad}
    n_promoted = 0
    for n in opened + closers:
        if id(n) in keep:
            src = n.input[0]
            n.op_type = 'Identity'
            del n.input[:]
            n.input.append(src)
            del n.attribute[:]
            n_promoted += n in opened
    copies = {}
    for n in chain:
        if id(n) not in keep:
            continue
        n_promoted += 1
        for k, i in enumerate(n.input):
            if i in init and rank[i] == 0 and i not in promoted:
                # A copy, not an edit: the constant may also feed a node that
                # needs a true scalar (a Range's start).
                if i not in copies:
                    copies[i] = i + '_r1'
                    g.initializer.append(numpy_helper.from_array(
                        numpy_helper.to_array(init[i]).reshape(1), copies[i]))
                n.input[k] = copies[i]
    return n_promoted


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


def count_fusable(model):
    """Conv+Relu pairs fuse_conv_relu would take. Zero once it has run.

    The point of counting is to tell "this backbone has no plain Conv->Relu to
    fuse" -- Relu6 and HardSwish backbones do not, and neither does a graph
    whose batchnorms never folded -- apart from "the pass ran and missed some",
    which would be a bug here.
    """
    producer = {out: n for n in model.graph.node for out in n.output}
    consumers = {}
    for node in model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)
    outputs = {o.name for o in model.graph.output}

    n = 0
    for relu in model.graph.node:
        if relu.op_type != 'Relu':
            continue
        conv = producer.get(relu.input[0])
        if conv is None or conv.op_type != 'Conv':
            continue
        if len(consumers.get(conv.output[0], [])) != 1:
            continue
        if conv.output[0] in outputs:
            continue
        n += 1
    return n


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
