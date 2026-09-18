"""Shared ONNX graph surgery for context-stacking embedders.

`yamnet_context`, `yamnet_context_aves` (and any future embedder built the
same way) widen a frame's embedding with its temporal neighbours in their
numpy `embed()` path (see each embedder's `stack_context()`), but none of
them override `to_onnx()` -- so the exported graph was silently just the
un-widened trunk. `add_context_stack()` is the ONNX equivalent of
`stack_context()`: it appends Range/Gather/Concat nodes that reproduce
`embeddings[np.clip(idx + offset, 0, n - 1)]` for each offset in
`range(-k, k+1)`, concatenated in the same order `stack_context()` uses.

Only the first `widen_dim` columns of the trunk's output are widened; any
remaining columns (an AVES or shifted-YAMNet block that stack_context()
leaves untouched) are passed through as-is and appended after the widened
block, matching every existing `stack_context()` implementation's column
order.
"""

import onnx
from onnx import TensorProto, helper


def crop_waveform_to_whole_frames(trunk_onnx, frame_samples, name_prefix='cropwhole'):
    """Make the graph frame exactly `n_samples // frame_samples` frames.

    A context-stacking embedder's own `embed_frames()` only ever hands its
    trunk model whole frames -- `n = (n_samples - length) // hop + 1`, floored,
    exactly like `EmbedderYamnetContext.n_frames()` -- and drops any ragged
    tail shorter than one frame, including returning zero frames for audio
    shorter than one whole frame. The Keras trunk exported by the default
    `BaseEmbedder.to_onnx()`, left to frame the *raw* waveform itself, has its
    own internal framing rule instead -- e.g. YAMNet's padding a short clip up
    to one frame rather than returning none. This makes the two agree in both
    directions: crops the input down to a whole number of frames before the
    trunk sees it (so it never gets a ragged tail to pad), and separately
    trims the trunk's own output down to that same frame count (so an input
    too short for even one frame -- which the crop above reduces to a
    zero-length waveform -- doesn't ride the trunk's own padding to one row
    when it should produce none). Assumes framehop_prop=1 (hop ==
    frame_samples), which is what 04_deploy/export_onnx.py always exports
    with.

    Returns trunk_onnx, modified in place and also returned.
    """
    graph = trunk_onnx.graph
    old_in = graph.input[0]
    old_in_name = old_in.name
    new_in_name = f'{name_prefix}_waveform'
    trunk_out_name = graph.output[0].name

    def const(name, value, dtype=TensorProto.INT64):
        arr = [value] if not isinstance(value, (list, tuple)) else list(value)
        return helper.make_tensor(f'{name_prefix}_{name}', dtype, [len(arr)], arr)

    inits = [const('zero', 0), const('axis0', 0), const('frame_samples', frame_samples)]
    front_nodes = [
        helper.make_node('Shape', [new_in_name], [f'{name_prefix}_total_len'],
                          name=f'{name_prefix}_shape'),
        helper.make_node('Div', [f'{name_prefix}_total_len', f'{name_prefix}_frame_samples'],
                          [f'{name_prefix}_n_frames'], name=f'{name_prefix}_n_frames'),
        helper.make_node('Mul', [f'{name_prefix}_n_frames', f'{name_prefix}_frame_samples'],
                          [f'{name_prefix}_usable_len'], name=f'{name_prefix}_usable_len'),
        helper.make_node(
            'Slice', [new_in_name, f'{name_prefix}_zero', f'{name_prefix}_usable_len',
                      f'{name_prefix}_axis0'],
            [old_in_name], name=f'{name_prefix}_crop'),
    ]
    trim_name = f'{name_prefix}_trimmed'
    tail_node = helper.make_node(
        'Slice', [trunk_out_name, f'{name_prefix}_zero', f'{name_prefix}_n_frames',
                  f'{name_prefix}_axis0'],
        [trim_name], name=f'{name_prefix}_trim')

    new_in = helper.make_tensor_value_info(new_in_name, TensorProto.FLOAT, ['samples'])
    del graph.input[:]
    graph.input.append(new_in)
    for node in reversed(front_nodes):
        graph.node.insert(0, node)
    graph.node.append(tail_node)
    graph.initializer.extend(inits)
    graph.output[0].name = trim_name
    del graph.output[0].type.tensor_type.shape.dim[:]
    return trunk_onnx


def add_context_stack(trunk_onnx, k, widen_dim, total_dim, n_embeddings, name_prefix='ctxstack'):
    """Append context-widening nodes to `trunk_onnx`'s single 2-D output.

    Args:
        trunk_onnx: onnx.ModelProto whose graph has exactly one output,
            shape (n_frames, total_dim).
        k: context_frames (frames of neighbour on each side to fold in).
        widen_dim: how many leading columns of the output get widened.
        total_dim: the trunk output's full width (widen_dim <= total_dim).
        n_embeddings: expected width of the new output, for a sanity check --
            widen_dim * (2k+1) + (total_dim - widen_dim).
        name_prefix: node/initializer name prefix, unique per caller so two
            calls in the same graph (there is only ever one) never collide.

    Returns trunk_onnx, modified in place and also returned.
    """
    expected = widen_dim * (2 * k + 1) + (total_dim - widen_dim)
    if expected != n_embeddings:
        raise ValueError(
            f'add_context_stack: widen_dim={widen_dim}, total_dim={total_dim}, '
            f'k={k} produces {expected}, not n_embeddings={n_embeddings}')

    graph = trunk_onnx.graph
    out_name = graph.output[0].name

    def const(name, value, dtype=TensorProto.INT64):
        arr = [value] if not isinstance(value, (list, tuple)) else list(value)
        return helper.make_tensor(f'{name_prefix}_{name}', dtype, [len(arr)], arr)

    inits = [
        const('zero', 0),
        const('one', 1),
        const('widen_dim', widen_dim),
        const('total_dim', total_dim),
    ]
    nodes = [
        helper.make_node('Shape', [out_name], [f'{name_prefix}_full_shape'],
                          name=f'{name_prefix}_shape'),
        helper.make_node('Slice', [f'{name_prefix}_full_shape',
                                    f'{name_prefix}_zero', f'{name_prefix}_one',
                                    f'{name_prefix}_zero'],
                          [f'{name_prefix}_n'], name=f'{name_prefix}_n_frames'),
    ]

    if widen_dim < total_dim:
        nodes.append(helper.make_node(
            'Slice', [out_name, f'{name_prefix}_widen_dim', f'{name_prefix}_total_dim',
                      f'{name_prefix}_one'],
            [f'{name_prefix}_passthrough'], name=f'{name_prefix}_slice_passthrough'))

    nodes.append(helper.make_node(
        'Slice', [out_name, f'{name_prefix}_zero', f'{name_prefix}_widen_dim',
                  f'{name_prefix}_one'],
        [f'{name_prefix}_widen_src'], name=f'{name_prefix}_slice_widen'))

    # Range wants rank-0 (scalar) tensors, not the rank-1 ones Slice needs above
    nodes.append(helper.make_node(
        'Squeeze', [f'{name_prefix}_zero'], [f'{name_prefix}_zero_scalar'],
        name=f'{name_prefix}_zero_scalar_squeeze'))
    nodes.append(helper.make_node(
        'Squeeze', [f'{name_prefix}_n'], [f'{name_prefix}_n_scalar'],
        name=f'{name_prefix}_n_scalar_squeeze'))
    nodes.append(helper.make_node(
        'Squeeze', [f'{name_prefix}_one'], [f'{name_prefix}_one_scalar'],
        name=f'{name_prefix}_one_scalar_squeeze'))
    # arange(n), reused for every offset
    nodes.append(helper.make_node(
        'Range', [f'{name_prefix}_zero_scalar', f'{name_prefix}_n_scalar',
                  f'{name_prefix}_one_scalar'],
        [f'{name_prefix}_idx_base'], name=f'{name_prefix}_arange'))
    # n - 1, for the upper clip bound
    nodes.append(helper.make_node(
        'Sub', [f'{name_prefix}_n', f'{name_prefix}_one'], [f'{name_prefix}_n_minus_1'],
        name=f'{name_prefix}_n_minus_1'))

    block_names = []
    for offset in range(-k, k + 1):
        suffix = f'off{offset}'.replace('-', 'm')
        tag = f'{name_prefix}_{suffix}'
        inits.append(const(f'{suffix}_c', offset))
        nodes.append(helper.make_node(
            'Add', [f'{name_prefix}_idx_base', f'{tag}_c'], [f'{tag}_shifted'],
            name=f'{tag}_add'))
        nodes.append(helper.make_node(
            'Clip', [f'{tag}_shifted', f'{name_prefix}_zero', f'{name_prefix}_n_minus_1'],
            [f'{tag}_clipped'], name=f'{tag}_clip'))
        nodes.append(helper.make_node(
            'Gather', [f'{name_prefix}_widen_src', f'{tag}_clipped'], [f'{tag}_gathered'],
            axis=0, name=f'{tag}_gather'))
        block_names.append(f'{tag}_gathered')

    concat_inputs = block_names + (
        [f'{name_prefix}_passthrough'] if widen_dim < total_dim else [])
    stacked_name = f'{name_prefix}_stacked'
    nodes.append(helper.make_node(
        'Concat', concat_inputs, [stacked_name], axis=1, name=f'{name_prefix}_concat'))

    graph.node.extend(nodes)
    graph.initializer.extend(inits)
    graph.output[0].name = stacked_name
    del graph.output[0].type.tensor_type.shape.dim[:]
    return trunk_onnx
