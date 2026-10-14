"""Where a graph's GPU time goes, by timing prefixes of it (whole graphs, not a per-node profile: on the
CUDA EP per-node times are asynchronous launches, not costs).

    python 05_distill/prefix_bench.py cut <model.onnx>                    # train env (needs onnx): writes <model dir>/prefix/
    <engine venv python> 05_distill/prefix_bench.py time <model.onnx> [--seconds 200]

For each node output along the graph, in order: cut the graph there, reduce the cut tensor to one scalar
on the device (so no large copy back to the host), and time session.run as bench_arch.py does. A row's
time minus the row above it is what that node adds. Integer (shape arithmetic) outputs are skipped.
"""
import argparse
import os
import statistics
import sys
import time

import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dpaths as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('phase', choices=['cut', 'time'])
    ap.add_argument('model')
    ap.add_argument('--out', help='where the prefixes go (default <model dir>/prefix)')
    ap.add_argument('--seconds', type=float, default=200)
    ap.add_argument('--repeats', type=int, default=15)
    ap.add_argument('--warmup', type=int, default=2)
    a = ap.parse_args()
    path = os.path.abspath(a.model)
    out_dir = a.out or os.path.join(os.path.dirname(path), 'prefix')
    if a.phase == 'cut':
        cut(path, out_dir)
    else:
        do_time(out_dir, a)


def cut(path, out_dir):
    import onnx
    from onnx import TensorProto, helper, shape_inference
    m = shape_inference.infer_shapes(onnx.load(path))
    dtype = {v.name: v.type.tensor_type.elem_type for v in list(m.graph.value_info) + list(m.graph.output)}
    os.makedirs(out_dir, exist_ok=True)
    index = []
    for i, node in enumerate(m.graph.node):
        out = node.output[0]
        if dtype.get(out) != TensorProto.FLOAT:
            continue
        g = helper.make_graph(list(m.graph.node[:i + 1]) + [helper.make_node('ReduceSum', [out], ['cut'], keepdims=0)],
                              'prefix', list(m.graph.input), [helper.make_tensor_value_info('cut', TensorProto.FLOAT, [])],
                              list(m.graph.initializer))
        onnx.save(helper.make_model(g, opset_imports=list(m.opset_import)), os.path.join(out_dir, f'{i:04d}.onnx'))
        index.append({'file': f'{i:04d}.onnx', 'node': node.name, 'op': node.op_type, 'out': out})
    json.dump(index, open(os.path.join(out_dir, 'index.json'), 'w'), indent=1)
    print(f'{len(index)} prefixes -> {out_dir}')


def do_time(out_dir, a):
    index = json.load(open(os.path.join(out_dir, 'index.json')))
    os.chdir(D.ENGINE)
    sys.path.insert(0, D.ENGINE)
    import onnxruntime as ort
    from src.inference import onnx as bd_onnx
    providers, options = bd_onnx.providers_for('GPU')

    n = int(round(a.seconds * 16000))
    hop, mn = 15360, 15600
    frames = 1 + int(np.ceil(np.float32(max(0, n - mn)) * (np.float32(1) / np.float32(hop))))
    feed = np.zeros(mn + hop * frames, np.float32)
    feed[:n] = np.random.default_rng(0).uniform(-0.1, 0.1, n).astype(np.float32)

    prev = 0.0
    print(f'{"ms":>8s} {"+ms":>7s}  node (op) -> output', flush=True)
    for row in index:
        try:
            s = ort.InferenceSession(os.path.join(out_dir, row['file']), providers=providers, provider_options=options)
        except Exception as e:                      # a cut that leaves the graph invalid (rare): say so, go on
            print(f'{"-":>8s} {"-":>7s}  {row["node"]} ({row["op"]}): {str(e)[:80]}', flush=True)
            continue
        key = s.get_inputs()[0].name
        for _ in range(a.warmup):
            s.run(None, {key: feed})
        ts = []
        for _ in range(a.repeats):
            t = time.perf_counter()
            s.run(None, {key: feed})
            ts.append(time.perf_counter() - t)
        ms = 1000 * statistics.median(ts)
        print(f'{ms:8.2f} {ms - prev:+7.2f}  {row["node"][-50:]} ({row["op"]}) -> {row["out"][-40:]}', flush=True)
        prev = ms


if __name__ == '__main__':
    main()
