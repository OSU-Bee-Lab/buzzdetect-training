"""Micro-graphs for the per-call cost: waveform -> one op -> ReduceSum. Train env (needs onnx)."""
import os, sys
import numpy as np, onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
N = 15600 + 15360 * 13          # only to size constants; the input stays dynamic


def save(name, nodes, inits=()):
    g = h.make_graph(nodes, name, [h.make_tensor_value_info('waveform', T.FLOAT, ['samples'])],
                     [h.make_tensor_value_info('predictions', T.FLOAT, [])], list(inits))
    m = h.make_model(g, opset_imports=[h.make_opsetid('', 17)])
    onnx.checker.check_model(m)
    onnx.save(m, os.path.join(out, name + '.onnx'))


c = lambda name, a: nh.from_array(np.asarray(a), name)
rs = lambda x: h.make_node('ReduceSum', [x], ['predictions'], keepdims=0)
save('sum', [rs('waveform')])
save('mul_sum', [h.make_node('Mul', ['waveform', 'k'], ['y']), rs('y')], [c('k', np.float32(2))])
save('relu_sum', [h.make_node('Relu', ['waveform'], ['y']), rs('y')])
save('padconst_sum', [h.make_node('Pad', ['waveform', 'p'], ['y']), rs('y')], [c('p', np.array([0, 15120], np.int64))])
save('concatconst_sum', [h.make_node('Concat', ['waveform', 'z'], ['y'], axis=0), rs('y')],
     [c('z', np.zeros(15120, np.float32))])
save('slice_reshape_matmul_sum',
     [h.make_node('Slice', ['waveform', 's0', 's1'], ['a']), h.make_node('Reshape', ['a', 'shp'], ['b']),
      h.make_node('MatMul', ['b', 'w'], ['y']), rs('y')],
     [c('s0', np.array([0], np.int64)), c('s1', np.array([256 * 12000], np.int64)), c('shp', np.array([-1, 256], np.int64)),
      c('w', np.random.default_rng(0).normal(size=(256, 244)).astype(np.float32))])
save('reshape_matmul_sum',       # needs samples % 240 == 0 (15600 + 15360 k is)
     [h.make_node('Reshape', ['waveform', 'shp'], ['b']), h.make_node('MatMul', ['b', 'w'], ['y']), rs('y')],
     [c('shp', np.array([-1, 240], np.int64)), c('w', np.random.default_rng(0).normal(size=(240, 244)).astype(np.float32))])
