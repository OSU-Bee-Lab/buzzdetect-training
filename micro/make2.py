"""Micro-graphs, round 2: where can the tail be joined cheaply? 200 s input only (3,225,840 samples).
All start waveform -> Slice[:12600*256] -> Reshape(12600,256) -> MatMul(256x32) = L (12600,32), a stand-in log-mel."""
import os, sys
import numpy as np, onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
c = lambda name, a: nh.from_array(np.asarray(a), name)
BASE = [h.make_node('Slice', ['waveform', 's0', 's1'], ['a']), h.make_node('Reshape', ['a', 'shp'], ['b']),
        h.make_node('MatMul', ['b', 'w'], ['L'])]
INIT = [c('s0', np.array([0], np.int64)), c('s1', np.array([256 * 12600], np.int64)), c('shp', np.array([-1, 256], np.int64)),
        c('w', rng.normal(size=(256, 32)).astype(np.float32))]
rs = lambda x: h.make_node('ReduceSum', [x], ['predictions'], keepdims=0)


def save(name, nodes, inits=()):
    g = h.make_graph(BASE + nodes, name, [h.make_tensor_value_info('waveform', T.FLOAT, ['samples'])],
                     [h.make_tensor_value_info('predictions', T.FLOAT, [])], INIT + list(inits))
    m = h.make_model(g, opset_imports=[h.make_opsetid('', 17)])
    onnx.checker.check_model(m)
    onnx.save(m, os.path.join(out, name + '.onnx'))


save('base', [rs('L')])
save('concat2d_rows', [h.make_node('Concat', ['L', 'z'], ['y'], axis=0), rs('y')], [c('z', np.zeros((60, 32), np.float32))])
save('pad2d_rows', [h.make_node('Pad', ['L', 'p'], ['y']), rs('y')], [c('p', np.array([0, 0, 60, 0], np.int64))])
save('reshape4d_concat', [h.make_node('Reshape', ['L', 's4'], ['P']), h.make_node('Concat', ['P', 'z4'], ['y'], axis=0), rs('y')],
     [c('s4', np.array([-1, 60, 32, 1], np.int64)), c('z4', np.zeros((1, 60, 32, 1), np.float32))])
# join after a stand-in trunk: per-patch mean -> (210,32); concat a (1,32) row
save('late_concat', [h.make_node('Reshape', ['L', 's3'], ['P']), h.make_node('ReduceMean', ['P'], ['q'], axes=[1], keepdims=0),
                     h.make_node('Concat', ['q', 'zq'], ['y'], axis=0), rs('y')],
     [c('s3', np.array([-1, 60, 32], np.int64)), c('zq', np.zeros((1, 32), np.float32))])
# the tail computed from the waveform's last samples with a small Pad, joined late
save('tail_smallpad_late_concat',
     [h.make_node('Reshape', ['L', 's3'], ['P']), h.make_node('ReduceMean', ['P'], ['q'], axes=[1], keepdims=0),
      h.make_node('Slice', ['waveform', 's1', 'end'], ['t']), h.make_node('Pad', ['t', 'pt'], ['tp']),
      h.make_node('Slice', ['tp', 's0', 'tlen'], ['tp2']), h.make_node('Reshape', ['tp2', 'shp'], ['tb']),
      h.make_node('MatMul', ['tb', 'w'], ['tL']), h.make_node('ReduceMean', ['tL'], ['tq'], axes=[0], keepdims=1),
      h.make_node('Concat', ['q', 'tq'], ['y'], axis=0), rs('y')],
     [c('s3', np.array([-1, 60, 32], np.int64)), c('end', np.array([2**62], np.int64)), c('pt', np.array([0, 15360], np.int64)),
      c('tlen', np.array([15360], np.int64))])
