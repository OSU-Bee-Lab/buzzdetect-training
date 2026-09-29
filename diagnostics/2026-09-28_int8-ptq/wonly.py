import sys, numpy as np, onnx
from onnx import numpy_helper
S=sys.argv[1]; m=onnx.load(S+'/pre.onnx')
convw={n.input[1] for n in m.graph.node if n.op_type=='Conv'}
for t in m.graph.initializer:
    if t.name in convw and len(t.dims)==4 and t.dims[1]*t.dims[0]>1:
        w=numpy_helper.to_array(t).astype(np.float32); mx=np.abs(w).reshape(w.shape[0],-1).max(1).reshape(-1,1,1,1)/127+1e-12
        t.CopyFrom(numpy_helper.from_array((np.round(w/mx).clip(-127,127)*mx).astype(np.float32),t.name))
onnx.save(m,S+'/q_wonly.onnx'); print('wrote wonly')
