import os, sys, time, json, glob, numpy as np, onnxruntime as ort, soundfile as sf
from collections import defaultdict
d='/home/luke/projects/buzzdetect/engine/models/v4-ft-ps-e60-moderate/'
x=np.fromfile(os.path.join(os.environ.get('INT8_WORK','.local/int8-ptq'),'real10min.f32'),dtype='float32'); sr=16000
print(len(x)/sr,'s')
x=np.tile(x,int(np.ceil(600*sr/len(x))))[:600*sr]  # 10 min
for name in ['model.onnx','model.fp16.onnx']:
    so=ort.SessionOptions(); so.enable_profiling=True; so.intra_op_num_threads=int(sys.argv[1]) if len(sys.argv)>1 else 1
    s=ort.InferenceSession(d+name,so,providers=['CPUExecutionProvider'])
    inp=s.get_inputs()[0]
    xi=x.astype(np.float16) if 'float16' in inp.type else x
    s.run(None,{inp.name:xi[:16000*30]})
    t=time.time(); s.run(None,{inp.name:xi}); dt=time.time()-t
    print(name, inp.type, f'{dt:.2f}s for 600s audio = {600/dt:.0f}x realtime, threads={so.intra_op_num_threads}')
    pf=s.end_profiling(); ev=json.load(open(pf))
    agg=defaultdict(float)
    for e in ev:
        if e.get('cat')=='Node' and e['name'].endswith('_kernel_time'): agg[e['args']['op_name']]+=e['dur']
    tot=sum(agg.values()); print({k:f'{v/tot:.1%}' for k,v in sorted(agg.items(),key=lambda kv:-kv[1])[:8]})
