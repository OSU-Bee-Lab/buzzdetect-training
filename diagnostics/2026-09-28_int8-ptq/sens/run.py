import sys, os, time, numpy as np, onnxruntime as ort
W=sys.argv[1]; variants=sys.argv[2:]
S=os.environ.get('INT8_GRAPHS','/home/luke/projects/buzzdetect-training/.local/int8-ptq/graphs')+'/'
P={'fp32':'/home/luke/projects/buzzdetect/engine/models/v4-ft-ps-e60-moderate/model.onnx'}
for v in ['wonly','entropy_u8','entropy_u8_pw','entropy_u8_pw_skiplast4','entropy_u8_pw_skipfirst4','minmax_u8']: P[v]=S+'q_'+v+'.onnx'
BI=8
for v in variants:
    so=ort.SessionOptions(); so.intra_op_num_threads=4
    s=ort.InferenceSession(P[v],so,providers=['CPUExecutionProvider'])
    for fi in range(5):
        out_p=f'{W}/act_{v}_{fi}.npy'
        if os.path.exists(out_p): continue
        t=time.time(); a=np.load(f'{W}/audio_{fi}.npy',mmap_mode='r'); runs=np.load(f'{W}/runs_{fi}.npy')
        outs=[]; o=0
        for n in runs:
            L=15360+(n-1)*3072; r=np.ascontiguousarray(a[o:o+L]); o+=L
            out=np.empty(n,dtype=np.float32)
            for k in range(min(5,n)):
                y=s.run(None,{'waveform':r[k*3072:]})[0][:,BI]
                assert len(y)==(n-1-k)//5+1,(v,fi,len(y),n,k)
                out[k::5]=y
            outs.append(out)
        np.save(out_p,np.concatenate(outs)); print(v,'fold',fi,f'{time.time()-t:.0f}s',flush=True)
