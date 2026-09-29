import sys, time, numpy as np, onnxruntime as ort
S=sys.argv[1]; th=int(sys.argv[2]); ref=sys.argv[3]; cands=sys.argv[4:]
x=np.fromfile(S+'/real10min.f32',dtype='float32')
def run(p):
    so=ort.SessionOptions(); so.intra_op_num_threads=th
    s=ort.InferenceSession(p,so,providers=['CPUExecutionProvider'])
    s.run(None,{'waveform':x[:16000*30]})
    t=time.time(); y=s.run(None,{'waveform':x})[0]; return y,time.time()-t
yr,tr=run(ref); print(f'ref {ref.split("/")[-1]}: {tr:.2f}s  {600/tr:.0f}x rt  (ORT {ort.__version__}, {th} thr)')
for c in cands:
    y,t=run(c); d=np.abs(y-yr)
    print(f'{c.split("/")[-1]}: {t:.2f}s {600/t:.0f}x rt speedup {tr/t:.2f}x | logit max|d|={d.max():.3f} mean={d.mean():.4f} top1 agree={np.mean(y.argmax(1)==yr.argmax(1)):.3f} buzz(ins_buzz idx8) corr={np.corrcoef(y[:,8],yr[:,8])[0,1]:.4f}')
