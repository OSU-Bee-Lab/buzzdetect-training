import os, sys, glob, pickle, random
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=['03_train','.']
import numpy as np, tensorflow as tf
from embedders.yamnet.yamnet import WaveformFeatures
_=WaveformFeatures.dtype
ROOT='/home/luke/projects/buzzdetect-training'
AUD=f'{ROOT}/02_set/sets/medium/audio/sr16000_fl1/raw'
EMB=f'{ROOT}/02_set/sets/medium/embeddings/yamnet_aves/raw'
ym=tf.keras.models.load_model(f'{ROOT}/embedders/yamnet/yamnet.keras',compile=False)
ym.layers[1].params.patch_hop_seconds=0.96
ps=glob.glob(f'{EMB}/**/*.pickle',recursive=True); random.seed(1); random.shuffle(ps)
n=0
for ep in ps:
    ap=AUD+ep[len(EMB):]
    if not os.path.exists(ap): continue
    a=np.asarray(pickle.load(open(ap,'rb')),dtype=np.float32)
    cy=np.asarray(pickle.load(open(ep,'rb')),dtype=np.float32).reshape(-1,1792)[:, :1024]
    cand={
      'full_1.0s': a,
      'offset0_0.96': a[:15360],
      'centered_0.96': a[320:320+15360],
      'end_0.96': a[-15360:],
    }
    row=[]
    for k,v in cand.items():
        try:
            o=np.asarray(ym(v),dtype=np.float32).reshape(-1,1024)
            row.append(f'{k} {np.abs(o[:len(cy)]-cy).max():.2e}')
        except Exception as e:
            row.append(f'{k} ERR')
    print(os.path.basename(ep)[:30], '|', '  '.join(row))
    n+=1
    if n>=6: break
