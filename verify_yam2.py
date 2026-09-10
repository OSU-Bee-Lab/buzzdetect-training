import os, sys, glob, pickle, random
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=['03_train','.']
import numpy as np, tensorflow as tf
from embedders.yamnet.yamnet import WaveformFeatures
_=WaveformFeatures.dtype
R='/home/luke/projects/buzzdetect-training'
AUD=f'{R}/02_set/sets/medium/audio/sr16000_fl1/raw'; EMB=f'{R}/02_set/sets/medium/embeddings/yamnet_aves/raw'
YC=f'{R}/02_set/sets/medium/embeddings/yamnet/raw'
ym=tf.keras.models.load_model(f'{R}/embedders/yamnet/yamnet.keras',compile=False)
print('patch params:', ym.layers[1].params.patch_window_seconds, ym.layers[1].params.patch_hop_seconds)
ps=glob.glob(f'{EMB}/**/*.pickle',recursive=True); random.seed(2); random.shuffle(ps)
n=0
for ep in ps:
    ap=AUD+ep[len(EMB):]
    if not os.path.exists(ap): continue
    a=np.asarray(pickle.load(open(ap,'rb')),dtype=np.float32)
    cy=np.asarray(pickle.load(open(ep,'rb')),dtype=np.float32).reshape(-1,1792)[:, :1024]
    for hop in (0.48, 0.96, 1.0):
        ym.layers[1].params.patch_hop_seconds=hop
        for tag,clip in (('cen',a[320:320+15360]),('full',a)):
            o=np.asarray(ym(clip),dtype=np.float32).reshape(-1,1024)
            best=min((np.abs(o[i]-cy[0]).max() for i in range(len(o))), default=9)
            print(f'  hop{hop} {tag} rows{len(o)} bestrowdiff {best:.2e}', end='')
        print()
    print(os.path.basename(ep)[:30]); n+=1
    if n>=4: break
