import os, sys, glob, pickle, random
sys.path[:0]=['03_train','.']
import numpy as np, tensorflow as tf
from embedders.yamnet.yamnet import WaveformFeatures
_=WaveformFeatures.dtype
gpus=tf.config.list_physical_devices('GPU')
for g in gpus: tf.config.experimental.set_memory_growth(g,True)
print('GPUs:',gpus)
R='/home/luke/projects/buzzdetect-training'
AUD=f'{R}/02_set/sets/medium/audio/sr16000_fl1/raw'; EMB=f'{R}/02_set/sets/medium/embeddings/yamnet_aves/raw'
ym=tf.keras.models.load_model(f'{R}/embedders/yamnet/yamnet.keras',compile=False)
ps=glob.glob(f'{EMB}/**/*.pickle',recursive=True); random.seed(2); random.shuffle(ps)
n=0
for ep in ps:
    ap=AUD+ep[len(EMB):]
    if not os.path.exists(ap): continue
    a=np.asarray(pickle.load(open(ap,'rb')),dtype=np.float32)
    cy=np.asarray(pickle.load(open(ep,'rb')),dtype=np.float32).reshape(-1,1792)[:, :1024]
    o=np.asarray(ym(a[320:320+15360]),dtype=np.float32).reshape(-1,1024)
    print(f'{os.path.basename(ep)[:34]:34} cen-GPU maxdiff {np.abs(o[0]-cy[0]).max():.2e}')
    n+=1
    if n>=6: break
