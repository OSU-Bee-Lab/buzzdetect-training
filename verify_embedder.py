"""Verify reconstructed embedders/yamnet_aves against the cached medium embeddings."""
import os, sys, glob, pickle, random
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['BUZZDETECT_AVES_DEVICE'] = 'cpu'
sys.path[:0] = ['03_train', '.']
import numpy as np
from embedders.embedding import load_embedder

ROOT = '/home/luke/projects/buzzdetect-training'
AUD = f'{ROOT}/02_set/sets/medium/audio/sr16000_fl1/raw'
EMB = f'{ROOT}/02_set/sets/medium/embeddings/yamnet_aves/raw'

emb_pickles = glob.glob(f'{EMB}/**/*.pickle', recursive=True)
random.seed(0)
random.shuffle(emb_pickles)

e = load_embedder('yamnet_aves', framehop_prop=1, initialize=True)

worst = 0.0
n = 0
for ep in emb_pickles:
    ap = AUD + ep[len(EMB):]
    if not os.path.exists(ap):
        continue
    audio = np.asarray(pickle.load(open(ap, 'rb')), dtype=np.float32)
    cached = np.asarray(pickle.load(open(ep, 'rb')), dtype=np.float32).reshape(-1, 1792)
    got = np.asarray(e.embed(audio), dtype=np.float32).reshape(-1, 1792)
    if got.shape != cached.shape:
        print(f'SHAPE MISMATCH {got.shape} vs {cached.shape}  {os.path.basename(ep)}')
        continue
    d = np.abs(got - cached)
    md = float(d.max())
    rel = md / (np.abs(cached).max() + 1e-9)
    worst = max(worst, md)
    y = np.abs(got[:, :1024] - cached[:, :1024]).max()
    a = np.abs(got[:, 1024:] - cached[:, 1024:]).max()
    print(f'{os.path.basename(ep)[:40]:40}  maxabs {md:.3e}  (yamnet {y:.2e}  aves {a:.2e})  rel {rel:.1e}')
    n += 1
    if n >= 15:
        break

print(f'\n{n} frames checked, worst maxabs diff {worst:.3e}')
