"""Dump per-fold (X, y) matrices for the rotating folds, once, so the readout
sweep doesn't re-read 33k pickles per configuration."""
import os, sys, numpy as np, pandas as pd
sys.path[:0] = ['03_train', '.']
import config as cfg
from dataset import build_fold_dataset
from train_utils import build_classes

setname = 'medium'
tr = pd.read_csv(cfg.path_translation(setname, 'general'))
classes = build_classes(tr)
ibuzz = classes.index('ins_buzz')
folds = pd.read_csv(os.path.join(cfg.SET_DIR, setname, 'folds.csv'))

for emb in ['yamnet_aves', 'yamnet', 'aves']:
    out = {}
    for role in ['rotate', 'train']:
        for f in sorted(folds.loc[folds.role == role, 'fold'].unique()):
            X, Y = [], []
            for s in build_fold_dataset(cfg.dir_embeddings_fold(setname, emb, f), tr):
                a = np.asarray(s.embeddings, dtype=np.float32)
                X.append(a)
                Y.append(np.tile(np.asarray(s.target_array, dtype=np.float32), (len(a), 1)))
            out[f'X::{f}'] = np.concatenate(X)
            out[f'Y::{f}'] = np.concatenate(Y)
            print(emb, f, out[f'X::{f}'].shape, 'buzz', int(out[f'Y::{f}'][:, ibuzz].sum()), flush=True)
    np.savez(f'.cache/{emb}.npz', ibuzz=ibuzz, classes=np.array(classes), **out)
    print('wrote', emb, flush=True)
