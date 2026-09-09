"""Offline readout sweep on cached embeddings.

Same metric (03_train/metrics.py:sens_at_fpr) and same leave-one-fold-out
rotation as 03_train, but the head is sklearn logistic regression, so a
configuration costs seconds instead of ~9 min. This ranks *transforms*; only
the winner earns a real CV. YAMNet is the control.

Loop order is (held-out fold) outer, (config) inner, so the 43-fold training
matrix is built once per fold rather than once per config -- the first version
of this rebuilt it 125 times and was OOM-killed.
"""
import gc, sys, warnings, numpy as np
sys.path[:0] = ['03_train', '.']
from metrics import sens_at_fpr
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

FPR = 0.005
ROT = ['JamesU - MustardBumbler/1_29',
       'Lily - Fit+Fast/2023_R3_Marysville/53',
       'Lily Adam - One Hive/recorders/willard/2024-08-07/1_11',
       'Luke - Diel Drivers/2026-04-08/1_150',
       'Luke - Diel Drivers/2026-05-06/1_95']
CS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
KS = [16, 32, 64, 128, 256]


def configs():
    for C in CS:
        yield ('raw', None, C)
    for C in CS:
        yield ('std', None, C)
    for k in KS:
        for C in CS:
            yield ('pca', k, C)


def main(emb):
    z = np.load(f'.cache/{emb}.npz', allow_pickle=True)
    ib = int(z['ibuzz'])
    folds = sorted({k[3:] for k in z.files if k.startswith('X::')})
    assert all(f in folds for f in ROT)

    res = {c: {} for c in configs()}
    for held in ROT:
        Xtr = np.concatenate([z[f'X::{f}'] for f in folds if f != held])
        ytr = np.concatenate([z[f'Y::{f}'][:, ib] for f in folds if f != held]).astype(bool)
        Xva = z[f'X::{held}']
        yva = z[f'Y::{held}'][:, ib].astype(bool)

        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        Ztr, Zva = (Xtr - mu) / sd, (Xva - mu) / sd

        # one eigendecomposition per fold, shared by every k and C
        C_ = np.cov(Ztr.astype(np.float64), rowvar=False)
        w, V = np.linalg.eigh(C_)
        order = np.argsort(w)[::-1]
        del C_

        for mode, k, Cval in configs():
            if mode == 'raw':
                A, B = Xtr, Xva
            elif mode == 'std':
                A, B = Ztr, Zva
            else:
                idx = order[:k]
                W = (V[:, idx] / np.sqrt(np.maximum(w[idx], 1e-8))).astype(np.float32)
                A, B = Ztr @ W, Zva @ W
            clf = LogisticRegression(C=Cval, max_iter=2000, solver='lbfgs',
                                     class_weight='balanced')
            clf.fit(A, ytr)
            res[(mode, k, Cval)][held] = sens_at_fpr(clf.decision_function(B), yva, [FPR])[FPR]
            if mode == 'pca':
                del A, B
            del clf
            gc.collect()
        del Xtr, ytr, Ztr, Zva, V, w
        gc.collect()
        print(f'  [{emb}] done {held}', flush=True)

    short = {f: f.split('/')[0][:11] + '/' + f.split('/')[-1] for f in ROT}
    print(f'\n=== {emb} ' + '=' * 70)
    print(f'{"mode":>5} {"k":>5} {"C":>7} {"mean":>7}   ' +
          '  '.join(f'{short[f][:14]:>14}' for f in ROT))
    best = None
    for c in configs():
        r = res[c]
        m = float(np.nanmean([r[f] for f in ROT]))
        print(f'{c[0]:>5} {str(c[1]):>5} {c[2]:>7.4g} {m:>7.3f}   ' +
              '  '.join(f'{r[f]:>14.3f}' for f in ROT))
        if best is None or m > best[0]:
            best = (m, c)
    print(f'  BEST {emb}: {best[1]} -> {best[0]:.3f}')


if __name__ == '__main__':
    main(sys.argv[1])
