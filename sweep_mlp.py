"""Offline NON-LINEAR readout sweep on cached AVES embeddings.

The in-pipeline --hidden CV (aves_h256/h1024) came back negative, but under
cv-baseline's YAMNet-tuned head config: val_loss early-stop, ~2 grad steps
per epoch, no weight decay (see aves-readout: the probe is undertrained, not
overfit). This asks the decision-gate question cleanly: does a *converged*
non-linear readout of frozen AVES beat the converged linear readout (0.194,
aves-readout) and close the gap to YAMNet (~0.26)?

Same metric and rotation as 03_train. sklearn MLPClassifier, standardized
input, L2 (alpha) as the only regulariser, run to convergence.
"""
import gc, sys, warnings, numpy as np
sys.path[:0] = ['03_train', '.']
from metrics import sens_at_fpr
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

FPR = 0.005
ROT = ['JamesU - MustardBumbler/1_29',
       'Lily - Fit+Fast/2023_R3_Marysville/53',
       'Lily Adam - One Hive/recorders/willard/2024-08-07/1_11',
       'Luke - Diel Drivers/2026-04-08/1_150',
       'Luke - Diel Drivers/2026-05-06/1_95']

ALPHAS = [1e-3, 1e-1]
HIDDEN = [(256,)]


def configs():
    yield ('lr', None, 1.0)
    for h in HIDDEN:
        for a in ALPHAS:
            yield ('mlp', h, a)


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

        for mode, h, a in configs():
            if mode == 'lr':
                clf = LogisticRegression(C=1.0, max_iter=4000, solver='lbfgs',
                                         class_weight='balanced')
            else:
                clf = MLPClassifier(hidden_layer_sizes=h, alpha=a,
                                    batch_size=2048, max_iter=120,
                                    early_stopping=True, n_iter_no_change=10,
                                    random_state=0, verbose=False)
            clf.fit(Ztr, ytr)
            score = clf.decision_function(Zva) if mode == 'lr' else clf.predict_proba(Zva)[:, 1]
            res[(mode, h, a)][held] = sens_at_fpr(score, yva, [FPR])[FPR]
            del clf
            gc.collect()
        del Xtr, ytr, Ztr, Zva
        gc.collect()
        print(f'  [{emb}] done {held}', flush=True)

    short = {f: f.split('/')[0][:11] + '/' + f.split('/')[-1] for f in ROT}
    print(f'\n=== {emb} ' + '=' * 70)
    print(f'{"mode":>5} {"hidden":>12} {"alpha":>7} {"mean":>7}   ' +
          '  '.join(f'{short[f][:14]:>14}' for f in ROT))
    for c in configs():
        r = res[c]
        m = float(np.nanmean([r[f] for f in ROT]))
        print(f'{c[0]:>5} {str(c[1]):>12} {c[2]:>7.4g} {m:>7.3f}   ' +
              '  '.join(f'{r[f]:>14.3f}' for f in ROT))


if __name__ == '__main__':
    main(sys.argv[1])
