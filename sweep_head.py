"""Extended offline readout sweep: does a non-linear / more-regularised head
over frozen yamnet_aves (1792-d concat) beat the converged LINEAR readout, and
close the gap to / beat converged YAMNet-MLP?

Decision gate for 'a different head on yamnet_aves'. sklearn, per-dim
standardized input, run to convergence. Same metric + rotation as 03_train.
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

CONFIGS = [
    ('lr',   None,       1.0),
    ('lr',   None,       0.1),   # stronger L2
    ('mlp',  (256,),     1e-3),
    ('mlp',  (256,),     1e-1),
]


def make(mode, h, a):
    if mode == 'lr':
        return LogisticRegression(C=a, max_iter=6000, solver='lbfgs',
                                  class_weight='balanced')
    return MLPClassifier(hidden_layer_sizes=h, alpha=a, batch_size=2048,
                         max_iter=300, early_stopping=True, n_iter_no_change=15,
                         random_state=0)


def main(emb):
    z = np.load(f'.cache/{emb}.npz', allow_pickle=True)
    ib = int(z['ibuzz'])
    folds = sorted({k[3:] for k in z.files if k.startswith('X::')})
    assert all(f in folds for f in ROT)

    res = {c: {} for c in CONFIGS}
    for held in ROT:
        Xtr = np.concatenate([z[f'X::{f}'] for f in folds if f != held])
        ytr = np.concatenate([z[f'Y::{f}'][:, ib] for f in folds if f != held]).astype(bool)
        Xva = z[f'X::{held}']
        yva = z[f'Y::{held}'][:, ib].astype(bool)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        Ztr, Zva = (Xtr - mu) / sd, (Xva - mu) / sd
        for c in CONFIGS:
            clf = make(*c)
            clf.fit(Ztr, ytr)
            s = clf.decision_function(Zva) if c[0] == 'lr' else clf.predict_proba(Zva)[:, 1]
            res[c][held] = sens_at_fpr(s, yva, [FPR])[FPR]
            del clf; gc.collect()
        del Xtr, ytr, Ztr, Zva; gc.collect()
        print(f'  [{emb}] done {held}', flush=True)

    short = {f: f.split('/')[0][:10] + '/' + f.split('/')[-1] for f in ROT}
    print(f'\n=== {emb} ' + '=' * 66)
    print(f'{"mode":>5} {"hidden":>11} {"reg":>7} {"mean":>7}   ' +
          '  '.join(f'{short[f][:13]:>13}' for f in ROT))
    for c in CONFIGS:
        r = res[c]
        m = float(np.nanmean([r[f] for f in ROT]))
        print(f'{c[0]:>5} {str(c[1]):>11} {c[2]:>7.3g} {m:>7.3f}   ' +
              '  '.join(f'{r[f]:>13.3f}' for f in ROT))


if __name__ == '__main__':
    for e in sys.argv[1:]:
        main(e)
