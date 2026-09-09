"""Raw labels of the negatives above each fold's own fpr0.005 threshold."""
import sys, glob, os
import numpy as np, pandas as pd
from collections import Counter

def census(model_dir):
    sx = pd.read_csv(os.path.join(model_dir, 'folds_sx.csv'))
    sx = sx[(sx.fpr == 0.005) & (sx.fold != 'total')]
    tot, per_fold = Counter(), {}
    buzz_stats = {}
    for _, r in sx.iterrows():
        fold, thr = r['fold'], r['threshold']
        if not np.isfinite(thr):
            continue
        files = glob.glob(os.path.join(model_dir, 'surprisal', fold, '*_surprisal.csv'))
        if not files:
            continue
        d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        p = d['activation_ins_buzz'].clip(1e-9, 1 - 1e-9)
        z = np.log(p / (1 - p))
        is_buzz = d['label'].astype(str).str.startswith('ins_buzz')
        neg_hot = d[(~is_buzz) & (z > thr)]
        c = Counter(neg_hot['label'].astype(str))
        per_fold[fold] = c
        tot.update(c)
        buzz_stats[fold] = (z[is_buzz].mean(), z[~is_buzz].mean())
    return tot, per_fold, buzz_stats

for md in sys.argv[1:]:
    tot, per_fold, bs = census(md)
    print(f'=== {os.path.basename(md)}  (n above threshold = {sum(tot.values())})')
    for lab, n in tot.most_common(6):
        print(f'   {lab:28s} {n}')
    mech = sum(v for k, v in tot.items() if k.startswith('mech_auto'))
    print(f'   -> mech_auto* total: {mech} / {sum(tot.values())}')
    print('   mean logit  buzz / non-buzz frames:')
    for f, (b, n) in bs.items():
        print(f'     {f[-22:]:24s} {b:7.3f} {n:7.3f}')
