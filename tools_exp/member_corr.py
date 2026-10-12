"""Correlation of the ins_buzz logit between ensemble members, split by `correct`, pooled over folds.

usage: member_corr.py <model dir> <model dir> [<model dir> ...]

Reads each member's per-fold predictions.csv (rows must match across members), concatenates all held-out
frames, and prints, for buzz frames and for negatives: the mean pairwise Pearson correlation, the
between-member SD (root mean within-frame variance) and the across-frame SD of the member mean.
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ['correct', 'loudness', 'sample']


def load(model):
    root = Path(model) / 'folds'
    return {str(p.parent.relative_to(root)): pd.read_csv(p) for p in sorted(root.rglob('predictions.csv'))}


def main(models):
    members = [load(m) for m in models]
    folds = sorted(members[0])
    for m, d in zip(models, members):
        if sorted(d) != folds:
            sys.exit(f'{m}: folds differ from {models[0]}')
    logits, correct = [], []
    for fold in folds:
        ref = members[0][fold]
        for m, d in zip(models, members):
            if not d[fold][KEYS].equals(ref[KEYS]):
                sys.exit(f'{m}: rows differ in {fold}')
        logits.append(np.stack([d[fold]['activation_ins_buzz'].to_numpy() for d in members], axis=1))
        correct.append(ref['correct'].to_numpy().astype(bool))
    logits, correct = np.concatenate(logits), np.concatenate(correct)
    print(f'{len(models)} members, {len(folds)} folds, {len(logits)} frames')
    for label, mask in (('buzz (correct)', correct), ('negatives', ~correct)):
        x = logits[mask]
        r = [np.corrcoef(x[:, i], x[:, j])[0, 1] for i, j in combinations(range(x.shape[1]), 2)]
        between = np.sqrt(x.var(axis=1, ddof=1).mean())
        across = x.mean(axis=1).std(ddof=1)
        print(f'{label}: n={mask.sum()}  pairwise r mean {np.mean(r):.3f} '
              f'({", ".join(f"{v:.3f}" for v in r)})  between-member SD {between:.2f}  across-frame SD {across:.2f}')


if __name__ == '__main__':
    main(sys.argv[1:])
