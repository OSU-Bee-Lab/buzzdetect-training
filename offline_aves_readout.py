# TensorFlow imported first — see 03_train/main.py for rationale.
"""Offline readout check on the cached AVES columns of yamnet_trunk_pitchshift_depth8_aves.

LOOP.md, "A new embedder doesn't inherit the probe config": before ps-d8-aves's null is logged,
ask whether ps-depth8's head regime (raw columns, Adam lr 2e-4, 30 epochs, label smoothing 0.2)
can read the frozen AVES block at all. Buzz-vs-rest logistic readouts on the AVES columns alone,
one per rotating fold, scored with metrics.sens_at_fpr (quiet/faint buzz frames left out, as the
headline does). No trunk, no Keras; one diagnostic sweep.
"""
import tensorflow as tf  # noqa: F401
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ROOT, os.path.join(ROOT, '03_train')]
import config as cfg  # noqa: E402
from dataset import build_fold_dataset, read_fold_roles, folds_by_role, ROLE_TRAIN, ROLE_ROTATE  # noqa: E402
from metrics import sens_at_fpr  # noqa: E402
from train_utils import build_classes, buzz_tier, TIERS_EXCLUDED_FROM_HEADLINE  # noqa: E402

SET, EMB, N_AVES, FPR = 'medium', 'yamnet_trunk_pitchshift_depth8_aves', 2304, 0.005
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

translation = pd.read_csv(cfg.path_translation(SET, 'general'))
buzz = build_classes(translation).index('ins_buzz')
roles = read_fold_roles(SET, EMB)
rotate, train_only = folds_by_role(roles, ROLE_ROTATE), folds_by_role(roles, ROLE_TRAIN)

X, y, scored_out, fold_of = [], [], [], []
for fi, fold in enumerate(rotate + train_only):
    samples = build_fold_dataset(cfg.dir_embeddings_fold(SET, EMB, fold), translation)
    for s in samples:
        X.append(np.asarray(s.embeddings, dtype=np.float16)[:, -N_AVES:].astype(np.float32))
        is_buzz = bool(s.target_array[buzz])
        y.append(np.full(s.frames, is_buzz))
        dropped = is_buzz and buzz_tier(s.labels_raw, s.labels_translate) in TIERS_EXCLUDED_FROM_HEADLINE
        scored_out.append(np.full(s.frames, dropped))
        fold_of.append(np.full(s.frames, fi))
        s.embeddings = None
    del samples
X, y, scored_out, fold_of = map(np.concatenate, (X, y, scored_out, fold_of))
print(f'{len(X)} frames, {int(y.sum())} buzz, {len(rotate)} rotating folds; AVES columns: '
      f'mean {X.mean():.3f}, mean |x| {np.abs(X).mean():.3f}, column SD median {np.median(X.std(0)):.3f}, '
      f'row norm median {np.median(np.linalg.norm(X, axis=1)):.1f}', flush=True)
Xt, yt = torch.tensor(X, device=dev), torch.tensor(y, dtype=torch.float32, device=dev)


def fit(tr, standardize, mode, l2=0.0):
    """Logistic readout on rows `tr`; returns a scoring function."""
    mu = Xt[tr].mean(0) if standardize else torch.zeros(N_AVES, device=dev)
    sd = Xt[tr].std(0) + 1e-6 if standardize else torch.ones(N_AVES, device=dev)
    A, b = (Xt[tr] - mu) / sd, yt[tr]
    torch.manual_seed(0)
    lin = torch.nn.Linear(N_AVES, 1).to(dev)
    if mode == 'head':       # ps-depth8's regime: Adam 2e-4, 30 epochs, batch 1024, smoothing 0.2
        opt = torch.optim.Adam(lin.parameters(), lr=2e-4)
        bs = b * 0.8 + 0.1
        for _ in range(30):
            perm = torch.randperm(len(A), device=dev)
            for i in range(0, len(A), 1024):
                j = perm[i:i + 1024]
                opt.zero_grad()
                torch.nn.functional.binary_cross_entropy_with_logits(lin(A[j]).squeeze(1), bs[j]).backward()
                opt.step()
    else:                    # converged ridge logistic
        opt = torch.optim.LBFGS(lin.parameters(), max_iter=300, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(lin(A).squeeze(1), b)
                    + l2 * (lin.weight ** 2).sum())
            loss.backward()
            return loss
        opt.step(closure)
    return lambda rows: lin((Xt[rows] - mu) / sd).squeeze(1).detach().cpu().numpy()


configs = [('raw, head regime (Adam 2e-4 x30, ls 0.2)', False, 'head', 0.0),
           ('standardized, head regime', True, 'head', 0.0)]
configs += [(f'standardized, ridge logistic l2={l2:g}', True, 'ridge', l2) for l2 in (1e-4, 1e-3, 1e-2, 1e-1)]

rows = []
for label, standardize, mode, l2 in configs:
    sens, sds = [], []
    for fi in range(len(rotate)):
        tr = torch.tensor(np.flatnonzero(fold_of != fi), device=dev)
        va = np.flatnonzero((fold_of == fi) & ~scored_out)
        act = fit(tr, standardize, mode, l2)(torch.tensor(va, device=dev))
        sens.append(sens_at_fpr(act, y[va], [FPR])[FPR])
        sds.append(float(act.std()))
    rows.append((label, float(np.nanmean(sens)), float(np.mean(sds)), sens))
    print(f'{label}: mean sens@{FPR} {rows[-1][1]:.3f}, logit SD {rows[-1][2]:.2f} | '
          + ' '.join(f'{os.path.basename(f)} {s:.3f}' for f, s in zip(rotate, sens)), flush=True)
