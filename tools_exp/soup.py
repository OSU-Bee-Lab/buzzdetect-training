"""Weight-space combinations of CV runs trained with TRUNK_SAVE_WEIGHTS=1, scored
on each held-out fold exactly like a trained rotation (train._score_fold), then
summarised by sx.py into folds_sx.csv. No training.

    python tools_exp/soup.py <name> --members <model dir> [<model dir> ...] [--wise A]

Per fold: the members' fold.weights.h5 are averaged elementwise (uniform soup;
one member is just that member). --wise A then moves the trunk tail a fraction A
of the way back to the pretrained YAMNet layers (WiSE-FT: tail = A*pretrained +
(1-A)*soup); the head (Dense) keeps the soup's weights, since it has no
pretrained counterpart. The frozen BN moving statistics are identical across
members and the pretrained model, so every combination leaves them unchanged.
"""
# TensorFlow imported first — see 03_train/main.py for rationale.
import tensorflow as tf  # noqa: F401
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, '03_train'))

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as cfg  # noqa: E402
from embedders.embedding import load_embedder  # noqa: E402
from train import _score_fold, FNAME_FOLD_WEIGHTS  # noqa: E402
from train_utils import build_classes  # noqa: E402
from thresholds import FNAME_PREDICTIONS_CLASSES  # noqa: E402
from sx import (summarize_folds, format_sx_report, read_fold_predictions,  # noqa: E402
                FNAME_SX_SUMMARY, FNAME_PREDICTIONS)


def find_tail(model):
    """The shared YAMNet tail sub-model (wrapped in TimeDistributed for n_ctx > 1)."""
    for layer in model.layers:
        inner = getattr(layer, 'layer', layer)
        if isinstance(inner, keras.Model):
            return inner
    raise ValueError('no tail sub-model found')


def build(embedder, n_classes, dropout):
    # learning rates only shape the optimizer, which scoring never touches
    return embedder.build_head(n_classes, lr_backbone=1e-5, lr_head=2e-4, dropout=dropout)


def main(name, members, wise):
    folds = sorted(read_fold_predictions(os.path.join(members[0], 'folds'))['fold'].unique())
    with open(os.path.join(members[0], 'folds', folds[0], 'config_model.json')) as f:
        c0 = json.load(f)
    setname, embeddername = c0['set'], c0['embeddername']
    translation = pd.read_csv(cfg.path_translation(setname, c0['translation']))
    classes = build_classes(translation)
    assert classes == c0['classes'], (classes, c0['classes'])
    embedder = load_embedder(embeddername, framehop_prop=1, initialize=False)

    dir_model = os.path.join(cfg.DIR_MODELS, name)
    for fold in folds:
        d_out = os.path.join(dir_model, 'folds', fold)
        if os.path.exists(os.path.join(d_out, FNAME_PREDICTIONS)):
            print(f'[{name}] {fold}: done; skipping', flush=True)
            continue
        paths = [os.path.join(m, 'folds', fold, FNAME_FOLD_WEIGHTS) for m in members]
        for p, m in zip(paths, members):
            with open(os.path.join(os.path.dirname(p), 'config_model.json')) as f:
                cm = json.load(f)
            assert (cm['embeddername'], cm['set'], cm['held_out_fold']) == (embeddername, setname, fold), (m, fold)
            if not os.path.exists(p):
                sys.exit(f'{p}: missing (member not trained with TRUNK_SAVE_WEIGHTS=1)')
        model = build(embedder, len(classes), c0['dropout'])
        pretrained_tail = [w.copy() for w in find_tail(model).get_weights()]
        stack = []
        for p in paths:
            model.load_weights(p)
            stack.append(model.get_weights())
        model.set_weights([np.mean([s[i] for s in stack], axis=0) for i in range(len(stack[0]))])
        if wise:
            tail = find_tail(model)
            tail.set_weights([wise * w0 + (1 - wise) * w for w0, w in zip(pretrained_tail, tail.get_weights())])
        predictions, predictions_classes = _score_fold(model, setname, embeddername, fold, translation, classes)
        os.makedirs(d_out, exist_ok=True)
        predictions.to_csv(os.path.join(d_out, FNAME_PREDICTIONS), index=False)
        predictions_classes.to_csv(os.path.join(d_out, FNAME_PREDICTIONS_CLASSES), index=False)
        shutil.copy(os.path.join(members[0], 'folds', fold, 'summary.json'), d_out)
        print(f'[{name}] {fold}: scored, buzz logit SD {predictions["activation_ins_buzz"].std():.2f}', flush=True)
        del model
        tf.keras.backend.clear_session()

    sx = summarize_folds(read_fold_predictions(os.path.join(dir_model, 'folds')))
    sx.to_csv(os.path.join(dir_model, FNAME_SX_SUMMARY), index=False)
    with open(os.path.join(dir_model, 'members.txt'), 'w') as f:
        f.write('\n'.join(os.path.abspath(m) for m in members) + f'\nwise {wise}\n')
    print(format_sx_report(name, sx))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--members', nargs='+', required=True)
    ap.add_argument('--wise', type=float, default=0.0)
    a = ap.parse_args()
    main(a.name, a.members, a.wise)
