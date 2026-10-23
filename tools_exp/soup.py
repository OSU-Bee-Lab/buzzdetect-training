"""Weight-space combinations of CV runs trained with TRUNK_SAVE_WEIGHTS=1, scored
on each held-out fold exactly like a trained rotation (train._score_fold), then
summarised by sx.py into folds_sx.csv. No training.

    python tools_exp/soup.py <name> --members <model dir> [<model dir> ...] [--wise A] [--bn-pool N]

Per fold: the members' fold.weights.h5 are averaged elementwise (uniform soup;
one member is just that member). --wise A then moves the trunk tail a fraction A
of the way back to the pretrained YAMNet layers (WiSE-FT: tail = A*pretrained +
(1-A)*soup); the head (Dense) keeps the soup's weights, since it has no
pretrained counterpart. The frozen BN moving statistics are identical across
members and the pretrained model, so every combination leaves them unchanged.

--bn-pool N (IDEAS #37a, pooled AdaBN) then re-estimates every tail BN layer's
moving mean/variance, in graph order, on N frames drawn at random from the
rotation's own training folds (`folds_train` in the fold's config_model.json,
asserted not to hold the held-out fold), every view of each frame, so later
layers see the earlier layers' new statistics. The held-out fold is never read.

--bn-site N (IDEAS #37b, exp/bn-site-d8, site-adaptive AdaBN) instead re-estimates
them on up to N frames of the held-out fold itself (labels unused), every view.
--bn-shrink N0 (IDEAS #37c) with --bn-site blends each layer's site statistics with
the model's own (on a TRUNK_BN_TRAIN model, the training pool's), weight
lam = n_site / (n_site + N0) on the site: mean = lam*m_s + (1-lam)*m_p, variance
the two-component mixture's, lam*v_s + (1-lam)*v_p + lam*(1-lam)*(m_s-m_p)^2.
"""
# TensorFlow imported first — see 03_train/main.py for rationale.
import tensorflow as tf  # noqa: F401
import argparse
import json
import os
import shutil
import zlib
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
from dataset import build_fold_dataset  # noqa: E402
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


def pool_maps(setname, embeddername, folds_train, translation, n, seed, map_shape):
    """n frames drawn uniformly from the training pool, reshaped to one cut-layer
    map per view: (n * n_views, *map_shape), float16."""
    per_fold = []
    for f in folds_train:
        per_fold.append(build_fold_dataset(cfg.dir_embeddings_fold(setname, embeddername, f), translation))
    sizes = np.array([sum(s.frames for s in samples) for samples in per_fold])
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(sizes.sum(), size=min(n, sizes.sum()), replace=False))
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    out = []
    for samples, a, size in zip(per_fold, starts, sizes):
        local = pick[(pick >= a) & (pick < a + size)] - a
        if not len(local):
            continue
        emb = np.concatenate([np.asarray(s.embeddings, dtype=np.float16) for s in samples])
        out.append(emb[local])
    x = np.concatenate(out)
    return x.reshape((-1,) + tuple(map_shape))


def _chain(tail):
    """The tail's layers in order. YAMNet's tail is a plain chain, but its layers
    also belong to the full YAMNet graph, so a sub-model cut at a BN input is
    ambiguous; applying them in turn is not. Checked against tail() itself."""
    return [l for l in tail.layers if not isinstance(l, keras.layers.InputLayer)]


def _apply(layer, x, batch):
    return np.concatenate([layer(x[i:i + batch], training=False).numpy()
                           for i in range(0, len(x), batch)])


def reestimate_bn(tail, maps, batch=512, lam=1.0):
    """AdaBN: each BatchNormalization layer's moving stats := its input's per-channel
    mean/variance over `maps`, in graph order, so each sees the updated layers below.
    lam < 1 shrinks them toward the layer's current stats (mixture moments)."""
    layers = _chain(tail)
    x = maps[:64].astype(np.float32)
    ref = tail(x, training=False).numpy()
    for l in layers:
        x = l(x, training=False).numpy()
    assert np.allclose(x, ref, atol=1e-4), 'tail is not a plain layer chain'
    x = maps.astype(np.float32)
    k = 0
    for l in layers:
        if isinstance(l, keras.layers.BatchNormalization):
            h = x.reshape(-1, x.shape[-1]).astype(np.float64)
            mean, var = h.mean(0), h.var(0)
            m_p, v_p = l.moving_mean.numpy().astype(np.float64), l.moving_variance.numpy().astype(np.float64)
            if lam < 1.0:
                var = lam * var + (1 - lam) * v_p + lam * (1 - lam) * (mean - m_p) ** 2
                mean = lam * mean + (1 - lam) * m_p
            shift = np.abs(mean - m_p).mean()
            l.moving_mean.assign(mean.astype(np.float32))
            l.moving_variance.assign(var.astype(np.float32))
            print(f'    {l.name}: mean |shift| {shift:.3f}', flush=True)
            k += 1
        x = _apply(l, x, batch)
    return k


def build(embedder, n_classes, dropout):
    # learning rates only shape the optimizer, which scoring never touches
    return embedder.build_head(n_classes, lr_backbone=1e-5, lr_head=2e-4, dropout=dropout)


def site_maps(setname, embeddername, fold, translation, n, seed, map_shape):
    """Up to n frames of the held-out fold itself (labels unused), one map per view."""
    samples = build_fold_dataset(cfg.dir_embeddings_fold(setname, embeddername, fold), translation)
    emb = np.concatenate([np.asarray(s.embeddings, dtype=np.float16) for s in samples])
    if len(emb) > n:
        emb = emb[np.sort(np.random.default_rng(seed).choice(len(emb), size=n, replace=False))]
    return emb.reshape((-1,) + tuple(map_shape)), len(emb)


def main(name, members, wise, bn_pool, bn_site=0, bn_shrink=0):
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
        if bn_pool:
            with open(os.path.join(members[0], 'folds', fold, 'config_model.json')) as f:
                folds_train = json.load(f)['folds_train']
            assert fold not in folds_train, fold
            tail = find_tail(model)
            maps = pool_maps(setname, embeddername, folds_train, translation, bn_pool,
                             seed=zlib.crc32(fold.encode()), map_shape=tail.input.shape[1:])
            k = reestimate_bn(tail, maps)
            print(f'[{name}] {fold}: {k} BN layers re-estimated on {len(maps)} maps '
                  f'from {len(folds_train)} training folds', flush=True)
            del maps
        if bn_site:
            tail = find_tail(model)
            maps, n_site = site_maps(setname, embeddername, fold, translation, bn_site,
                                     seed=zlib.crc32(fold.encode()), map_shape=tail.input.shape[1:])
            lam = n_site / (n_site + bn_shrink) if bn_shrink else 1.0
            k = reestimate_bn(tail, maps, lam=lam)
            print(f'[{name}] {fold}: {k} BN layers re-estimated on {n_site} held-out frames '
                  f'({len(maps)} maps), lam {lam:.3f}', flush=True)
            del maps
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
        f.write('\n'.join(os.path.abspath(m) for m in members) + f'\nwise {wise}\nbn_pool {bn_pool}\n'
                f'bn_site {bn_site}\nbn_shrink {bn_shrink}\n')
    print(format_sx_report(name, sx))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--members', nargs='+', required=True)
    ap.add_argument('--wise', type=float, default=0.0)
    ap.add_argument('--bn-pool', type=int, default=0)
    ap.add_argument('--bn-site', type=int, default=0)
    ap.add_argument('--bn-shrink', type=int, default=0)
    a = ap.parse_args()
    assert not (a.bn_pool and a.bn_site), '--bn-pool and --bn-site are alternatives'
    assert a.bn_site or not a.bn_shrink, '--bn-shrink needs --bn-site'
    main(a.name, a.members, a.wise, a.bn_pool, a.bn_site, a.bn_shrink)
