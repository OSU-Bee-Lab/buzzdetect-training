# TensorFlow imported first — see 03_train/main.py for rationale.
import json
import os
import pickle
import re
import shutil
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import tensorflow as tf

import config as cfg

from dataset import (
    build_fold_dataset, load_augmented, read_fold_roles, folds_by_role,
    ROLE_TRAIN, ROLE_ROTATE, ROLE_HOLDOUT,
)
from train_utils import build_weights, build_classes, can_write, Sample
from embedders.embedding import load_embedder
from plot_history import plot_history
from write_model_py import write_model_py

# metrics_by_group/metrics_at_fpr are the same threshold-sweep logic 04_test
# used against its on-disk results; reused here in-memory.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '04_test'))
from metrics import metrics_by_group, metrics_at_fpr

FNAME_PREDICTIONS = 'predictions.csv'
FNAME_FOLD_SUMMARY = 'summary.json'
FNAME_FOLDS_SUMMARY = 'folds_summary.csv'
FNAME_POOLED_METRICS = 'folds_pooled_metrics.csv'
FNAME_POOLED_SX = 'folds_pooled_sx.csv'
SUBDIR_FOLDS = 'folds'
SUBDIR_HOLDOUT = 'holdout'


@dataclass
class TrainingData:
    """One CV iteration's worth of data, plus everything derived from it that
    the training call and its artifacts need."""
    train_tf: tf.data.Dataset
    val_tf: tf.data.Dataset
    classes: list
    weight_dict: dict
    weights: pd.DataFrame
    translation: pd.DataFrame
    size_batch: int
    size_shuffle: int
    folds_train: list = field(default_factory=list)
    frames_train: int = 0
    frames_val: int = 0
    val_fold: str = None


def _to_tf(data, size_batch, size_shuffle):
    embeddings, targets = [], []
    for s in data:
        embeddings.extend(s.embeddings)
        targets.extend([s.target_array] * s.frames)
    idx = np.random.permutation(len(embeddings))
    emb_np = np.array(embeddings, dtype=np.float32)[idx]
    tgt_np = np.array(targets, dtype=np.float32)[idx]
    return (
        tf.data.Dataset.from_tensor_slices((emb_np, tgt_np))
        .cache().shuffle(size_shuffle).batch(size_batch).prefetch(tf.data.AUTOTUNE)
    )


def _load_data(setname, embeddername, folds_train, name_translation, aug_dirnames,
               val_fold=None):
    """Pool folds_train for training; val_fold, if given, is a whole separate
    deployment used as the early-stopping monitor.

    Validation is always a whole fold, never a split within one. Snips from a
    deployment share a recorder, a site and a background, so a within-fold
    split would leak site identity into the stopping signal and bias the
    stopping epoch late. val_fold=None means no monitor at all — the caller
    fixes the epoch count instead.

    Augmented embeddings go to training only.
    """
    translation = pd.read_csv(os.path.join(cfg.DIR_TRANSLATIONS, name_translation + '.csv'))
    classes = build_classes(translation)

    data_train = []
    for fold in folds_train:
        data_train += build_fold_dataset(
            cfg.dir_embeddings_fold(setname, embeddername, fold), translation,
        )
    frames_train = sum(s.frames for s in data_train)

    data_val = None
    frames_val = 0
    if val_fold is not None:
        data_val = build_fold_dataset(
            cfg.dir_embeddings_fold(setname, embeddername, val_fold), translation,
        )
        frames_val = sum(s.frames for s in data_val)

    if aug_dirnames:
        data_train += load_augmented(setname, embeddername, aug_dirnames, translation, folds_train)

    weights = build_weights(data_train, classes)
    weight_dict = {i: w for i, w in enumerate(weights['weight'])}

    size_batch = 65568
    size_shuffle = 10 * size_batch

    return TrainingData(
        train_tf=_to_tf(data_train, size_batch, size_shuffle),
        val_tf=_to_tf(data_val, size_batch, size_shuffle) if data_val is not None else None,
        classes=classes,
        weight_dict=weight_dict,
        weights=weights,
        translation=translation,
        size_batch=size_batch,
        size_shuffle=size_shuffle,
        folds_train=list(folds_train),
        frames_train=frames_train,
        frames_val=frames_val,
        val_fold=val_fold,
    )


def _score_fold(model, setname, embeddername, fold, translation, classes):
    """Score a trained model on a fold it never saw, ins_buzz only.

    Same threshold sweep 04_test computes from on-disk inference results, run
    here straight off the fold's known frame labels. Returns
    (metrics_df, predictions_df); predictions are kept so the per-fold results
    can be pooled into one ROC afterwards.
    """
    samples = build_fold_dataset(
        cfg.dir_embeddings_fold(setname, embeddername, fold), translation,
    )
    buzz_index = classes.index('ins_buzz')

    embeddings = np.concatenate([np.array(s.embeddings, dtype=np.float32) for s in samples])
    correct = np.concatenate([np.full(s.frames, bool(s.target_array[buzz_index])) for s in samples])
    activation = model(embeddings, training=False)[:, buzz_index].numpy()

    predictions = pd.DataFrame({'activation_ins_buzz': activation, 'correct': correct})
    return metrics_by_group(predictions), predictions


def _sens_by_fpr(metrics_df):
    return metrics_at_fpr(metrics_df).set_index('fpr')['sensitivity']


def _format_sens(sens):
    return ', '.join(
        f'sens@fpr{f:.1%}={sens[f]:.3f}' if pd.notna(sens[f]) else f'sens@fpr{f:.1%}=n/a'
        for f in sens.index
    )


def _write_scores(dir_out, model, setname, embeddername, fold, translation, classes, tag):
    """Score `fold`, write metrics/sx/predictions under dir_out, return
    {sens_fpr<x>: value} for the summary table plus the raw predictions."""
    os.makedirs(dir_out, exist_ok=True)
    metrics_df, predictions = _score_fold(model, setname, embeddername, fold, translation, classes)

    metrics_df.to_csv(os.path.join(dir_out, cfg.FNAME_METRICS), index=False)
    sx_df = metrics_at_fpr(metrics_df)
    sx_df.to_csv(os.path.join(dir_out, cfg.FNAME_SX), index=False)
    predictions.to_csv(os.path.join(dir_out, FNAME_PREDICTIONS), index=False)

    sens = sx_df.set_index('fpr')['sensitivity']
    print(f'[{tag}] fold {fold}: {_format_sens(sens)}')
    return {f'sens_fpr{f:g}': sens[f] for f in sens.index}, predictions


def _collect_fold_results(dir_folds, folds_rotate):
    """Assemble the CV summary and pooled predictions from what's on disk.

    Reading back rather than accumulating in memory keeps a partial rerun
    honest: folds skipped because they were already trained still contribute
    their row and their predictions, instead of being silently dropped from
    the summary.
    """
    summary_rows, predictions = [], []
    for fold in folds_rotate:
        dir_fold = os.path.join(dir_folds, str(fold))
        path_summary = os.path.join(dir_fold, FNAME_FOLD_SUMMARY)
        path_predictions = os.path.join(dir_fold, FNAME_PREDICTIONS)
        if not (os.path.exists(path_summary) and os.path.exists(path_predictions)):
            print(f'[{fold}] no complete result on disk; omitted from the CV summary')
            continue
        with open(path_summary) as f:
            summary_rows.append({'fold': fold, **json.load(f)})
        df = pd.read_csv(path_predictions)
        df['fold'] = fold
        predictions.append(df)

    return summary_rows, predictions


def _train_one(dir_model, modelname, embeddername, setname, name_translation,
               data: TrainingData, epochs_max, aug_dirnames, verbose,
               held_out_fold, save_binary, epochs_fixed=None, patience=50):
    """Train one model. Returns (result_row, model); (None, None) if the model
    directory is already populated."""
    if not can_write(dir_model):
        print(f'[{modelname}] already trained; skipping')
        return None, None

    monitor = (f'early stopping on {data.val_fold} ({data.frames_val} frames)'
               if data.val_fold else f'{epochs_fixed} fixed epochs, no monitor')
    print(f'[{modelname}] training on {len(data.folds_train)} fold(s), '
          f'{data.frames_train} frames; {monitor}...')
    os.makedirs(dir_model, exist_ok=True)

    embedder = load_embedder(embeddername, framehop_prop=1, initialize=False)
    # modelname may contain a fold identifier derived from a source name
    # (e.g. "2025-06-04 original annotations"), which can hold characters
    # invalid in a Keras/TF scope name — sanitize for that use only.
    tf_name = re.sub(r'[^A-Za-z0-9_.\\/>-]', '_', modelname)
    model = tf.keras.Sequential(name=tf_name)
    model.add(tf.keras.layers.Input(shape=(embedder.n_embeddings,), dtype=tf.float32, name='input'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(len(data.classes)))

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.2),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        metrics=['accuracy'],
    )

    if data.val_tf is None:
        # Shipped model: no fold is held out, so there's nothing clean to
        # monitor. Train a fixed number of epochs instead, set by the caller
        # from the median best epoch across the rotations.
        history = model.fit(
            data.train_tf, epochs=epochs_fixed, class_weight=data.weight_dict,
            verbose=1 if verbose else 0,
        )
        best_epoch = epochs_fixed - 1
        print(f'[{modelname}] done — {epochs_fixed} fixed epochs')
        result = {
            'n_epochs': epochs_fixed,
            'best_epoch': epochs_fixed,
            'frames_train': data.frames_train,
        }
    else:
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, min_delta=0.002, restore_best_weights=True,
        )
        history = model.fit(
            data.train_tf,
            epochs=epochs_max,
            validation_data=data.val_tf,
            callbacks=callback,
            class_weight=data.weight_dict,
            verbose=1 if verbose else 0,
        )

        best_epoch = callback.best_epoch
        best_val_loss = float(callback.best)
        print(f'[{modelname}] done — {len(history.history["val_loss"])} epochs, '
              f'best epoch {best_epoch + 1}, val_loss {best_val_loss:.4f}')
        result = {
            'n_epochs': len(history.history['val_loss']),
            'best_epoch': best_epoch + 1,
            'best_val_loss': best_val_loss,
            'best_val_accuracy': history.history['val_accuracy'][best_epoch],
            'frames_train': data.frames_train,
            'frames_val': data.frames_val,
        }

    if save_binary:
        model.save(os.path.join(dir_model, 'model.keras'), include_optimizer=True)

    with open(os.path.join(dir_model, 'history.pickle'), 'wb') as f:
        pickle.dump(history, f)

    data.weights.to_csv(os.path.join(dir_model, 'weights.csv'), index=False)
    data.translation.to_csv(os.path.join(dir_model, 'translation.csv'), index=False)

    config_model = {
        'embeddername': embeddername,
        'set': setname,
        'translation': name_translation,
        'classes': data.classes,
        'size_shuffle': data.size_shuffle,
        'size_batch': data.size_batch,
        'digits_results': 8,
        'aug_dirnames': aug_dirnames or [],
        'held_out_fold': held_out_fold,
        'folds_train': data.folds_train,
        'val_fold': data.val_fold,
        'epochs_fixed': epochs_fixed,
        'patience': patience,
    }
    # 'w' for the same reason as write_model_py's — can_write() is the gate
    with open(os.path.join(dir_model, 'config_model.json'), 'w') as f:
        f.write(json.dumps(config_model))

    plot_history(history, modelname, best_epoch, os.path.join(dir_model, 'loss_curves.svg'))
    if save_binary:
        write_model_py(dir_model, modelname, embeddername, config_model['digits_results'])

    return result, model


def train_set(name, embeddername, setname, name_translation,
              epochs_max=400, aug_dirnames=None, verbose=False, patience=50):
    roles = read_fold_roles(setname, embeddername)
    folds_rotate = folds_by_role(roles, ROLE_ROTATE)
    folds_train_always = folds_by_role(roles, ROLE_TRAIN)
    folds_holdout = folds_by_role(roles, ROLE_HOLDOUT)

    if len(folds_rotate) < 2:
        raise ValueError(
            f'set {setname!r} has {len(folds_rotate)} fold(s) with role '
            f'{ROLE_ROTATE!r}; cross-validation needs at least 2. Roles found: '
            f'{ {r: len(folds_by_role(roles, r)) for r in sorted(set(roles.values()))} }'
        )

    print(f'{len(folds_rotate)} rotating fold(s), {len(folds_train_always)} '
          f'train-only, {len(folds_holdout)} holdout')

    dir_model_full = os.path.join(cfg.DIR_MODELS, name)
    dir_folds = os.path.join(dir_model_full, SUBDIR_FOLDS)

    # CV: hold out one 'rotate' fold at a time, train on the other 'rotate'
    # folds plus every 'train' fold. The held-out fold doubles as the
    # early-stopping monitor — a within-fold split would leak site identity
    # into the stopping signal, and dedicating a second fold to it would cost
    # another deployment. Fold model binaries are not kept, only their scores
    # and training artifacts, archived under dir_folds.
    for held_out in folds_rotate:
        folds_train = [f for f in folds_rotate if f != held_out] + folds_train_always
        dir_model = os.path.join(dir_folds, str(held_out))
        modelname = f'{name}_fold{held_out}'

        if not can_write(dir_model):
            print(f'[{modelname}] already trained; skipping')
            continue

        data = _load_data(setname, embeddername, folds_train, name_translation,
                          aug_dirnames, val_fold=held_out)
        result, model = _train_one(
            dir_model, modelname, embeddername, setname, name_translation,
            data, epochs_max, aug_dirnames, verbose,
            held_out, save_binary=False, patience=patience,
        )
        if result is None:
            continue

        scores, _ = _write_scores(
            dir_model, model, setname, embeddername, held_out,
            data.translation, data.classes, modelname,
        )
        with open(os.path.join(dir_model, FNAME_FOLD_SUMMARY), 'w') as f:
            json.dump({**result, **scores}, f)

    summary_rows, predictions_pooled = _collect_fold_results(dir_folds, folds_rotate)

    if summary_rows:
        os.makedirs(dir_model_full, exist_ok=True)
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(os.path.join(dir_model_full, FNAME_FOLDS_SUMMARY), index=False)

        # Two headline numbers, deliberately: pooled is frame-weighted, which
        # matches how the shipped model is actually biased; the unweighted mean
        # treats every deployment equally. A gap between them means one
        # high-volume fold is carrying the result — see README.
        pooled = pd.concat(predictions_pooled, ignore_index=True)
        pooled_metrics = metrics_by_group(pooled.drop(columns='fold'))
        pooled_metrics.to_csv(os.path.join(dir_model_full, FNAME_POOLED_METRICS), index=False)
        pooled_sx = metrics_at_fpr(pooled_metrics)
        pooled_sx.to_csv(os.path.join(dir_model_full, FNAME_POOLED_SX), index=False)

        cols_sens = [c for c in summary.columns if c.startswith('sens_fpr')]
        print(f'\n[{name}] CV over {len(summary)} fold(s)')
        print(f'  pooled (frame-weighted): {_format_sens(pooled_sx.set_index("fpr")["sensitivity"])}')
        print('  unweighted mean across folds: ' + ', '.join(
            f'{c}={summary[c].mean():.3f}' for c in cols_sens
        ))
        print('  per-fold spread understates uncertainty about a new '
              'deployment (training pools overlap heavily)\n')

    # Shipped model: trains on every fold except 'holdout'. No fold is held
    # out, so there is nothing clean left to monitor — the epoch count comes
    # from the median best epoch across the rotations. Only model saved with a
    # binary.
    epochs_fixed = epochs_max
    if summary_rows:
        epochs_fixed = int(round(np.median([r['best_epoch'] for r in summary_rows])))
    else:
        print(f'[{name}] no fold results to take a median epoch from; '
              f'training the shipped model for the full {epochs_max} epochs')

    folds_shipped = folds_rotate + folds_train_always
    data = _load_data(setname, embeddername, folds_shipped, name_translation,
                      aug_dirnames, val_fold=None)
    result, model = _train_one(
        dir_model_full, name, embeddername, setname, name_translation,
        data, epochs_max, aug_dirnames, verbose,
        None, save_binary=True, epochs_fixed=epochs_fixed, patience=patience,
    )

    if result is None:
        return

    dir_set = cfg.dir_set(setname)
    shutil.copy(os.path.join(dir_set, 'annotations.csv'), dir_model_full)
    shutil.copy(os.path.join(dir_set, 'folds.csv'), dir_model_full)

    # 'holdout' folds never train, so the shipped model can be scored on them
    # directly — an estimate untouched by the CV rotation.
    for fold in folds_holdout:
        _write_scores(
            os.path.join(dir_model_full, SUBDIR_HOLDOUT, str(fold)),
            model, setname, embeddername, fold,
            data.translation, data.classes, name,
        )
