# TensorFlow imported first — see 03_train/main.py for rationale.
import json
import os
import pickle
import re
import shutil
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

import config as cfg

from dataset import build_fold_dataset, load_augmented, discover_folds
from train_utils import build_weights, build_classes, can_write, Sample
from embedders.embedding import load_embedder
from plot_history import plot_history
from write_model_py import write_model_py

# metrics_by_group/metrics_at_fpr are the same threshold-sweep logic 04_test
# used against its on-disk results; reused here in-memory.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '04_test'))
from metrics import metrics_by_group, metrics_at_fpr


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


def _load_data(setname, embeddername, folds, name_translation, aug_dirnames, val_fold=None):
    """Pool every fold except val_fold for training; val_fold (if given) is a
    whole separate fold used only for early stopping. val_fold=None means no
    internal val at all (every given fold goes to training)."""
    translation = pd.read_csv(os.path.join(cfg.DIR_TRANSLATIONS, name_translation + '.csv'))
    classes = build_classes(translation)

    train_folds = [f for f in folds if f != val_fold] if val_fold is not None else list(folds)

    data_train: list[Sample] = []
    for fold in train_folds:
        data_train += build_fold_dataset(
            cfg.dir_embeddings_fold(setname, embeddername, fold), translation,
        )

    data_val = None
    if val_fold is not None:
        data_val = build_fold_dataset(
            cfg.dir_embeddings_fold(setname, embeddername, val_fold), translation,
        )

    if aug_dirnames:
        data_train += load_augmented(setname, embeddername, aug_dirnames, translation, train_folds)

    weights = build_weights(data_train, classes)
    weight_dict = {i: w for i, w in enumerate(weights['weight'])}

    size_batch = 65568
    size_shuffle = 10 * size_batch

    return (
        _to_tf(data_train, size_batch, size_shuffle),
        _to_tf(data_val, size_batch, size_shuffle) if data_val is not None else None,
        classes, weight_dict, weights, translation,
        size_batch, size_shuffle, train_folds,
    )


def _evaluate_holdout(model, setname, embeddername, held_out_fold, translation, classes):
    """Score the just-trained model on its held-out fold, ins_buzz only —
    the same threshold-sweep 04_test computed from on-disk inference results,
    computed here directly from the held-out fold's known frame labels."""
    samples = build_fold_dataset(
        cfg.dir_embeddings_fold(setname, embeddername, held_out_fold), translation,
    )
    buzz_index = classes.index('ins_buzz')

    embeddings = np.concatenate([np.array(s.embeddings, dtype=np.float32) for s in samples])
    correct = np.concatenate([np.full(s.frames, bool(s.target_array[buzz_index])) for s in samples])
    activation = model(embeddings, training=False)[:, buzz_index].numpy()

    results = pd.DataFrame({'activation_ins_buzz': activation, 'correct': correct})
    return metrics_by_group(results)


def _train_one(dir_model, modelname, embeddername, setname, name_translation,
               data_train_tf, data_val_tf, classes, weight_dict, weights, translation,
               size_batch, size_shuffle, epochs_max, aug_dirnames, verbose,
               held_out_fold, train_folds, val_fold, save_binary, epochs_fixed=None):
    if not can_write(dir_model):
        print(f'[{modelname}] already trained; skipping')
        return None

    print(f'[{modelname}] training...')
    os.makedirs(dir_model, exist_ok=True)

    embedder = load_embedder(embeddername, framehop_prop=1, initialize=False)
    # modelname may contain a fold identifier derived from a source name
    # (e.g. "2025-06-04 original annotations"), which can hold characters
    # invalid in a Keras/TF scope name — sanitize for that use only.
    tf_name = re.sub(r'[^A-Za-z0-9_.\\/>-]', '_', modelname)
    model = tf.keras.Sequential(name=tf_name)
    model.add(tf.keras.layers.Input(shape=(embedder.n_embeddings,), dtype=tf.float32, name='input'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(len(classes)))

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.2),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        metrics=['accuracy'],
    )

    if epochs_fixed is not None:
        # Shipped model: no internal val (would sacrifice a whole deployment),
        # so no early stopping — train for a fixed epoch count instead,
        # decided by the caller (median best_epoch across the CV folds).
        history = model.fit(
            data_train_tf, epochs=epochs_fixed, class_weight=weight_dict,
            verbose=1 if verbose else 0,
        )
        best_epoch = epochs_fixed - 1
        print(f'[{modelname}] done — trained {epochs_fixed} fixed epochs (median best_epoch across CV folds)')
        result = {'n_epochs': epochs_fixed, 'best_epoch': epochs_fixed}
    else:
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=50, min_delta=0.002, restore_best_weights=True,
        )
        history = model.fit(
            data_train_tf,
            epochs=epochs_max,
            validation_data=data_val_tf,
            callbacks=callback,
            class_weight=weight_dict,
            verbose=1 if verbose else 0,
        )
        best_epoch = callback.best_epoch
        best_val_loss = float(callback.best)
        best_val_accuracy = history.history['val_accuracy'][best_epoch]
        print(f'[{modelname}] done — {len(history.history["val_loss"])} epochs, '
              f'best epoch {best_epoch + 1}, val_loss {best_val_loss:.4f}')
        result = {
            'n_epochs': len(history.history['val_loss']),
            'best_epoch': best_epoch + 1,
            'best_val_loss': best_val_loss,
            'best_val_accuracy': best_val_accuracy,
        }

    if held_out_fold is not None:
        metrics_df = _evaluate_holdout(model, setname, embeddername, held_out_fold, translation, classes)
        metrics_df.to_csv(os.path.join(dir_model, cfg.FNAME_METRICS), index=False)

        sx_df = metrics_at_fpr(metrics_df)
        sx_df.to_csv(os.path.join(dir_model, cfg.FNAME_SX), index=False)
        sx_by_fpr = sx_df.set_index('fpr')['sensitivity']
        print(f'[{modelname}] held-out fold {held_out_fold}: ' + ', '.join(
            f'sens@fpr{f:.0%}={sx_by_fpr[f]:.3f}' if pd.notna(sx_by_fpr[f]) else f'sens@fpr{f:.0%}=n/a'
            for f in sx_by_fpr.index
        ))
        for f in sx_by_fpr.index:
            result[f'sens_fpr{f:g}'] = sx_by_fpr[f]

    if save_binary:
        model.save(os.path.join(dir_model, 'model.keras'), include_optimizer=True)

    with open(os.path.join(dir_model, 'history.pickle'), 'wb') as f:
        pickle.dump(history, f)

    weights.to_csv(os.path.join(dir_model, 'weights.csv'), index=False)
    translation.to_csv(os.path.join(dir_model, 'translation.csv'), index=False)

    config_model = {
        'embeddername': embeddername,
        'set': setname,
        'translation': name_translation,
        'classes': classes,
        'size_shuffle': size_shuffle,
        'size_batch': size_batch,
        'digits_results': 8,
        'aug_dirnames': aug_dirnames or [],
        'held_out_fold': held_out_fold,
        'train_folds': train_folds,
        'val_fold': val_fold,
    }
    with open(os.path.join(dir_model, 'config_model.json'), 'x') as f:
        f.write(json.dumps(config_model))

    plot_history(history, modelname, best_epoch, os.path.join(dir_model, 'loss_curves.svg'))
    if save_binary:
        write_model_py(dir_model, modelname, embeddername, config_model['digits_results'])

    return result


def train_set(name, embeddername, setname, name_translation,
              epochs_max=400, aug_dirnames=None, verbose=False):
    folds = discover_folds(setname, embeddername)
    n = len(folds)
    if n < 3:
        raise ValueError(
            f'need at least 3 folds for cross-validation (test + val + >=1 train), '
            f'found {n} in set {setname!r}'
        )

    dir_model_full = os.path.join(cfg.DIR_MODELS, name)
    dir_folds = os.path.join(dir_model_full, 'folds')

    # CV: one held-out fold at a time, one model per fold. val is always the
    # next fold after the held-out one (fixed rotation, wrapping) — arbitrary
    # but deterministic. Model binaries are not kept — only the held-out
    # evaluation and training artifacts, archived under dir_folds.
    summary_rows = []
    for i, held_out in enumerate(folds):
        val_fold = folds[(i + 1) % n]
        remaining = [f for f in folds if f != held_out]

        data_train_tf, data_val_tf, classes, weight_dict, weights, translation, size_batch, size_shuffle, train_folds = (
            _load_data(setname, embeddername, remaining, name_translation, aug_dirnames, val_fold=val_fold)
        )
        modelname = f'{name}_fold{held_out}'
        dir_model = os.path.join(dir_folds, str(held_out))
        result = _train_one(
            dir_model, modelname, embeddername, setname, name_translation,
            data_train_tf, data_val_tf, classes, weight_dict, weights, translation,
            size_batch, size_shuffle, epochs_max, aug_dirnames, verbose,
            held_out, train_folds, val_fold, save_binary=False,
        )
        if result:
            summary_rows.append({'fold': held_out, **result})

    if summary_rows:
        os.makedirs(dir_model_full, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(os.path.join(dir_model_full, 'folds_summary.csv'), index=False)

    # Shipped model: trains on every fold pooled, no internal val fold (would
    # sacrifice a deployment) — epoch count fixed at the median best_epoch
    # from the CV sweep instead of live early stopping.
    if summary_rows:
        epochs_fixed = int(round(np.median([r['best_epoch'] for r in summary_rows])))
    else:
        path_summary = os.path.join(dir_model_full, 'folds_summary.csv')
        if os.path.exists(path_summary):
            epochs_fixed = int(round(pd.read_csv(path_summary)['best_epoch'].median()))
        else:
            epochs_fixed = epochs_max

    data_train_tf, _, classes, weight_dict, weights, translation, size_batch, size_shuffle, train_folds = (
        _load_data(setname, embeddername, folds, name_translation, aug_dirnames, val_fold=None)
    )
    result = _train_one(
        dir_model_full, name, embeddername, setname, name_translation,
        data_train_tf, None, classes, weight_dict, weights, translation,
        size_batch, size_shuffle, epochs_max, aug_dirnames, verbose,
        None, train_folds, None, save_binary=True, epochs_fixed=epochs_fixed,
    )
    if result:
        dir_set = cfg.dir_set(setname)
        shutil.copy(os.path.join(dir_set, 'annotations.csv'), dir_model_full)
        shutil.copy(os.path.join(dir_set, 'folds.csv'), dir_model_full)
