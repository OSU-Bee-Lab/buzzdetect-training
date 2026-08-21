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
    survey_untranslated, ROLE_TRAIN, ROLE_ROTATE, ROLE_HOLDOUT,
)
from train_utils import build_weights, build_classes, can_write, Sample
from embedders.embedding import load_embedder
from plot_history import plot_history
from write_model_py import write_model_py

from metrics import metrics_by_group, metrics_at_fpr

from sx import summarize_sx, summarize_sx_byfold, format_sx_report, FPR_TARGETS, FNAME_SX_SUMMARY, FNAME_SX_BYFOLD


@tf.keras.utils.register_keras_serializable(package='buzzdetect')
class TailBCELoss(tf.keras.losses.Loss):
    """BinaryCrossentropy(label_smoothing) plus an OHEM-style tail term on one
    neuron ('ins_buzz' in practice).

    The endpoint metric (sensitivity_mean @ fpr0.005) is decided by a handful
    of top-scoring negative frames per fold (neg_frames_fold_median=22 on
    cv-baseline) — plain BCE spends nearly all its gradient on the easy
    negative mass nowhere near that boundary. This adds a second term that
    looks, within each training batch, at only the highest-scoring negative
    frames on `buzz_index` and adds extra loss for them specifically — an
    OHEM-style hard-negative mining term, not a focal reweighting (focal
    reweights every example by the model's own confidence, globally; this
    reweights by rank position among negatives in the current batch).

    `call()` returns one loss value per sample, matching what
    `BinaryCrossentropy.call()` returns — so Keras's `sample_weight` /
    `class_weight` machinery multiplies and reduces this exactly the way it
    does the baseline loss, including the known argmax-collapse class_weight
    bug (see log.jsonl, class-weight-fix). Not fixed here; kept live so this
    stays paired with cv-baseline.

    Ported from exp/tail-loss (log.jsonl, trust=artifact), which collapsed
    sensitivity 0.305 -> 0.007 for two reasons, both fixed here:

    1. That loss was the *compiled* loss, so Keras also evaluated it (OHEM
       term included) on the single held-out validation fold, and that value
       fed EarlyStopping's val_loss monitor. The OHEM term's hard-negative set
       is batch-local; a single fold's negatives are a much smaller, noisier
       population than the pooled training batch, so which frames counted as
       "hard" swung early in training and val_loss hit a spurious minimum in
       the first handful of epochs. Fix lives in `_train_one`: EarlyStopping
       now monitors a separately compiled plain-BCE metric (`make_bce_metric`
       below), not this loss.
    2. `top_k`'s `k` was computed from `n_neg` (a fraction of however many
       negatives happened to be in the batch) — a data-dependent shape that
       triggered an out-of-range gather under Keras's fused multi-step
       execution. `ohem_k` here is a fixed Python int baked into the graph at
       trace time, never derived from a tensor.
    """

    def __init__(self, buzz_index, label_smoothing=0.2, ohem_k=200, ohem_weight=1.0,
                 name='tail_bce_loss'):
        super().__init__(name=name)
        self.buzz_index = buzz_index
        self.label_smoothing = label_smoothing
        self.ohem_k = ohem_k
        self.ohem_weight = ohem_weight

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)

        # base: per-sample mean over classes of smoothed sigmoid cross-entropy,
        # numerically the same as BinaryCrossentropy(from_logits=True,
        # label_smoothing=...).call().
        labels_smoothed = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        elementwise = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_smoothed, logits=y_pred)
        base = tf.reduce_mean(elementwise, axis=-1)

        logits_buzz = y_pred[:, self.buzz_index]
        labels_buzz = y_true[:, self.buzz_index]
        is_neg = tf.equal(labels_buzz, 0.0)
        neg_idx = tf.where(is_neg)[:, 0]
        neg_logits = tf.gather(logits_buzz, neg_idx)

        # Fixed-size top_k: ohem_k is a Python int, never a tensor derived
        # from batch contents, so this op's output shape is static at
        # graph-trace time. Every batch encountered in practice (pooled
        # training pool ~65-70k frames; a validation fold's few-thousand
        # frames, mostly negative) has far more than ohem_k negatives, so no
        # runtime clamping against n_neg happens here. A batch with fewer
        # than ohem_k negatives would make top_k raise, not silently cheat
        # the sample size.
        top_vals, top_pos = tf.math.top_k(neg_logits, k=self.ohem_k)
        top_sample_idx = tf.gather(neg_idx, top_pos)
        target = tf.fill(tf.shape(top_vals), tf.constant(0.5, dtype=y_pred.dtype) * self.label_smoothing)
        hard_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=top_vals)
        addition = tf.scatter_nd(
            tf.expand_dims(tf.cast(top_sample_idx, tf.int32), -1),
            self.ohem_weight * hard_loss,
            tf.shape(base),
        )
        return base + addition

    def get_config(self):
        config = super().get_config()
        config.update({
            'buzz_index': self.buzz_index,
            'label_smoothing': self.label_smoothing,
            'ohem_k': self.ohem_k,
            'ohem_weight': self.ohem_weight,
        })
        return config


def make_bce_metric(label_smoothing):
    """Plain BCE as a metric, decoupled from TailBCELoss, so EarlyStopping can
    monitor 'val_bce' instead of the compiled (tail-contaminated) val_loss.

    Numerically identical to TailBCELoss's `base` term / to
    BinaryCrossentropy(from_logits=True, label_smoothing=...) — no OHEM
    addition, so its value on the held-out fold isn't batch-locality-noisy the
    way the tail term is. The tail term stays live in the training gradient
    (it's still part of the compiled loss); it just never reaches the
    stopping signal.
    """
    def bce(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        labels_smoothed = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
        elementwise = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_smoothed, logits=y_pred)
        return tf.reduce_mean(elementwise, axis=-1)
    bce.__name__ = 'bce'
    return bce


# tail-loss-retest experiment knobs (see notes.md). ohem_weight is fixed at
# 1.0; ohem_k is the one hyperparameter varied across runs — edit here and use
# a fresh --name per value tried. 200 ~= 10x the ~22-frame validation-side
# neg_frames_fold_median, matching the training pool's ~10x scale (pooled
# across folds vs. one fold at validation).
OHEM_K = 200
OHEM_WEIGHT = 1.0

FNAME_PREDICTIONS = 'predictions.csv'
FNAME_FOLD_SUMMARY = 'summary.json'
FNAME_FOLDS_SUMMARY = 'folds_summary.csv'
FNAME_POOLED_METRICS = 'folds_pooled_metrics.csv'
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

    if not data_train:
        raise ValueError(
            f'no trainable frames across {len(folds_train)} fold(s) of set '
            f'{setname!r} under translation {name_translation!r} — every sample '
            f'was ignored or excluded'
        )

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

    Runs the threshold sweep straight off the fold's known frame labels.
    Returns
    (metrics_df, predictions_df); predictions are kept so the per-fold results
    can be pooled into one ROC afterwards.
    """
    samples = build_fold_dataset(
        cfg.dir_embeddings_fold(setname, embeddername, fold), translation,
    )
    if not samples:
        return None, None
    buzz_index = classes.index('ins_buzz')

    embeddings = np.concatenate([np.array(s.embeddings, dtype=np.float32) for s in samples])
    correct = np.concatenate([np.full(s.frames, bool(s.target_array[buzz_index])) for s in samples])
    activation = model(embeddings, training=False)[:, buzz_index].numpy()

    predictions = pd.DataFrame({'activation_ins_buzz': activation, 'correct': correct})
    return metrics_by_group(predictions), predictions


def _format_sens(sens):
    return ', '.join(
        f'sens@fpr{f:.1%}={sens[f]:.3f}' if pd.notna(sens[f]) else f'sens@fpr{f:.1%}=n/a'
        for f in sens.index
    )


def _write_scores(dir_out, model, setname, embeddername, fold, translation, classes):
    """Score `fold`, write metrics/sx/predictions under dir_out, return
    {sens_fpr<x>: value} for the summary table plus the raw predictions.

    Silent by design — the caller folds these numbers into its one-line
    per-fold report rather than printing a second line here."""
    os.makedirs(dir_out, exist_ok=True)
    metrics_df, predictions = _score_fold(model, setname, embeddername, fold, translation, classes)
    if metrics_df is None:
        return {}, None, None

    metrics_df.to_csv(os.path.join(dir_out, cfg.FNAME_METRICS), index=False)
    sx_df = metrics_at_fpr(metrics_df, FPR_TARGETS)
    sx_df.to_csv(os.path.join(dir_out, cfg.FNAME_SX), index=False)
    predictions.to_csv(os.path.join(dir_out, FNAME_PREDICTIONS), index=False)

    sens = sx_df.set_index('fpr')['sensitivity']
    return {f'sens_fpr{f:g}': sens[f] for f in sens.index}, predictions, sens


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

    if verbose:
        monitor = (f'early stopping on {data.val_fold} ({data.frames_val} frames)'
                   if data.val_fold else f'{epochs_fixed} fixed epochs, no monitor')
        print(f'[{modelname}] training on {len(data.folds_train)} fold(s), '
              f'{data.frames_train} frames; {monitor}...', flush=True)
    os.makedirs(dir_model, exist_ok=True)

    embedder = load_embedder(embeddername, framehop_prop=1, initialize=False)
    # modelname may contain a fold identifier derived from a source name
    # (e.g. "2025-06-04 original annotations"), which can hold characters
    # invalid in a Keras/TF scope name — sanitize for that use only.
    # Keras rejects '/' in layer/model names outright; fold ids are paths, so
    # strip separators here rather than relying on the caller's naming.
    tf_name = re.sub(r'[^A-Za-z0-9_.>-]', '_', modelname)
    model = tf.keras.Sequential(name=tf_name)
    model.add(tf.keras.layers.Input(shape=(embedder.n_embeddings,), dtype=tf.float32, name='input'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(len(data.classes)))

    buzz_index = data.classes.index('ins_buzz')
    model.compile(
        loss=TailBCELoss(buzz_index=buzz_index, label_smoothing=0.2,
                         ohem_k=OHEM_K, ohem_weight=OHEM_WEIGHT),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        metrics=['accuracy', make_bce_metric(label_smoothing=0.2)],
    )

    if data.val_tf is None:
        # Shipped model: no fold is held out, so there's nothing clean to
        # monitor. Train a fixed number of epochs instead, set by the caller
        # from the median best epoch across the rotations.
        history = model.fit(
            data.train_tf, epochs=epochs_fixed, class_weight=data.weight_dict,
            verbose=2 if verbose else 0,  # 2 = one line per epoch, no progress bar
        )
        best_epoch = epochs_fixed - 1
        result = {
            'n_epochs': epochs_fixed,
            'best_epoch': epochs_fixed,
            'frames_train': data.frames_train,
        }
    else:
        # Monitor plain BCE ('val_bce'), NOT 'val_loss' — val_loss is
        # TailBCELoss including the OHEM term, which is batch-local and noisy
        # on a single held-out fold (see class docstring / exp/tail-loss in
        # log.jsonl). The tail term still trains the weights; it just isn't
        # allowed to drive early stopping.
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_bce', mode='min', patience=patience, min_delta=0.002,
            restore_best_weights=True,
        )
        history = model.fit(
            data.train_tf,
            epochs=epochs_max,
            validation_data=data.val_tf,
            callbacks=callback,
            class_weight=data.weight_dict,
            verbose=2 if verbose else 0,  # 2 = one line per epoch, no progress bar
        )

        best_epoch = callback.best_epoch
        best_val_bce = float(callback.best)
        result = {
            'n_epochs': len(history.history['val_loss']),
            'best_epoch': best_epoch + 1,
            'best_val_loss': history.history['val_loss'][best_epoch],  # tail-inclusive, informational only
            'best_val_bce': best_val_bce,  # the actual stopping monitor
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


def _confirm_untranslated(setname, embeddername, folds, name_translation, assume_yes):
    """Report raw labels the translation has no row for, and get a go-ahead.

    Training on a set with a missing translation row is expensive and silent —
    the frames simply never contribute — so this is a gate before the first
    epoch rather than a warning after the fact. Returns True to proceed.
    """
    translation = pd.read_csv(os.path.join(cfg.DIR_TRANSLATIONS, name_translation + '.csv'))
    unknown = survey_untranslated(setname, embeddername, folds, translation)
    if not unknown:
        return True

    n_files = sum(unknown.values())
    print(f'\n{len(unknown)} raw label(s) across {n_files} embedding file(s) have no '
          f'row in translations/{name_translation}.csv:')
    for label, n in unknown.items():
        print(f'    {label}  ({n} file(s))')
    print('  Frames carrying only these labels will not train. Add a "from" row '
          'mapping each to a class, to "ignore", or to "exclude" to silence this.')

    if assume_yes:
        print('  --yes given; continuing.\n')
        return True
    if not sys.stdin.isatty():
        print('  Refusing to train unattended with unresolved labels. '
              'Fix the translation, or pass --yes to accept them.\n')
        return False

    reply = input('  Continue anyway? [y/N] ').strip().lower()
    print()
    return reply in ('y', 'yes')


def train_set(name, embeddername, setname, name_translation,
              epochs_max=400, aug_dirnames=None, verbose=False, patience=50,
              assume_yes=False):
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

    print(f'[{name}] set {setname}, embedder {embeddername}, translation '
          f'{name_translation}: {len(folds_rotate)} rotating fold(s), '
          f'{len(folds_train_always)} train-only, {len(folds_holdout)} holdout')

    if not _confirm_untranslated(setname, embeddername, list(roles),
                                 name_translation, assume_yes):
        return

    dir_model_full = os.path.join(cfg.DIR_MODELS, name)
    dir_folds = os.path.join(dir_model_full, SUBDIR_FOLDS)

    # CV: hold out one 'rotate' fold at a time, train on the other 'rotate'
    # folds plus every 'train' fold. The held-out fold doubles as the
    # early-stopping monitor — a within-fold split would leak site identity
    # into the stopping signal, and dedicating a second fold to it would cost
    # another deployment. Fold model binaries are not kept, only their scores
    # and training artifacts, archived under dir_folds.
    for i, held_out in enumerate(folds_rotate, 1):
        folds_train = [f for f in folds_rotate if f != held_out] + folds_train_always
        dir_model = os.path.join(dir_folds, str(held_out))
        modelname = f'{name}_fold{held_out}'
        tag = f'[{i}/{len(folds_rotate)}] {held_out}'

        if not can_write(dir_model):
            print(f'{tag}: already trained; skipping')
            continue

        data = _load_data(setname, embeddername, folds_train, name_translation,
                          aug_dirnames, val_fold=held_out)
        if data.frames_val == 0:
            # Nothing to early-stop on or score against — a legitimate state if
            # every label in this deployment is ignored or excluded, but it
            # can't take a turn as the held-out fold.
            print(f'{tag}: no usable frames under this translation; skipping rotation')
            continue

        result, model = _train_one(
            dir_model, modelname, embeddername, setname, name_translation,
            data, epochs_max, aug_dirnames, verbose,
            held_out, save_binary=False, patience=patience,
        )
        if result is None:
            continue

        scores, _, sens = _write_scores(
            dir_model, model, setname, embeddername, held_out,
            data.translation, data.classes,
        )
        with open(os.path.join(dir_model, FNAME_FOLD_SUMMARY), 'w') as f:
            json.dump({**result, **scores}, f)

        # One line per fold: everything worth knowing about this rotation, so a
        # default run stays roughly one line per fold rather than three.
        print(f"{tag}: {result['n_epochs']} epochs (best {result['best_epoch']}), "
              f"val_loss {result['best_val_loss']:.4f}, "
              f"{data.frames_train}/{data.frames_val} frames train/val, "
              f"{_format_sens(sens)}", flush=True)

    summary_rows, predictions_pooled = _collect_fold_results(dir_folds, folds_rotate)

    if summary_rows:
        os.makedirs(dir_model_full, exist_ok=True)
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(os.path.join(dir_model_full, FNAME_FOLDS_SUMMARY), index=False)

        # The headline read — sx.py explains the policies, the weightings, and
        # why the primary one is the per-deployment mean. The full pooled sweep
        # is still written out: it is what plots a ROC and what
        # metrics_at_precision reads. Its sens-at-target is a column of
        # folds_sx.csv now, so it gets no file of its own.
        pooled = pd.concat(predictions_pooled, ignore_index=True)
        pooled_metrics = metrics_by_group(pooled.drop(columns='fold'))
        pooled_metrics.to_csv(os.path.join(dir_model_full, FNAME_POOLED_METRICS), index=False)

        sx = summarize_sx(pooled)
        sx.to_csv(os.path.join(dir_model_full, FNAME_SX_SUMMARY), index=False)
        sx_byfold = summarize_sx_byfold(pooled)
        sx_byfold.to_csv(os.path.join(dir_model_full, FNAME_SX_BYFOLD), index=False)
        print(format_sx_report(name, sx, sx_byfold))

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

    print(f'[shipped] {name}: {epochs_fixed} fixed epochs on '
          f'{len(folds_shipped)} fold(s), {data.frames_train} frames → {dir_model_full}')

    dir_set = cfg.dir_set(setname)
    shutil.copy(os.path.join(dir_set, 'annotations.csv'), dir_model_full)
    shutil.copy(os.path.join(dir_set, 'folds.csv'), dir_model_full)

    # 'holdout' folds never train, so the shipped model can be scored on them
    # directly — an estimate untouched by the CV rotation.
    for fold in folds_holdout:
        _, _, sens = _write_scores(
            os.path.join(dir_model_full, SUBDIR_HOLDOUT, str(fold)),
            model, setname, embeddername, fold,
            data.translation, data.classes,
        )
        if sens is None:
            print(f'[holdout] {fold}: no usable frames under this translation; not scored')
        else:
            print(f'[holdout] {fold}: {_format_sens(sens)}')
