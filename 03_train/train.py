# TensorFlow imported first — see 03_train/main.py for rationale.
import json
import math
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
from train_utils import (build_weights, build_classes, can_write,
                         weighted_bce_loss, Sample)
from embedders.embedding import load_embedder
from plot_history import plot_history, plot_sens_history
from write_model_py import write_model_py

from callbacks import SensAtFPR, RestoreTrueBest

from sx import summarize_folds, format_sx_report, _fold_sens, FPR_TARGETS, FNAME_SX_SUMMARY
from surprisal import write_fold_surprisal

FNAME_PREDICTIONS = 'predictions.csv'
FNAME_FOLD_SUMMARY = 'summary.json'
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
    # (embeddings, is_buzz) for the validation fold, unshuffled and paired,
    # for the per-epoch sens@FPR monitor. None when there is no val fold.
    val_eval: tuple = None


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


def _eval_arrays(samples, classes):
    """Frame-level (embeddings, is_buzz) for a fold, in sample order.

    Scoring pairs each frame's activation with its own label, so unlike
    _to_tf's training pipeline this must not shuffle.
    """
    buzz_index = classes.index('ins_buzz')
    embeddings = np.concatenate([np.array(s.embeddings, dtype=np.float32) for s in samples])
    correct = np.concatenate([np.full(s.frames, bool(s.target_array[buzz_index])) for s in samples])
    return embeddings, correct


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
    translation = pd.read_csv(cfg.path_translation(setname, name_translation))
    classes = build_classes(translation)

    data_train = []
    for fold in folds_train:
        data_train += build_fold_dataset(
            cfg.dir_embeddings_fold(setname, embeddername, fold), translation,
        )
    frames_train = sum(s.frames for s in data_train)

    data_val = None
    frames_val = 0
    val_eval = None
    if val_fold is not None:
        data_val = build_fold_dataset(
            cfg.dir_embeddings_fold(setname, embeddername, val_fold), translation,
        )
        frames_val = sum(s.frames for s in data_val)
        if data_val:
            val_eval = _eval_arrays(data_val, classes)

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
        val_eval=val_eval,
    )


def _score_fold(model, setname, embeddername, fold, translation, classes):
    """Score a trained model on a fold it never saw, ins_buzz only.

    Returns the frame-level (activation, correct) table, or None if the fold
    has no usable frames. Every reported number is derived from this: it is the
    only per-fold result kept on disk, and sx.py and resummarize.py rebuild the
    sweeps from it on demand.
    """
    samples = build_fold_dataset(
        cfg.dir_embeddings_fold(setname, embeddername, fold), translation,
    )
    if not samples:
        return None

    embeddings, correct = _eval_arrays(samples, classes)
    activation = model(embeddings, training=False)[:, classes.index('ins_buzz')].numpy()

    return pd.DataFrame({'activation_ins_buzz': activation, 'correct': correct})


def _format_sens(sens):
    return ', '.join(
        f'sens@fpr{f:.1%}={sens[f]:.3f}' if pd.notna(sens[f]) else f'sens@fpr{f:.1%}=n/a'
        for f in sens.index
    )


def _write_predictions(dir_out, model, setname, embeddername, fold, translation, classes):
    """Score `fold`, write predictions.csv under dir_out, return
    (predictions, sens) — sens for the caller's one-line report.

    The sweeps are not written. metrics.csv and sx.csv used to land here too,
    both pure functions of predictions.csv and neither read by anything;
    metrics.csv alone was four times the size of the file it derived from.

    sens comes from sx._fold_sens, the guarded read, so this line agrees with
    folds_summary.csv and folds_sx.csv instead of quietly interpolating a
    sensitivity inside a single frame.

    Otherwise silent by design — the caller folds these numbers into its
    per-fold line rather than printing a second one here."""
    os.makedirs(dir_out, exist_ok=True)
    predictions = _score_fold(model, setname, embeddername, fold, translation, classes)
    if predictions is None:
        return None, None

    predictions.to_csv(os.path.join(dir_out, FNAME_PREDICTIONS), index=False)
    cols, _ = _fold_sens(predictions, FPR_TARGETS)
    return predictions, cols['sensitivity']


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


def _consensus_epoch(summary_rows, tol):
    """Shipped-model epoch count, read off the pooled per-fold val_loss curves.

    Each rotation early-stops at its own val_loss argmin, but on a frozen-
    embedding probe that basin is very flat — the per-fold argmins scatter by
    100+ epochs and their median lurches with fold composition. Instead:

      1. extend every fold's curve to the longest length, holding its own min
         past its natural end (EarlyStopping restores best weights, so the
         effective loss is frozen there);
      2. take the running min of each — the "best so far" trace;
      3. min-max normalise each to [0, 1];
      4. average them, weighted by validation-frame count — a 300-frame val
         fold's curve is mostly noise and should not swing the result;
      5. return the earliest epoch within `tol` of the averaged curve's floor.

    `tol` is a fraction of the averaged curve's own span (epoch-1 value down to
    its floor): tol=0.01 means "all but the last 1% of the improvement the
    consensus curve ever makes". It is the pooled-curve analogue of
    EarlyStopping's min_delta — larger tol stops earlier. Tune it by eye
    against the per-fold loss_curves.svg plots.

    Returns None if no fold carries a curve (an old run resumed from summaries
    written before this field existed); the caller falls back to the median.
    """
    curves = [(r['val_loss_curve'], r.get('frames_val', 1))
              for r in summary_rows if r.get('val_loss_curve')]
    if not curves:
        return None

    length = max(len(c) for c, _ in curves)
    stack, weights = [], []
    for curve, frames in curves:
        arr = np.asarray(curve, dtype=float)
        arr = np.concatenate([arr, np.full(length - len(arr), arr.min())])
        arr = np.minimum.accumulate(arr)
        span = arr.max() - arr.min()
        stack.append((arr - arr.min()) / span if span else np.zeros(length))
        weights.append(max(frames, 1))

    mean = np.average(stack, axis=0, weights=weights)
    threshold = mean.min() + tol * (mean.max() - mean.min())
    return int(np.argmax(mean <= threshold)) + 1


def _train_one(dir_model, modelname, embeddername, setname, name_translation,
               data: TrainingData, epochs_max, aug_dirnames, verbose,
               held_out_fold, save_binary, epochs_fixed=None, patience=50,
               stop_tol=None):
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

    # Per-class weights go in the loss, not in fit(class_weight=). Keras'
    # class_weight= assumes single-label targets: for a multi-hot y it collapses
    # each sample to argmax(y) and scales the whole sample by that one scalar, so
    # a buzz frame co-occurring with an earlier-indexed class never gets
    # ins_buzz's weight at all. See train_utils.weighted_bce_loss.
    weights_ordered = [data.weight_dict[i] for i in range(len(data.classes))]
    model.compile(
        loss=weighted_bce_loss(weights_ordered, label_smoothing=0.2),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        metrics=['accuracy'],
    )

    if data.val_tf is None:
        # Shipped model: no fold is held out, so there's nothing clean to
        # monitor. Train a fixed number of epochs instead, set by the caller
        # from the median best epoch across the rotations.
        history = model.fit(
            data.train_tf, epochs=epochs_fixed,
            callbacks=[tf.keras.callbacks.TerminateOnNaN()],
            # _to_tf already applies .shuffle(); say so, or Keras warns that it's
            # ignoring shuffle=True on a Dataset input every run.
            shuffle=False,
            # --verbose is for a human watching: 1 = live progress bar. Agents
            # leave the flag off (0) so per-epoch lines don't fill their context.
            verbose=1 if verbose else 0,
        )
        best_epoch = epochs_fixed - 1
        result = {
            'n_epochs': epochs_fixed,
            'best_epoch': epochs_fixed,
            'frames_train': data.frames_train,
        }
    else:
        callback = RestoreTrueBest(
            monitor='val_loss', patience=patience, min_delta=0.002, restore_best_weights=True,
        )
        # Reporting only, and listed first so its keys are in `logs` before
        # EarlyStopping and History see them. Stopping still happens on
        # val_loss; these curves are the evidence for whether it should.
        sens_callback = SensAtFPR(
            *data.val_eval, data.classes.index('ins_buzz'), FPR_TARGETS,
            batch_size=data.size_batch,
        )
        history = model.fit(
            data.train_tf,
            epochs=epochs_max,
            validation_data=data.val_tf,
            callbacks=[sens_callback, callback, tf.keras.callbacks.TerminateOnNaN()],
            shuffle=False,  # _to_tf already shuffles; see the fixed-epochs fit above
            # --verbose is for a human watching: 1 = live progress bar. Agents
            # leave the flag off (0) so per-epoch lines don't fill their context.
            verbose=1 if verbose else 0,
        )

        best_epoch = callback.best_epoch
        best_val_loss = float(callback.best)
        result = {
            'n_epochs': len(history.history['val_loss']),
            'best_epoch': best_epoch + 1,
            'best_val_loss': best_val_loss,
            'frames_train': data.frames_train,
            'frames_val': data.frames_val,
            # The whole val_loss trace, not just its argmin: the shipped-model
            # epoch count is read off the pooled curve (_consensus_epoch), and
            # per-fold argmins are too jumpy in this flat basin to median.
            'val_loss_curve': [float(x) for x in history.history['val_loss']],
            **_sens_history_summary(history.history, best_epoch),
        }

    if any(not math.isfinite(x) for x in history.history['loss']):
        raise RuntimeError(
            f'[{modelname}] training loss went non-finite (NaN/Inf) at epoch '
            f'{next(i for i, x in enumerate(history.history["loss"], 1) if not math.isfinite(x))}. '
            f'The tensorflow-metal (Apple GPU) backend produces this within a few '
            f'epochs on this data; CPU does not. Re-run with BUZZDETECT_NO_GPU=1 '
            f'(CUDA_VISIBLE_DEVICES does not affect the Metal device). CPU is ~GPU '
            f'speed for the 1024-d probe. Completed rotations are kept and skipped '
            f'on the re-run.'
        )

    if save_binary:
        model.save(os.path.join(dir_model, 'model.keras'), include_optimizer=True)

    if save_binary:
        # Provenance for the model that ships. The rotations' copies were
        # byte-identical to these (translation) or near enough (weights), and
        # their pickled histories were the biggest thing in the model dir after
        # the sweeps -- what a rotation is kept for is its predictions and its
        # curves, both of which survive below.
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
        'stop_tol': stop_tol,
    }
    # 'w' for the same reason as write_model_py's — can_write() is the gate
    with open(os.path.join(dir_model, 'config_model.json'), 'w') as f:
        f.write(json.dumps(config_model))

    plot_history(history, modelname, best_epoch, os.path.join(dir_model, 'loss_curves.svg'))
    plot_sens_history(history, modelname, best_epoch, FPR_TARGETS,
                      SensAtFPR.key, os.path.join(dir_model, 'sens_curves.svg'))
    if save_binary:
        write_model_py(dir_model, modelname, embeddername, config_model['digits_results'])

    return result, model


def _sens_history_summary(hist, best_epoch):
    """sens@FPR at the restored epoch, and where it actually peaked.

    The gap between the two is the whole point of the monitor: val_loss picks
    best_epoch, and these say what that choice cost (or saved) on the number
    the model is judged by. Nothing acts on them.
    """
    out = {}
    for fpr in FPR_TARGETS:
        curve = hist.get(SensAtFPR.key(fpr))
        if not curve:
            continue
        arr = np.array(curve, dtype=float)
        out[f'val_sens_fpr{fpr:g}_at_best'] = arr[best_epoch]
        # The whole curve, not just the restored point. val_loss_curve is kept
        # for the same reason: together they let a stopping rule be re-scored
        # offline -- "what would patience=N have shipped, and what did it cost
        # on the metric" -- instead of costing a CV to ask. NaN is JSON-illegal,
        # so blank out the epochs that never reached the target FPR.
        out[f'val_sens_fpr{fpr:g}_curve'] = [
            None if np.isnan(x) else float(x) for x in arr]
        if np.isnan(arr).all():
            # Fold never reaches this FPR at any epoch -- too few negative
            # frames for the target, most likely. sx.py's _fold_sens says more.
            continue
        peak = int(np.nanargmax(arr))
        out[f'val_sens_fpr{fpr:g}_peak'] = arr[peak]
        out[f'val_sens_fpr{fpr:g}_peak_epoch'] = peak + 1
    return out


def _confirm_untranslated(setname, embeddername, folds, name_translation, assume_yes):
    """Report raw labels the translation has no row for, and get a go-ahead.

    Training on a set with a missing translation row is expensive and silent —
    the frames simply never contribute — so this is a gate before the first
    epoch rather than a warning after the fact. Returns True to proceed.
    """
    translation = pd.read_csv(cfg.path_translation(setname, name_translation))
    unknown = survey_untranslated(setname, embeddername, folds, translation)
    if not unknown:
        return True

    n_files = sum(unknown.values())
    path = os.path.relpath(cfg.path_translation(setname, name_translation), cfg.ROOT)
    print(f'\n{len(unknown)} raw label(s) across {n_files} embedding file(s) have no '
          f'row in {path}:')
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
              assume_yes=False, stop_tol=0.01, skip_cv=False, train_shipped=False,
              only_folds=None, surprisal=True):
    roles = read_fold_roles(setname, embeddername)
    folds_rotate = folds_by_role(roles, ROLE_ROTATE)
    folds_train_always = folds_by_role(roles, ROLE_TRAIN)
    folds_holdout = folds_by_role(roles, ROLE_HOLDOUT)

    # --only-folds narrows which rotations are *scored*, not what they train on:
    # a rotation's pool is already "everything but the held-out fold", so a probe
    # run trains on exactly the data a full rotation would. folds_sx.csv over a
    # subset is NOT comparable to a full CV, and the shipped model is skipped.
    folds_scored = folds_rotate
    if only_folds:
        keep = set(only_folds)
        missing = keep - set(folds_rotate)
        if missing:
            raise ValueError(f'--only-folds not in the rotate set: {sorted(missing)}')
        folds_scored = [f for f in folds_rotate if f in keep]
        print(f'[{name}] --only-folds: scoring {len(folds_scored)} of '
              f'{len(folds_rotate)} rotating folds; shipped model skipped')

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
    #
    # --skip-cv trains no rotations at all: it goes straight to the shipped
    # model, taking its epoch count from whatever fold summaries are already on
    # disk. For picking up a shipped model after an interrupted CV without
    # paying to finish every remaining fold; folds_sx.csv is then a partial CV.
    if skip_cv:
        print(f'[{name}] --skip-cv: no rotations trained; shipped epoch count '
              f'comes from existing fold results only')
    for i, held_out in enumerate([] if skip_cv else folds_scored, 1):
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

        _, sens = _write_predictions(
            dir_model, model, setname, embeddername, held_out,
            data.translation, data.classes,
        )
        if surprisal:
            write_fold_surprisal(
                dir_model_full, model, setname, embeddername, held_out,
                data.translation, data.classes,
            )
        # Training-side facts only. The scores are not duplicated here: they
        # are recomputed from predictions.csv into folds_sx.csv, so a resumed
        # run and a fresh one cannot disagree about them.
        with open(os.path.join(dir_model, FNAME_FOLD_SUMMARY), 'w') as f:
            json.dump(result, f)

        # One line per fold: everything worth knowing about this rotation, so a
        # default run stays roughly one line per fold rather than three.
        print(f"{tag}: {result['n_epochs']} epochs (best {result['best_epoch']}), "
              f"val_loss {result['best_val_loss']:.4f}, "
              f"{data.frames_train}/{data.frames_val} frames train/val, "
              f"{_format_sens(sens)}", flush=True)

    summary_rows, predictions_pooled = _collect_fold_results(dir_folds, folds_scored)

    if skip_cv and not summary_rows:
        raise ValueError(
            f'--skip-cv: no fold results under {dir_folds} to take an epoch '
            f'count from. Run at least one rotation first.')

    if summary_rows:
        os.makedirs(dir_model_full, exist_ok=True)

        # The one results table — per fold, then the total. sx.py explains the
        # policies, the weightings, and why the primary read is the
        # per-deployment mean.
        pooled = pd.concat(predictions_pooled, ignore_index=True)
        facts = {r['fold']: {'frames_val': r['frames_val'], 'best_epoch': r['best_epoch']}
                 for r in summary_rows}
        sx = summarize_folds(pooled, facts)
        sx.to_csv(os.path.join(dir_model_full, FNAME_SX_SUMMARY), index=False)
        print(format_sx_report(name, sx))

    # The shipped model is a deliverable, not a measurement: folds_sx.csv is
    # built entirely from the rotations above, so nothing an experiment is judged
    # on depends on it. Default is therefore rotations-only -- the experiment
    # path -- and training it is the opt-in. Its epoch count comes from the fold
    # curves (_consensus_epoch), which are on disk by now, so a later run with
    # the same --name plus --skip-cv produces the same model.
    if only_folds:
        print(f'[{name}] --only-folds: a fold subset cannot stand in for the '
              f'full CV the shipped epoch count is read from; not training it.')
        return
    if not train_shipped:
        print(f'[{name}] rotations done; shipped model not trained (default). '
              f'Pass --train-shipped, or run --skip-cv later with this --name.')
        return

    # Shipped model: trains on every fold except 'holdout'. No fold is held
    # out, so there is nothing clean left to monitor — the epoch count is read
    # off the pooled rotation val_loss curves (_consensus_epoch), falling back
    # to the median per-fold best epoch. Only model saved with a binary.
    epochs_fixed = epochs_max
    if summary_rows:
        n_curves = sum(1 for r in summary_rows if r.get('val_loss_curve'))
        epochs_fixed = _consensus_epoch(summary_rows, stop_tol)
        if epochs_fixed is not None:
            print(f'[{name}] shipped epoch count {epochs_fixed} '
                  f'(consensus val_loss curve over {n_curves}/{len(folds_rotate)} '
                  f'rotate folds, stop_tol={stop_tol})')
        else:
            epochs_fixed = int(round(np.median([r['best_epoch'] for r in summary_rows])))
            print(f'[{name}] shipped epoch count {epochs_fixed} '
                  f'(median per-fold best epoch over {len(summary_rows)}/'
                  f'{len(folds_rotate)} rotate folds; no curves on disk)')
    else:
        print(f'[{name}] no fold results to take an epoch count from; '
              f'training the shipped model for the full {epochs_max} epochs')

    folds_shipped = folds_rotate + folds_train_always
    data = _load_data(setname, embeddername, folds_shipped, name_translation,
                      aug_dirnames, val_fold=None)
    result, model = _train_one(
        dir_model_full, name, embeddername, setname, name_translation,
        data, epochs_max, aug_dirnames, verbose,
        None, save_binary=True, epochs_fixed=epochs_fixed, patience=patience,
        stop_tol=stop_tol,
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
        _, sens = _write_predictions(
            os.path.join(dir_model_full, SUBDIR_HOLDOUT, str(fold)),
            model, setname, embeddername, fold,
            data.translation, data.classes,
        )
        if sens is None:
            print(f'[holdout] {fold}: no usable frames under this translation; not scored')
        else:
            print(f'[holdout] {fold}: {_format_sens(sens)}')
        if surprisal:
            write_fold_surprisal(
                dir_model_full, model, setname, embeddername, fold,
                data.translation, data.classes,
            )
