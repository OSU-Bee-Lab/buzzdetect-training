# TensorFlow imported first — see 03_train/main.py for rationale.
import gc
import json
import math
import os
import pickle
import platform
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import tensorflow as tf

import config as cfg

from dataset import (
    build_fold_dataset, load_augmented, read_fold_roles, folds_by_role,
    survey_untranslated, ROLE_TRAIN, ROLE_ROTATE, ROLE_HOLDOUT,
)
from train_utils import (build_weights, build_classes, can_write,
                         weighted_bce_loss, Sample, buzz_tier,
                         TIERS_EXCLUDED_FROM_HEADLINE)
from embedders.embedding import load_embedder
from plot_history import plot_history, plot_sens_history
from write_model_py import write_model_py

from callbacks import SensAtFPR

from sx import summarize_folds, format_sx_report, _fold_sens, FPR_TARGETS, FNAME_SX_SUMMARY
from surprisal import write_fold_surprisal
from thresholds import (class_predictions_frame, write_model_card,
                        FNAME_PREDICTIONS_CLASSES)

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


def _rss_gb():
    try:
        with open('/proc/self/status') as f:
            return next(int(l.split()[1]) for l in f if l.startswith('VmRSS')) / 2**20
    except Exception:
        return float('nan')


def _to_tf(data, size_batch, size_shuffle, free=False):
    if os.environ.get('TRUNK_FP16'):
        return _to_tf_lowmem(data, size_batch, free)
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


def _pack_fp16(samples, free=False):
    """Pack samples' embeddings into one preallocated float16 array, freeing
    each sample's raw copy as it's copied in (see _load_data: packing one
    source fold at a time instead of the whole training pool keeps the raw
    and packed copies from coexisting at full size)."""
    n = sum(s.frames for s in samples)
    width = len(samples[0].embeddings[0])
    emb = np.empty((n, width), dtype=np.float16)
    tgt = np.empty((n, len(samples[0].target_array)), dtype=np.float32)
    at = 0
    for s in samples:
        emb[at:at + s.frames] = np.asarray(s.embeddings)
        tgt[at:at + s.frames] = s.target_array
        at += s.frames
        if free:
            s.embeddings = None  # the float16 copy above is the only one kept
    return emb, tgt


def _to_tf_lowmem(data, size_batch, free=False):
    """Wide float16 embeddings (yamnet_trunk 12288-d, yamnet_trunk_context 36864-d):
    one preallocated float16 array, reshuffled every epoch by a generator that casts
    per batch. The plain path holds ~5 copies, which is ~26 GB for the context arm."""
    emb, tgt = _pack_fp16(data, free=free)
    return _Fp16Batches(emb, tgt, size_batch)


def _assemble_fp16(chunks_emb, chunks_tgt, size_batch):
    """Concatenate per-fold float16 chunks (see _load_data) into one array
    without the peak a plain np.concatenate would add: each chunk is copied
    into the final buffer and dropped immediately, so memory is the final
    array plus at most one live chunk, never both full copies at once."""
    n = sum(len(c) for c in chunks_emb)
    emb = np.empty((n, chunks_emb[0].shape[1]), dtype=np.float16)
    tgt = np.empty((n, chunks_tgt[0].shape[1]), dtype=np.float32)
    at = 0
    while chunks_emb:
        c_emb, c_tgt = chunks_emb.pop(0), chunks_tgt.pop(0)
        emb[at:at + len(c_emb)] = c_emb
        tgt[at:at + len(c_tgt)] = c_tgt
        at += len(c_emb)
    return _Fp16Batches(emb, tgt, size_batch)


class _Fp16Batches(tf.keras.utils.PyDataset):
    """Reshuffled-every-epoch float16 batches, cast to float32 per batch. A PyDataset
    rather than Dataset.from_generator: from_generator parks its closure (here the
    whole embedding array) in TensorFlow's global py-function registry, so every
    fold's data outlived the fold and the host ran out of RAM after a few folds."""

    def __init__(self, emb, tgt, size_batch):
        super().__init__()
        self.emb, self.tgt, self.size_batch = emb, tgt, size_batch
        self.perm = np.random.permutation(len(emb))

    def __len__(self):
        return -(-len(self.emb) // self.size_batch)

    def __getitem__(self, i):
        b = np.sort(self.perm[i * self.size_batch:(i + 1) * self.size_batch])
        return self.emb[b].astype(np.float32), self.tgt[b]

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.emb))


def _eval_arrays(samples, classes):
    """Frame-level (embeddings, is_buzz, is_quiet_buzz, sample_id) for a fold,
    in sample order.

    Scoring pairs each frame's activation with its own label, so unlike
    _to_tf's training pipeline this must not shuffle.

    `loudness` is how audible the frame's buzz was judged to be — one of
    quiet/untagged/normal/loud, empty for a non-buzz frame. Every buzz frame is
    an ordinary positive in `correct` and every one of them trains; the tier
    only steers *scoring*, in sx.py, which drops the quiet ones from the
    headline and reports sensitivity per tier beside it. See
    train_utils.buzz_tier.

    `sample_id` numbers the snip each frame came from. Every frame of a sample
    shares one `correct`, so a buzz sample is one buzz event — which is the
    unit any honest n, bootstrap or standard error on this metric has to
    resample, frames within an event being anything but independent. sx.py
    reads it (buzz_event_blocks) and falls back to row adjacency for the runs
    written before this column existed.
    """
    buzz_index = classes.index('ins_buzz')
    embeddings = np.concatenate([np.array(s.embeddings, dtype=np.float32) for s in samples])
    correct = np.concatenate([np.full(s.frames, bool(s.target_array[buzz_index])) for s in samples])
    loudness = np.concatenate([
        np.full(s.frames, buzz_tier(s.labels_raw, s.labels_translate))
        for s in samples
    ])
    sample_id = np.concatenate([np.full(s.frames, i) for i, s in enumerate(samples)])
    return embeddings, correct, loudness, sample_id


def _load_data(setname, embeddername, folds_train, name_translation, aug_dirnames,
               val_fold=None):
    """Pool folds_train for training; val_fold, if given, is a whole separate
    deployment used to record the val_loss curve (nothing stops on it).

    Validation is always a whole fold, never a split within one. Snips from a
    deployment share a recorder, a site and a background, so a within-fold
    split would leak site identity into that curve. val_fold=None means no
    curve at all — the caller fixes the epoch count instead.

    Augmented embeddings go to training only.
    """
    translation = pd.read_csv(cfg.path_translation(setname, name_translation))
    classes = build_classes(translation)

    # Packed per source fold as it loads, not after the whole training pool is
    # in RAM: `large`'s ~40-fold pool held every fold's raw float32 embeddings
    # (~18 GB) alongside the packed float16 copy (~9 GB) under the old
    # load-everything-then-convert order, which OOM-killed the process on
    # every fold (fine on `medium`'s much smaller pool).
    fp16 = bool(os.environ.get('TRUNK_FP16'))
    data_train = []
    chunks_emb, chunks_tgt = [], []
    for fold in folds_train:
        fold_samples = build_fold_dataset(
            cfg.dir_embeddings_fold(setname, embeddername, fold), translation,
        )
        if fp16 and fold_samples:
            emb_chunk, tgt_chunk = _pack_fp16(fold_samples, free=True)
            chunks_emb.append(emb_chunk)
            chunks_tgt.append(tgt_chunk)
        data_train += fold_samples
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
            # The per-epoch monitor stays on the inclusive reading (all buzz,
            # every tier) as its first two elements. It steers nothing under
            # the fixed-budget rule; it is a curve, and changing what it counts
            # would break comparison with the curves already on disk. The
            # exclquiet correct array rides alongside it for the headline-
            # matching curve on sens_curves.svg.
            embeddings_val, correct_val, loudness_val, _ = _eval_arrays(data_val, classes)
            correct_val_exclquiet = correct_val & ~np.isin(loudness_val, TIERS_EXCLUDED_FROM_HEADLINE)
            val_eval = (embeddings_val, correct_val, correct_val_exclquiet)

    if aug_dirnames:
        aug_samples = load_augmented(setname, embeddername, aug_dirnames, translation, folds_train)
        if fp16 and aug_samples:
            emb_chunk, tgt_chunk = _pack_fp16(aug_samples, free=True)
            chunks_emb.append(emb_chunk)
            chunks_tgt.append(tgt_chunk)
        data_train += aug_samples

    if not data_train:
        raise ValueError(
            f'no trainable frames across {len(folds_train)} fold(s) of set '
            f'{setname!r} under translation {name_translation!r} — every sample '
            f'was ignored or excluded'
        )

    weights = build_weights(data_train, classes)
    weight_dict = {i: w for i, w in enumerate(weights['weight'])}

    size_batch = int(os.environ.get('TRUNK_BATCH', 65568))
    size_shuffle = 10 * size_batch

    train_tf = (_assemble_fp16(chunks_emb, chunks_tgt, size_batch) if fp16
                else _to_tf(data_train, size_batch, size_shuffle, free=True))

    return TrainingData(
        train_tf=train_tf,
        # already extracted into val_eval above; free=True drops the raw
        # copy instead of keeping it resident for the rest of the fold.
        val_tf=_to_tf(data_val, size_batch, size_shuffle, free=True) if data_val is not None else None,
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
    """Score a trained model on a fold it never saw.

    Returns (predictions, predictions_classes), or (None, None) if the fold has
    no usable frames. `predictions` is the ins_buzz frame-level (activation,
    correct) table; `predictions_classes` carries every class's logit and
    target for thresholds.py, which suggests a threshold per class. Every reported number is derived from this: it is the
    only per-fold result kept on disk, and sx.py and resummarize.py rebuild the
    sweeps from it on demand.
    """
    samples = build_fold_dataset(
        cfg.dir_embeddings_fold(setname, embeddername, fold), translation,
    )
    if not samples:
        return None, None

    embeddings, correct, loudness, sample_id = _eval_arrays(samples, classes)
    logits = np.concatenate([model(embeddings[i:i + 1024], training=False).numpy()
                             for i in range(0, len(embeddings), 1024)])
    activation = logits[:, classes.index('ins_buzz')]
    targets = np.concatenate([np.tile(np.asarray(s.target_array), (s.frames, 1))
                              for s in samples])

    # `sample` goes last: tools/eval_sampling_sd.py reads this file by column
    # index, and older models' predictions.csv has to stay readable beside it.
    return pd.DataFrame({
        'activation_ins_buzz': activation, 'correct': correct, 'loudness': loudness,
        'sample': sample_id,
    }), class_predictions_frame(logits, targets, sample_id, classes, loudness)


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
    predictions, predictions_classes = _score_fold(
        model, setname, embeddername, fold, translation, classes)
    if predictions is None:
        return None, None

    predictions.to_csv(os.path.join(dir_out, FNAME_PREDICTIONS), index=False)
    predictions_classes.to_csv(os.path.join(dir_out, FNAME_PREDICTIONS_CLASSES), index=False)
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

    Each rotation trains the full fixed budget, but on a frozen-embedding
    probe the val_loss basin is very flat — the per-fold argmins scatter by
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
               data: TrainingData, epochs, aug_dirnames, verbose,
               held_out_fold, save_binary, epochs_shipped=None,
               stop_tol=None, dropout=0.0):
    """Train one model. Returns (result_row, model); (None, None) if the model
    directory is already populated."""
    marker = 'model.keras' if save_binary else 'config_model.json'
    if not can_write(dir_model, marker):
        print(f'[{modelname}] already trained; skipping')
        return None, None

    if verbose:
        monitor = (f'validating on {data.val_fold} ({data.frames_val} frames), fixed budget'
                   if data.val_fold else f'{epochs_shipped} fixed epochs, no monitor')
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
    # The era's baseline is the bare linear probe: one Dense straight off the
    # frozen embedding, no dropout, no hidden layer. Dropout was 0.2 and
    # hardcoded through 2026-09-11; it is a regulariser tuned on YAMNet's
    # 89.6%-sparse non-negative code, and on a dense signed code it is heavy
    # multiplicative noise instead. It is now an experiment (--dropout), not a
    # premise, so that the anchor every result is read against is the simplest
    # thing that could work.
    if dropout:
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(len(data.classes)))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    if hasattr(embedder, 'build_head'):
        # Fine-tunable embedder (yamnet_trunk): layers 13-14 live in the head.
        model = embedder.build_head(
            len(data.classes), lr_backbone=float(os.environ.get('TRUNK_LR_BACKBONE', 0)),
            lr_head=float(os.environ.get('TRUNK_LR_HEAD', 2e-4)), dropout=dropout, name=tf_name)
        optimizer = model.optimizer

    # Per-class weights go in the loss, not in fit(class_weight=). Keras'
    # class_weight= assumes single-label targets: for a multi-hot y it collapses
    # each sample to argmax(y) and scales the whole sample by that one scalar, so
    # a buzz frame co-occurring with an earlier-indexed class never gets
    # ins_buzz's weight at all. See train_utils.weighted_bce_loss.
    weights_ordered = [data.weight_dict[i] for i in range(len(data.classes))]
    model.compile(
        loss=weighted_bce_loss(weights_ordered, label_smoothing=0.2),
        optimizer=optimizer,
        metrics=['accuracy'],
    )

    if data.val_tf is None:
        # Shipped model: no fold is held out, so there's nothing clean to
        # monitor. Train a fixed number of epochs instead, set by the caller
        # from the median best epoch across the rotations.
        history = model.fit(
            data.train_tf, epochs=epochs_shipped,
            callbacks=[tf.keras.callbacks.TerminateOnNaN()],
            # _to_tf already applies .shuffle(); say so, or Keras warns that it's
            # ignoring shuffle=True on a Dataset input every run.
            shuffle=False,
            # --verbose is for a human watching: 1 = live progress bar. Agents
            # leave the flag off (0) so per-epoch lines don't fill their context.
            verbose=1 if verbose else 0,
        )
        best_epoch = epochs_shipped - 1
        result = {
            'n_epochs': epochs_shipped,
            'best_epoch': epochs_shipped,
            'frames_train': data.frames_train,
        }
    else:
        # Reporting only, and listed first so its keys are in `logs` before any
        # other callback or History sees them.
        val_embeddings, val_correct, val_correct_exclquiet = data.val_eval
        sens_callback = SensAtFPR(
            val_embeddings, val_correct, data.classes.index('ins_buzz'), FPR_TARGETS,
            batch_size=data.size_batch, correct_exclquiet=val_correct_exclquiet,
        )
        # THE ONLY ROTATION RULE. Every rotation trains exactly `epochs` and
        # ships its final weights: no early stopping, no restore-best, no
        # per-fold epoch selection. All arms of a comparison are scored at one
        # identical epoch, so a capacity or normalisation change cannot be
        # confounded by the stopping rule.
        #
        # It replaced val_loss early stopping on 2026-09-11. That rule carried
        # no measurable *selection* optimism (-0.002 over 17 runs) but
        # undertrained unevenly: 1_150 hit its val_loss argmin at epoch 2-32
        # under every embedder tried while its buzz curve climbed to e120-185,
        # so that fold shipped a barely-trained probe. Removing it measured
        # +0.031 and +0.040 on two embedders. Evidence:
        # archive/2026-09-08_cv-medium-v2/ and
        # exp/pairwise-rank:notes/new-era-audit.md. Per-fold early stopping
        # (once a --early-stop flag here) was removed outright rather than kept
        # as an option — it is strictly worse for a rotation, and its only
        # other use, deriving the shipped model's epoch count, is now read off
        # the pooled rotation curves instead (see _consensus_epoch below),
        # which every fold's full-length curve supports without it.
        #
        # The sens curves are still persisted, so an offline cross-fold epoch
        # rule (tools/honest_epoch.py) can pick a shared epoch below the
        # budget as a diagnostic. Every fold's curve runs the full length, so
        # that tool's truncation caveat does not bind.
        history = model.fit(
            data.train_tf,
            epochs=epochs,
            validation_data=data.val_tf,
            callbacks=[sens_callback, tf.keras.callbacks.TerminateOnNaN()],
            shuffle=False,  # _to_tf already shuffles
            # --verbose is for a human watching: 1 = live progress bar.
            # Agents leave the flag off (0) so per-epoch lines don't fill
            # their context.
            verbose=1 if verbose else 0,
        )
        best_epoch = len(history.history['val_loss']) - 1
        best_val_loss = float(history.history['val_loss'][best_epoch])
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

    # Read alongside config_model.json, not carried in TrainingData: it
    # describes the set's extraction, not this training run, and a set
    # trained before config_extract.json existed has none on disk.
    path_config_extract = os.path.join(cfg.dir_set(setname), 'config_extract.json')
    overlap_event_prop = None
    if os.path.exists(path_config_extract):
        with open(path_config_extract) as f:
            overlap_event_prop = json.load(f).get('overlap_event_prop')

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
        'epochs_shipped': epochs_shipped,
        'stop_tol': stop_tol,
        # The fixed budget every rotation trained for.
        'epochs': epochs,
        'dropout': dropout,
        'trained_date': date.today().isoformat(),
        'overlap_event_prop': overlap_event_prop,
    }
    # 'w' for the same reason as write_model_py's — can_write() is the gate
    with open(os.path.join(dir_model, 'config_model.json'), 'w') as f:
        f.write(json.dumps(config_model))

    plot_history(history, modelname, best_epoch, os.path.join(dir_model, 'loss_curves.svg'))
    plot_sens_history(history, modelname, best_epoch, FPR_TARGETS,
                      SensAtFPR.key, os.path.join(dir_model, 'sens_curves.svg'),
                      key_exclquiet=SensAtFPR.key_exclquiet)
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


def _forbid_metal():
    """Refuse to train on the Metal PluggableDevice.

    There's no CUDA path on macOS, so any GPU tf.config.list_physical_devices
    finds there *is* Metal — the same reasoning extract.py::_gpu_visible()
    uses in reverse (it checks for /dev/nvidia0 rather than asking TF, because
    Metal is the only GPU a Mac can report).

    tensorflow-metal has a documented history of destabilizing training on
    this project: NaNs within ~10 epochs in one case (03_train/CLAUDE.md), and
    a shipped model's training loss climbing for 20+ epochs after an early
    minimum in another (test_config_preview, 2026-09-17) — both cleared
    immediately on CPU with no other change. Rather than let that surface
    silently in a loss curve nobody is watching (the shipped model has none),
    refuse to start.
    """
    # list_logical_devices (not list_physical_devices) so --cpu / BUZZDETECT_NO_GPU
    # hiding it via set_visible_devices([], 'GPU') is respected: physical devices
    # lists hardware presence regardless of visibility and would still trip this.
    if platform.system() == 'Darwin' and tf.config.list_logical_devices('GPU'):
        raise RuntimeError(
            'TensorFlow sees a GPU on macOS -- this is the Metal PluggableDevice. '
            'tensorflow-metal has produced non-finite or unstable training loss on '
            'this project (see 03_train/CLAUDE.md, "Metal GPU NaN"); CPU has not, '
            'and is ~GPU speed for this size of probe. Re-run with --cpu (or set '
            'BUZZDETECT_NO_GPU=1) before importing tensorflow -- CUDA_VISIBLE_DEVICES '
            'does not touch the Metal device.'
        )


_RUN_CONFIG_DEFAULTS = {'set': 'medium', 'embeddername': 'yamnet', 'translation': 'general'}


def _resolve_run_config(dir_model_full, setname, embeddername, name_translation, aug_dirnames):
    """Fill in unspecified --set/--embedder/--translation/--augment from the
    model's existing config_model.json, and guard against a model name
    accumulating folds trained under different pipeline settings.

    can_write() only asks "does this fold's dir exist" -- it has no way to
    know the pool a resumed run intends differs from the one that produced
    what's already there. A --name reused with the wrong --set (or embedder,
    translation, augmentation) would silently keep old folds and add new ones
    trained on a different pool, and folds_sx.csv/the shipped model would mix
    both without complaint. An arg left as None (not passed on the command
    line) inherits from disk rather than a hardcoded default, so a bare
    `--name X --train-shipped` resumes X's own pipeline instead of falling
    back to whatever main.py's argparse defaults happen to be. An arg that
    *is* passed and disagrees with disk still errors -- that's the only way
    a genuine pipeline change is distinguished from an accidental default.

    Runs before any fold trains, so the mismatch is caught before touching
    data; _train_one later overwrites this same path for the shipped model
    with the fuller config_model.json, whose identity fields still agree with
    what was just resolved.
    """
    path = os.path.join(dir_model_full, 'config_model.json')
    existing = None
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)

    requested = {'set': setname, 'embeddername': embeddername, 'translation': name_translation}
    resolved = {}
    for key, val in requested.items():
        if val is not None:
            resolved[key] = val
        elif existing is not None and existing.get(key) is not None:
            resolved[key] = existing[key]
        else:
            resolved[key] = _RUN_CONFIG_DEFAULTS[key]
    resolved['aug_dirnames'] = aug_dirnames if aug_dirnames is not None else (
        (existing or {}).get('aug_dirnames', []))

    if existing is not None:
        mismatched = {k: (existing.get(k), v) for k, v in resolved.items()
                      if existing.get(k) != v}
        if mismatched:
            detail = '\n'.join(f'  {k}: on disk {old!r} != requested {new!r}'
                               for k, (old, new) in mismatched.items())
            raise ValueError(
                f'{dir_model_full} already holds folds trained under different '
                f'settings:\n{detail}\nA model name must not mix folds trained '
                f'under different pipelines -- use a different --name, or delete '
                f'{dir_model_full} to start over.'
            )
    else:
        os.makedirs(dir_model_full, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(resolved, f)

    return resolved['set'], resolved['embeddername'], resolved['translation'], resolved['aug_dirnames']


def train_set(name, embeddername, setname, name_translation,
              epochs=400, aug_dirnames=None, verbose=False,
              assume_yes=False, stop_tol=0.01, skip_cv=False, train_shipped=False,
              only_folds=None, surprisal=True, dropout=0.0):
    _forbid_metal()
    dir_model_full = os.path.join(cfg.DIR_MODELS, name)
    setname, embeddername, name_translation, aug_dirnames = _resolve_run_config(
        dir_model_full, setname, embeddername, name_translation, aug_dirnames)
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

    dir_folds = os.path.join(dir_model_full, SUBDIR_FOLDS)

    # CV: hold out one 'rotate' fold at a time, train on the other 'rotate'
    # folds plus every 'train' fold. The held-out fold doubles as the
    # val_loss monitor whose curve feeds the shipped model's epoch count
    # (_consensus_epoch) — a within-fold split would leak site identity into
    # that signal, and dedicating a second fold to it would cost another
    # deployment. Fold model binaries are not kept, only their scores and
    # training artifacts, archived under dir_folds.
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
            # Nothing to validate or score against — a legitimate state if
            # every label in this deployment is ignored or excluded, but it
            # can't take a turn as the held-out fold.
            print(f'{tag}: no usable frames under this translation; skipping rotation')
            continue

        result, model = _train_one(
            dir_model, modelname, embeddername, setname, name_translation,
            data, epochs, aug_dirnames, verbose,
            held_out, save_binary=False, dropout=dropout,
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
              f"{_format_sens(sens)}, rss {_rss_gb():.1f}G", flush=True)

        # Each fold builds a fresh model; without this the GPU allocations pile up
        # across folds and a later fold dies with "Dst tensor is not initialized".
        del model
        # data.train_tf (_Fp16Batches) holds the whole fold's packed embedding
        # array (~9 GB on `large`); dropping only `model` left it and this fold's
        # `data` reachable until the next `data = _load_data(...)` reassignment,
        # so the outgoing and incoming fold's arrays briefly coexisted and OOM
        # killed the process on `large`'s bigger pools. Clear it explicitly first.
        data.train_tf = data.val_tf = data.val_eval = None
        del data
        tf.keras.backend.clear_session()
        gc.collect()

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
    # out, so there is nothing clean left to monitor during its own training —
    # instead its epoch count is read off the pooled rotation val_loss curves
    # (_consensus_epoch), falling back to the median per-fold best epoch, and
    # to the raw --epochs budget (with a warning) only if no fold results
    # exist at all. This is always attempted, regardless of what trained the
    # rotations: every rotation now runs the same fixed budget, so every
    # curve on disk is full-length and safe for _consensus_epoch to read (see
    # the comment above the rotation fit() call).
    if not summary_rows:
        epochs_shipped = epochs
        print(f'[{name}] WARNING: no fold results to take an epoch count from; '
              f'training the shipped model for the full {epochs} epochs, '
              f'unmonitored. This risks overfitting — run the CV rotations '
              f'first so the shipped model can read a stopping point from them.')
    else:
        n_curves = sum(1 for r in summary_rows if r.get('val_loss_curve'))
        epochs_shipped = _consensus_epoch(summary_rows, stop_tol) if n_curves else None
        if epochs_shipped is not None:
            print(f'[{name}] shipped epoch count {epochs_shipped} '
                  f'(consensus val_loss curve over {n_curves}/{len(folds_rotate)} '
                  f'rotate folds, stop_tol={stop_tol})')
        else:
            epochs_shipped = int(round(np.median([r['best_epoch'] for r in summary_rows])))
            print(f'[{name}] shipped epoch count {epochs_shipped} '
                  f'(median per-fold best epoch over {len(summary_rows)}/'
                  f'{len(folds_rotate)} rotate folds; no curves on disk)')

    folds_shipped = folds_rotate + folds_train_always
    data = _load_data(setname, embeddername, folds_shipped, name_translation,
                      aug_dirnames, val_fold=None)
    result, model = _train_one(
        dir_model_full, name, embeddername, setname, name_translation,
        data, epochs, aug_dirnames, verbose,
        None, save_binary=True, epochs_shipped=epochs_shipped,
        stop_tol=stop_tol, dropout=dropout,
    )

    if result is None:
        return

    print(f'[shipped] {name}: {epochs_shipped} fixed epochs on '
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

    # Suggested per-class thresholds, from the rotations' held-out predictions
    # (not the shipped model, which has no held-out audio of its own), into
    # config_model.json -- where 04_deploy/export_onnx.py picks them up -- and a
    # README skeleton to fill in. See thresholds.py.
    write_model_card(dir_model_full, name)
