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
from utils import git_branch
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
    # training folds with frames: the site adversary's classes (TRUNK_ADV)
    n_sites: int = 0

# Inference chunk for fold scoring; a deep trunk tail (depth 6) OOMs the 4 GB
# card at 1024. Chunking leaves logits unchanged (training=False, frozen BN).
_SCORE_CHUNK = int(os.environ.get('SCORE_CHUNK', 1024))


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
    # Pin the pool to host memory: left to itself, from_tensor_slices places the
    # whole array on the GPU as one constant, and moderate's yamnet pool OOMs a
    # 4 GB card before the first epoch. Batches still move to the GPU for fit.
    with tf.device('/CPU:0'):
        return (
            tf.data.Dataset.from_tensor_slices((emb_np, tgt_np))
            .cache().shuffle(size_shuffle).batch(size_batch).prefetch(tf.data.AUTOTUNE)
        )


# TRUNK_ADV=<weight>: site-adversarial tail (site-adv). A head predicts which
# training fold a frame came from, behind a gradient reversal of this weight,
# so the fine-tuned tail is pushed to drop site identity. 0/unset: off.
_ADV = float(os.environ.get('TRUNK_ADV', 0))


def _targets(data):
    """Frame-level float32 targets for the trunk pipelines. Under TRUNK_ADV a
    trailing column carries each frame's training-fold index (`Sample.site`, set
    in _load_data; -1 for the validation fold, which the adversary never sees)."""
    tgt = np.concatenate(
        [np.tile(s.target_array, (s.frames, 1)) for s in data]).astype(np.float32)
    if _ADV:
        site = np.concatenate([np.full(s.frames, s.site, np.float32) for s in data])
        tgt = np.concatenate([tgt, site[:, None]], axis=1)
    return tgt


# TRUNK_ADV_MODE: how the tail is turned against the site head.
#   reverse (default): gradient reversal; the tail maximises the adversary's
#     cross-entropy. Unbounded: site-adv-w01/w03/w10 all diverged on it.
#   confuse: the tail minimises KL(uniform || adversary's prediction), which is
#     bounded below (0 when the adversary is at chance); the adversary itself
#     trains on a stop-gradient copy of the code, so neither loss reaches the
#     other's weights.
#   entropy: as confuse, but the tail minimises ln(n) - H(adversary's
#     prediction), bounded on both sides ([0, ln n]). KL(uniform || p) grows
#     without limit as the head rules sites out; site-conf-w01 flattened the
#     pooled code on it.
_ADV_MODE = os.environ.get('TRUNK_ADV_MODE', 'reverse')
if _ADV_MODE not in ('reverse', 'confuse', 'entropy'):
    raise ValueError(f'TRUNK_ADV_MODE is reverse, confuse or entropy, not {_ADV_MODE}')
# TRUNK_ADV_SCOPE: which training frames the tail's site term acts on (confuse
# and entropy only; the adversary itself always trains on every frame).
#   all (default), or nonbuzz: frames with no ins_buzz label, so the term cannot
#   push against the site/label dependence in the pool (the Hard Negatives folds
#   hold no buzz).
_ADV_SCOPE = os.environ.get('TRUNK_ADV_SCOPE', 'all')
if _ADV_SCOPE not in ('all', 'nonbuzz'):
    raise ValueError(f'TRUNK_ADV_SCOPE is all or nonbuzz, not {_ADV_SCOPE}')
if _ADV_SCOPE != 'all' and _ADV_MODE == 'reverse':
    raise ValueError('TRUNK_ADV_SCOPE needs TRUNK_ADV_MODE=confuse or entropy')
# TRUNK_ADV_RAMP=<epochs>: the tail's site weight climbs linearly from 0 to
# TRUNK_ADV over this many epochs (confuse and entropy only). 0/unset: no ramp.
_ADV_RAMP = int(os.environ.get('TRUNK_ADV_RAMP', 0))
# TRUNK_GATE=<epoch>:<sens>: stop the CV after its first rotation if that fold's
# per-epoch sens@fpr0.005 monitor sat below <sens> at <epoch>. A collapsed class
# head shows there and not in val_loss; the fold's results stay on disk.
_GATE = os.environ.get('TRUNK_GATE')


class _RampWeight(tf.keras.callbacks.Callback):
    """Sets the tail's site weight at the start of each epoch (TRUNK_ADV_RAMP)."""

    def __init__(self, var, target, epochs):
        super().__init__()
        self.var, self.target, self.epochs = var, target, epochs

    def on_epoch_begin(self, epoch, logs=None):
        self.var.assign(self.target * min(1.0, epoch / self.epochs))


class _GradReverse(tf.keras.layers.Layer):
    """Identity forward, gradient scaled by -weight backward."""

    def __init__(self, weight, **kw):
        super().__init__(**kw)
        self.weight = weight

    def call(self, x):
        return (1.0 + self.weight) * tf.stop_gradient(x) - self.weight * x


class _SiteConfuse(tf.keras.layers.Layer):
    """Site head for TRUNK_ADV_MODE=confuse. Output is [adversary logits |
    confusion logits]: the same Dense(units, relu) -> Dense(n_sites), once on
    stop_gradient(code) with live weights (trains the adversary only) and once
    on the code with the weights stopped (its gradient reaches the tail only)."""

    def __init__(self, n_sites, units=256, **kw):
        super().__init__(**kw)
        self.n_sites, self.units = n_sites, units

    def build(self, shape):
        self.w1 = self.add_weight(name='w1', shape=(shape[-1], self.units), initializer='glorot_uniform')
        self.b1 = self.add_weight(name='b1', shape=(self.units,), initializer='zeros')
        self.w2 = self.add_weight(name='w2', shape=(self.units, self.n_sites), initializer='glorot_uniform')
        self.b2 = self.add_weight(name='b2', shape=(self.n_sites,), initializer='zeros')

    def call(self, code):
        def head(x, w1, b1, w2, b2):
            return tf.matmul(tf.nn.relu(tf.matmul(x, w1) + b1), w2) + b2
        live = [tf.cast(v, code.dtype) for v in (self.w1, self.b1, self.w2, self.b2)]
        return tf.concat([head(tf.stop_gradient(code), *live),
                          head(code, *[tf.stop_gradient(v) for v in live])], axis=-1)


def _add_site_adversary(model, n_sites, loss_classes, buzz_index):
    """(fit model, loss, metrics, callbacks) for a trunk head with a site adversary.

    The fit model's output is [class logits | site logits]; `model` itself
    shares every class-path weight and stays the one that is scored. The site
    terms are masked where the site index is -1, so val_loss is the class loss."""
    n_classes = model.output_shape[-1]
    code = model.layers[-1].input  # the pooled trunk code the class Dense reads
    confuse = _ADV_MODE in ('confuse', 'entropy')
    # the tail's site weight; a variable so TRUNK_ADV_RAMP can move it per epoch
    weight = tf.Variable(0.0 if _ADV_RAMP else _ADV, trainable=False, dtype=tf.float32)
    callbacks = [_RampWeight(weight, _ADV, _ADV_RAMP)] if confuse and _ADV_RAMP else []
    if confuse:
        site = _SiteConfuse(n_sites, name='site_confuse')(code)
    else:
        x = _GradReverse(_ADV, name='site_reverse')(code)
        x = tf.keras.layers.Dense(256, activation='relu', name='site_hidden')(x)
        site = tf.keras.layers.Dense(n_sites, name='site_logits')(x)
    fit_model = tf.keras.Model(
        model.input, tf.keras.layers.Concatenate()([model.output, site]), name=model.name)

    def _site_terms(y_true, y_pred):
        idx = tf.cast(y_true[:, n_classes], tf.int32)
        mask = tf.cast(idx >= 0, tf.float32)
        logits = tf.cast(y_pred[:, n_classes:n_classes + n_sites], tf.float32)
        return tf.maximum(idx, 0), logits, mask

    def loss(y_true, y_pred):
        idx, logits, mask = _site_terms(y_true, y_pred)
        site_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=idx, logits=logits)
        if confuse:
            logp = tf.nn.log_softmax(tf.cast(y_pred[:, n_classes + n_sites:], tf.float32))
            if _ADV_MODE == 'entropy':
                # ln n - H(p): 0 at chance, ln n when the head is certain
                term = float(np.log(n_sites)) + tf.reduce_sum(tf.exp(logp) * logp, axis=-1)
            else:
                # KL(uniform || p) = -mean_k log p_k - ln n; 0 at chance
                term = -tf.reduce_mean(logp, axis=-1) - float(np.log(n_sites))
            if _ADV_SCOPE == 'nonbuzz':
                term *= tf.cast(y_true[:, buzz_index] < 0.5, tf.float32)
            site_loss += weight * term
        return loss_classes(y_true[:, :n_classes], y_pred[:, :n_classes]) + site_loss * mask

    def accuracy(y_true, y_pred):
        hit = tf.equal(y_true[:, :n_classes] > 0.5, y_pred[:, :n_classes] > 0.0)
        return tf.reduce_mean(tf.cast(hit, tf.float32), axis=-1)

    def site_acc(y_true, y_pred):
        # the adversary's accuracy on training folds (chance is 1/n_sites);
        # 0 on the validation fold, where there is no site to predict
        idx, logits, mask = _site_terms(y_true, y_pred)
        return tf.cast(tf.equal(tf.argmax(logits, -1, output_type=tf.int32), idx), tf.float32) * mask

    return fit_model, loss, [accuracy, site_acc], callbacks


def _to_tf_lowmem(data, size_batch, free=False):
    """Wide float16 embeddings (yamnet_trunk 12288-d, yamnet_trunk_context 36864-d):
    one float16 buffer, reshuffled every epoch, cast to float32 per batch. The plain
    path holds ~5 copies, which is ~26 GB for the context arm.

    Pure tf.data, no Python in the input pipeline. Dataset.from_generator parks
    its closure (here the whole embedding array) in TensorFlow's global
    py-function registry, so every fold's data outlived the fold and the host
    ran out of RAM 2-4 folds in (trunk-ft-pitchshift, trunk-pitchshift-contrast).
    A keras PyDataset does not avoid it: Keras 3's TF backend wraps one in
    from_generator itself (measured: +1 array of RSS per fit, del/gc or not).
    The buffer is a CPU Variable filled in place by scatter_update, in blocks:
    tf.constant(ndarray) copies, doubling the peak, and a slice assign
    (emb[a:b].assign) copies the whole buffer on every call (~1 s each)."""
    n = sum(s.frames for s in data)
    width = len(data[0].embeddings[0])
    with tf.device('/CPU:0'):
        emb = tf.Variable(tf.zeros((n, width), tf.float16), trainable=False)
        at, block = 0, []
        for i, s in enumerate(data):
            block.append(np.asarray(s.embeddings, dtype=np.float16))
            if free:
                s.embeddings = None  # the float16 copy is the only one kept
            if sum(len(b) for b in block) >= 8192 or i == len(data) - 1:
                block = np.concatenate(block)
                emb.scatter_update(tf.IndexedSlices(block, tf.range(at, at + len(block))))
                at += len(block)
                block = []
        tgt = tf.constant(_targets(data))

    def gather(idx):
        idx = tf.sort(idx)
        return tf.cast(tf.gather(emb, idx), tf.float32), tf.gather(tgt, idx)

    return (
        tf.data.Dataset.range(n).shuffle(n, reshuffle_each_iteration=True)
        .batch(size_batch).map(gather).prefetch(tf.data.AUTOTUNE)
    )


class _StreamPool:
    """TRUNK_STREAM=1: the training pool as one float16 file on disk, memory-
    mapped, instead of _to_tf_lowmem's in-RAM Variable. For pools bigger than
    host RAM (a two-view trunk on `moderate` is ~18 GB of float16 against a
    23 GB host; building the lowmem Variable also holds the loaded samples
    beside it, so the peak is ~2x that).

    Samples are appended fold by fold as they load and their embeddings are
    dropped at once, so at most one fold's samples are in RAM. Row order is the
    sample order _to_tf_lowmem uses, and the dataset below draws the same
    reshuffled-every-epoch, sorted-per-batch indices, so training sees the same
    thing; only where the rows live changes. The file is unlinked as soon as it
    is mapped: the mapping keeps it readable, and nothing is left on disk after
    close() or a crash."""

    def __init__(self):
        os.makedirs(cfg.DIR_STREAM_SCRATCH, exist_ok=True)
        self.path = os.path.join(cfg.DIR_STREAM_SCRATCH, f'{os.getpid()}_{len(_streams)}.f16')
        self.f = open(self.path, 'wb')
        self.n, self.width, self.arr = 0, None, None

    def append(self, samples):
        for s in samples:
            block = np.asarray(s.embeddings, dtype=np.float16)
            self.width = self.width or block.shape[1]
            self.f.write(block.tobytes())
            self.n += len(block)
            s.embeddings = None

    def dataset(self, data, size_batch):
        self.f.close()
        self.arr = np.memmap(self.path, dtype=np.float16, mode='r', shape=(self.n, self.width))
        os.unlink(self.path)
        _streams.append(self)
        tgt = tf.constant(_targets(data))
        width = self.width

        # numpy_function keeps this closure alive in TF's py-function registry
        # past the fold (see _to_tf_lowmem), so it holds `self`, never the
        # array: close() drops the mapping whatever the registry keeps.
        def read(idx):
            return self.arr[idx]

        def gather(idx):
            idx = tf.sort(idx)
            emb = tf.numpy_function(read, [idx], tf.float16, stateful=False)
            emb.set_shape((None, width))
            return tf.cast(emb, tf.float32), tf.gather(tgt, idx)

        return (
            tf.data.Dataset.range(self.n).shuffle(self.n, reshuffle_each_iteration=True)
            .batch(size_batch).map(gather, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

    def close(self):
        self.arr = None


_streams = []


def _close_streams():
    while _streams:
        _streams.pop().close()


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

    pool = _StreamPool() if os.environ.get('TRUNK_STREAM') else None
    if pool is not None and aug_dirnames:
        raise NotImplementedError('TRUNK_STREAM does not stream augmented embeddings')
    data_train = []
    sites = []  # training folds that contributed frames, in site-index order
    for fold in folds_train:
        samples = build_fold_dataset(
            cfg.dir_embeddings_fold(setname, embeddername, fold), translation,
        )
        for s in samples:
            s.site = len(sites)
        if samples:
            sites.append(fold)
        if pool is not None:
            pool.append(samples)
        data_train += samples
    frames_train = sum(s.frames for s in data_train)

    data_val = None
    frames_val = 0
    val_eval = None
    if val_fold is not None:
        data_val = build_fold_dataset(
            cfg.dir_embeddings_fold(setname, embeddername, val_fold), translation,
        )
        frames_val = sum(s.frames for s in data_val)
        for s in data_val:
            s.site = -1
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
        data_train += load_augmented(setname, embeddername, aug_dirnames, translation, folds_train)

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

    return TrainingData(
        train_tf=(pool.dataset(data_train, size_batch) if pool is not None
                  else _to_tf(data_train, size_batch, size_shuffle, free=True)),
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
        n_sites=len(sites),
    )


def _score_fold(model, setname, embeddername, fold, translation, classes):
    """Score a trained model on a fold it never saw.

    Returns (predictions, predictions_classes), or (None, None) if the fold has
    no usable frames. `predictions` is the ins_buzz frame-level (activation,
    correct) table; `predictions_classes` carries every class's logit and
    target, read by thresholds.py for the suggested ins_buzz threshold. Every reported number is derived from this: it is the
    only per-fold result kept on disk, and sx.py and resummarize.py rebuild the
    sweeps from it on demand.
    """
    samples = build_fold_dataset(
        cfg.dir_embeddings_fold(setname, embeddername, fold), translation,
    )
    if not samples:
        return None, None

    embeddings, correct, loudness, sample_id = _eval_arrays(samples, classes)
    logits = np.concatenate([model(embeddings[i:i + _SCORE_CHUNK], training=False).numpy()
                             for i in range(0, len(embeddings), _SCORE_CHUNK)])
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
        # Fine-tunable embedder (yamnet_trunk and family): layers 13-14 (or
        # 12-14) live in the head, optionally trainable.
        model = embedder.build_head(
            len(data.classes), lr_backbone=float(os.environ.get('TRUNK_LR_BACKBONE', 0)),
            lr_head=float(os.environ.get('TRUNK_LR_HEAD', 2e-4)), dropout=dropout, name=tf_name,
            **({'hidden': int(os.environ['TRUNK_HIDDEN'])} if os.environ.get('TRUNK_HIDDEN') else {}))
        optimizer = model.optimizer
        if os.environ.get('TRUNK_ACCUM'):
            # TRUNK_BATCH=512 TRUNK_ACCUM=2 is batch 1024's update for a tail too
            # deep to fit 1024 on the 4 GB card (BN is frozen, so no batch stats
            # change; Keras averages the micro-batches, then clips). Set before
            # fit: the optimizer builds its accumulators lazily.
            optimizer.gradient_accumulation_steps = int(os.environ['TRUNK_ACCUM'])

    # Per-class weights go in the loss, not in fit(class_weight=). Keras'
    # class_weight= assumes single-label targets: for a multi-hot y it collapses
    # each sample to argmax(y) and scales the whole sample by that one scalar, so
    # a buzz frame co-occurring with an earlier-indexed class never gets
    # ins_buzz's weight at all. See train_utils.weighted_bce_loss.
    weights_ordered = [data.weight_dict[i] for i in range(len(data.classes))]
    loss, metrics = weighted_bce_loss(weights_ordered, label_smoothing=0.2), ['accuracy']
    # `model` is what gets scored and saved; under TRUNK_ADV a wider model
    # sharing its weights is the one that is fit.
    model_scored = model
    callbacks_adv = []
    if _ADV:
        if not (hasattr(embedder, 'build_head') and os.environ.get('TRUNK_FP16')):
            raise NotImplementedError('TRUNK_ADV needs a trunk embedder and TRUNK_FP16=1')
        if save_binary:
            raise NotImplementedError('TRUNK_ADV has no shipped-model path yet')
        model, loss, metrics, callbacks_adv = _add_site_adversary(
            model, data.n_sites, loss, data.classes.index('ins_buzz'))
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
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
            callbacks=[sens_callback, tf.keras.callbacks.TerminateOnNaN(), *callbacks_adv],
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
            # TRUNK_ADV instrumentation: the adversary's per-epoch accuracy at
            # naming the training fold (chance is 1/n_sites)
            **({'site_acc_curve': [float(x) for x in history.history['site_acc']],
                'n_sites': data.n_sites} if _ADV else {}),
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
        'modelname_internal': modelname,
        'branch': git_branch(),
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

    return result, model_scored


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
    gated = None  # TRUNK_GATE's message once it has tripped
    for i, held_out in enumerate([] if skip_cv else folds_scored, 1):
        folds_train = [f for f in folds_rotate if f != held_out] + folds_train_always
        dir_model = os.path.join(dir_folds, str(held_out))
        modelname = f'{name}_fold{held_out}'
        tag = f'[{i}/{len(folds_rotate)}] {held_out}'

        # summary.json is the fold's last write and what the CV summary needs;
        # a fold killed after config_model.json but before it would otherwise
        # read as trained and drop out of folds_sx.csv. Rebuild it instead.
        if not can_write(dir_model, FNAME_FOLD_SUMMARY):
            print(f'{tag}: already trained; skipping')
            continue
        if os.path.exists(dir_model) and os.listdir(dir_model):
            print(f'{tag}: incomplete fold on disk (no {FNAME_FOLD_SUMMARY}); retraining')
            shutil.rmtree(dir_model)

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

        predictions, sens = _write_predictions(
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
        # The held-out ins_buzz logit SD is there to expose a dead class head:
        # a near-constant output still lands in the normal val_loss range on
        # the label-smoothed loss (site-conf-w01: SD 0.1 against a trained
        # head's 0.65-1.2, val_loss within 0.03 of the control).
        logit_sd = float('nan') if predictions is None else predictions['activation_ins_buzz'].std()
        print(f"{tag}: {result['n_epochs']} epochs (best {result['best_epoch']}), "
              f"val_loss {result['best_val_loss']:.4f}, "
              f"buzz logit SD {logit_sd:.2f}, "
              f"{data.frames_train}/{data.frames_val} frames train/val, "
              f"{_format_sens(sens)}, host RAM {_rss_gb():.1f} GB", flush=True)

        if _GATE and held_out == folds_scored[0]:
            gate_epoch, gate_sens = _GATE.split(':')
            at = result['val_sens_fpr0.005_curve'][int(gate_epoch) - 1]
            if at < float(gate_sens):
                gated = (f'[gate] {held_out}: sens@fpr0.005 {at:.3f} at epoch {gate_epoch}, '
                         f'below {gate_sens}; CV stopped after its first rotation')
                print(gated, flush=True)
                break

        # Each fold builds a fresh model and loads its own data; without this
        # GPU allocations and the fold's arrays pile up, and a wide embedder's
        # later fold dies with "Dst tensor is not initialized" (from
        # exp/trunk-ft-v3; hit on a plain 2048-d probe in pitchshift-contrast).
        del model, data
        tf.keras.backend.clear_session()
        _close_streams()
        gc.collect()

    summary_rows, predictions_pooled = _collect_fold_results(dir_folds, [held_out] if gated else folds_scored)

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

    if gated:
        sys.exit(gated)

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

    # The suggested ins_buzz threshold, from the rotations' held-out predictions
    # (not the shipped model, which has no held-out audio of its own), into
    # config_model.json -- where 04_deploy/export_onnx.py picks them up -- and a
    # README skeleton to fill in. See thresholds.py.
    write_model_card(dir_model_full, name)
