import os
import re
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import pandas as pd

import config as cfg


def build_classes(translation):
    classes = translation['to'].unique().tolist()
    # The sentinels are matched case-insensitively: translations/build.R writes
    # them uppercase, but hand-curated rows have used lowercase.
    classes = [
        c for c in classes
        if not (c is np.nan or not isinstance(c, str) or c.strip().lower() in ['', 'ignore', 'exclude'])
    ]
    classes = sorted(classes)

    return classes


@dataclass
class Sample:
    path: str
    frames: int = None
    labels_raw: list = None
    labels_translate: list = None
    target_array: list = None
    classes: list = None
    embeddings: list = None


def build_weights(data_train: list[Sample], classes):
    weights = pd.DataFrame()
    weights['target'] = range(len(classes))
    weights['class'] = classes

    target_grid = np.vstack([s.target_array for s in data_train])

    weights['samples'] = np.sum(target_grid, axis=0)
    samples_total = weights['samples'].sum()

    n_classes_present = sum(weights['samples'] > 0)

    def weighter(samples_class):
        if samples_class == 0:
            return 1

        weight = samples_total / (samples_class * n_classes_present)

        return weight

    weights['weight'] = [weighter(f) for f in weights['samples']]

    return weights


def weighted_bce_loss(weights, label_smoothing=0.2):
    """Per-class weighted binary crossentropy for a multi-hot, multi-label
    target — the loss `class_weight=` was meant to express, applied correctly.

    Keras' `class_weight=` argument to `model.fit()` assumes single-label
    targets: for a multi-hot `y` it collapses each sample to one class via
    `argmax(y, axis=-1)` and scales that sample's entire loss by one scalar,
    picked by class *index* order. Any sample where the true buzz label
    co-occurs with an earlier-sorted class (e.g. `ambient_background`) never
    gets `ins_buzz`'s weight applied at all, and no class ever gets weighted
    per-neuron. `weighted_cross_entropy_with_logits` reweights only the
    positive-target term of each neuron independently via `pos_weight`, which
    is what per-class inverse-frequency weighting is supposed to mean for a
    multi-label target: negative supervision is untouched, so a rare class's
    boost doesn't also mute the negative signal on every other neuron in the
    same frame.

    Args:
        weights: per-class weight, ordered by class index (e.g. `build_weights`'s
            'weight' column).
        label_smoothing: applied to targets before the loss, matching
            `tf.keras.losses.BinaryCrossentropy`'s `label_smoothing` formula.
    """
    weight_tensor = tf.constant(weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        if label_smoothing:
            y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

        per_neuron = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true, logits=y_pred, pos_weight=weight_tensor,
        )
        return tf.reduce_mean(per_neuron, axis=-1)

    return loss


def labels_from_path(path_in):
    base = os.path.basename(path_in)
    base = os.path.splitext(base)[0]
    labels = re.split(pattern='\\+', string=base)

    return labels


def can_write(dir_model):
    if not os.path.exists(os.path.join(dir_model, 'config_model.json')):
        return True

    if os.path.basename(dir_model) == 'test':
        return True

    if not os.listdir(dir_model):
        return True

    return False



def pairwise_rank_loss(weights, buzz_index, confuser_index, lam,
                       margin=1.0, n_pairs=64, label_smoothing=0.2):
    """`weighted_bce_loss` plus a pairwise buzz-above-confuser ranking term.

    The direct successor `mech-margin` named in its own conclusion. That
    experiment penalised the confuser's logit in ABSOLUTE terms and failed
    monotonically across a 64x dose ladder, for a reason that specifies the
    replacement: **the metric reads rank only.** Every fold is thresholded on
    its own held-out audio, so pushing a confuser population below a fixed
    logit buys nothing — `mech_auto`'s *share* of threshold-setting false
    positives actually ROSE with the dose (26.5% -> 39.3%) while its absolute
    logits fell. Both populations sank; buzz sank faster.

    So the term here is pairwise and scale-free: for each buzz frame in the
    batch, sample `n_pairs` confuser frames and penalise
    `softplus(conf_logit - buzz_logit + margin)` on the `ins_buzz` neuron.
    Adding a constant to every logit leaves it unchanged, which is exactly the
    invariance the metric has and `mech-margin` lacked.

    Two historical failure modes, and why neither applies:

    - `tail-loss` (E2) was a ranking loss that failed as a **monitor artifact**:
      batch-local ranking makes the compiled loss invalid as `val_loss`, and
      early stopping read it. Run this under `--fixed-epochs`, which uses no
      monitor at all, and that mode cannot fire. Do not run it with early
      stopping.
    - "batch-local pairing is not deterministic per frame", which is why
      IDEAS proposed pairing against a frozen reference model. At this
      pipeline's `size_batch` of 65568 against ~72k training frames there are
      ~2 steps per epoch, so a batch is most of the pool and in-batch pairing
      is already near-global. The frozen-reference machinery buys little here.

    Pairs are sampled rather than enumerated: ~7.7k buzz x ~5k confuser frames
    is 38M pairs per step, where `n_pairs` per buzz frame is 493k and unbiased.

    Args:
        weights: per-class weight, ordered by class index (as `weighted_bce_loss`).
        buzz_index: column of the `ins_buzz` logit.
        confuser_index: column of the confuser class (e.g. `mech_auto`).
        lam: weight on the ranking term. 0 reduces exactly to `weighted_bce_loss`.
        margin: logit gap a pair must clear before it stops contributing.
        n_pairs: confuser frames sampled per buzz frame per step.
        label_smoothing: as `weighted_bce_loss`.
    """
    base = weighted_bce_loss(weights, label_smoothing=label_smoothing)

    def loss(y_true, y_pred):
        base_loss = base(y_true, y_pred)

        y_true = tf.cast(y_true, y_pred.dtype)
        # Raw membership, not the smoothed target: smoothing is for the base
        # loss. A frame labelled both buzz and confuser is in NEITHER pool — it
        # cannot inform an ordering between them.
        is_buzz = y_true[:, buzz_index] > 0.5
        is_conf = y_true[:, confuser_index] > 0.5
        b = tf.cast(tf.logical_and(is_buzz, tf.logical_not(is_conf)), y_pred.dtype)
        c = tf.cast(tf.logical_and(is_conf, tf.logical_not(is_buzz)), y_pred.dtype)

        scores = y_pred[:, buzz_index]

        # Sample confuser frames by categorical draw over the membership mask
        # rather than tf.boolean_mask. boolean_mask produces a BOUNDED-DYNAMIC
        # dimension (f32[<=n]) that XLA's dynamic padder refuses to broadcast
        # into an add -- it fails at compile time inside model.fit with
        # "Dynamic input unexpectedly found for unsupported instruction".
        # tools/smoke_model.py caught exactly that; keep the shapes static.
        logits_c = tf.math.log(c + 1e-12)[None, :]
        idx = tf.random.categorical(logits_c, n_pairs, dtype=tf.int32)[0]
        sampled = tf.gather(scores, idx)                       # (n_pairs,)

        # (n, n_pairs): every buzz frame against n_pairs confuser frames.
        violation = sampled[None, :] - scores[:, None] + margin
        per_pair = tf.nn.softplus(violation) * b[:, None]
        n_buzz = tf.reduce_sum(b)

        # With no buzz or no confuser frames in the batch there is no ordering
        # to express. Gated rather than tf.cond'd: an all-zero mask still gives
        # log(1e-12) uniformly, which samples fine and is then zeroed here.
        gate = tf.cast(
            tf.logical_and(n_buzz > 0, tf.reduce_sum(c) > 0), y_pred.dtype)
        rank = gate * tf.reduce_sum(per_pair) / (n_buzz * n_pairs + 1e-8)

        # base_loss is per-sample (Keras reduces it); the scalar term broadcasts
        # onto it, so the compiled loss keeps the shape Keras expects.
        return base_loss + lam * rank

    return loss
