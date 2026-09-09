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


def weighted_bce_loss(weights, label_smoothing=0.2, margin_spec=None):
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
        margin_spec: `None` (default) leaves the loss byte-identical to the
            plain weighted BCE. Otherwise `(idx_target, idx_condition, lam, m)`
            adds a class-conditional hinge

                lam * 1[y[idx_condition] and not y[idx_target]] * relu(z + m)

            on the *raw* (unsmoothed) labels, pushing the target class's logit
            at least `m` below zero on frames carrying the conditioning class
            but not the target. Used to spend extra loss on the confuser class
            that actually sets the low-FPR operating point (`mech_auto` for
            `ins_buzz`) rather than on every negative equally.

            This is deliberately a *per-frame* function of `(y_true, y_pred)`:
            no top-k, no within-batch rank, no batch-composition dependence.
            That is what keeps the compiled loss a valid `val_loss` monitor,
            and it is the difference from the archived `tail-loss` OHEM term,
            whose batch-local hard-negative set made `val_loss` meaningless on
            a single small validation fold.
    """
    weight_tensor = tf.constant(weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_raw = tf.cast(y_true, y_pred.dtype)
        y_true = y_raw
        if label_smoothing:
            y_true = y_raw * (1.0 - label_smoothing) + 0.5 * label_smoothing

        per_neuron = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true, logits=y_pred, pos_weight=weight_tensor,
        )
        per_sample = tf.reduce_mean(per_neuron, axis=-1)

        if margin_spec is not None:
            i_target, i_condition, lam, m = margin_spec
            # Raw labels, not the smoothed ones: this selects frames by what
            # they *are*, so smoothing would turn the indicator into 0.1/0.9.
            mask = y_raw[:, i_condition] * (1.0 - y_raw[:, i_target])
            hinge = tf.nn.relu(y_pred[:, i_target] + m)
            per_sample = per_sample + lam * mask * hinge

        return per_sample

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

