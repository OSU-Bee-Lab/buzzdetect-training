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


BUZZ_CLASS = 'ins_buzz'

# ---------------------------------------------------------------- loudness
#
# How loud the annotator judged a buzz, read off a suffix in the raw label.
# The vocabulary is what the annotations actually use, surveyed 2026-09-11 over
# 1842 buzz annotations in `01_annotate/Even Sample/data`:
#
#   (no suffix)   1375   e.g. ins_buzz_medium, ins_buzz_low, ins_buzz_pollination
#   _quiet         406
#   _loud           39
#   _background     14
#   _faint           8
#
# Note the bare labels' own suffixes (_high/_medium/_low) are PITCH, not
# loudness -- ins_buzz_medium_quiet is a medium-pitch buzz that is quiet. Only
# the markers below are read as loudness; everything else is TIER_UNTAGGED.
#
# There is deliberately no `normal`: the annotations do not use one, and an
# untagged buzz is "not yet worked up", not "judged normal". Tagging is in
# progress and TIER_UNTAGGED should shrink toward nothing.
TIER_FAINT = 'faint'
TIER_QUIET = 'quiet'
TIER_BACKGROUND = 'background'
TIER_UNTAGGED = 'untagged'
TIER_NORMAL = 'normal'
TIER_LOUD = 'loud'

# Ascending by how audible the frame is, which is also the order the tiers are
# reported in. A sample carrying several buzz labels takes the MAXIMUM tier
# over them -- a frame is only as hard as its most audible buzz.
#
# `untagged` sits above the faint end so an unknown is never mistaken for a
# faint one, and below `normal`/`loud` so an unknown never promotes a frame
# past a known one. `background` is placed just under `untagged`: it describes
# a buzz sitting behind the foreground of the mix, which is a statement about
# audibility, but a weaker one than `quiet`.
TIERS = (TIER_FAINT, TIER_QUIET, TIER_BACKGROUND, TIER_UNTAGGED, TIER_NORMAL, TIER_LOUD)
_TIER_RANK = {t: i for i, t in enumerate(TIERS)}

# Suffix -> tier. A label carrying two markers is a bad annotation; it takes
# the louder, so a frame is never dropped from the score by accident.
TIER_MARKERS = (
    ('_faint', TIER_FAINT),
    ('_quiet', TIER_QUIET),
    ('_background', TIER_BACKGROUND),
    ('_normal', TIER_NORMAL),
    ('_loud', TIER_LOUD),
)

# ---------------------------------------------------------------- the policy
#
# THE ONE PLACE that decides which tiers leave the headline sensitivity. Every
# number downstream derives from it, and `03_train/resummarize.py` rebuilds any
# model's folds_sx.csv from predictions.csv without retraining -- so changing
# this line re-scores the whole era for free. That is exactly why
# predictions.csv stores the OBSERVED tier rather than a scored/not-scored
# boolean: the policy is a judgment call and has to stay revisable.
#
# Luke's rule (2026-09-11): a buzz needing audio filtering and an expert ear to
# perceive is below what an operator could reasonably ask buzzdetect for, so
# missing one is not a false negative and catching one is not a credit.
# `faint` is included on that reasoning -- it is unambiguously fainter than
# `quiet`, and the two were stripped together by the annotation pipeline until
# this era. `background` is NOT excluded: it describes position in the mix
# rather than audibility, and keeping it is the conservative choice.
#
# Excluded frames still TRAIN, as ordinary ins_buzz positives. They are really
# there, and calling faint buzz a negative would teach the model that faint
# buzz is background.
TIERS_EXCLUDED_FROM_HEADLINE = (TIER_FAINT, TIER_QUIET)

# What a frame with no buzz at all gets. Kept out of TIERS: it is not a tier of
# buzz, it is the absence of one, and must never enter a per-tier mean.
TIER_NONE = ''


def label_tier(label):
    """The loudness tier one raw buzz label declares, or TIER_UNTAGGED."""
    low = str(label).lower()
    hits = [t for marker, t in TIER_MARKERS if marker in low]
    return max(hits, key=_TIER_RANK.__getitem__) if hits else TIER_UNTAGGED


def buzz_tier(labels_raw, labels_translate, buzz_class=BUZZ_CLASS):
    """This sample's loudness tier: the max over its buzz labels, or TIER_NONE.

    `labels_raw` and `labels_translate` are the parallel lists dataset.py
    builds -- translate_labels preserves order and length -- so this reads the
    raw label (which carries the marker) while deciding buzz-ness from the
    translated one (which is what `correct` is built from).
    """
    tiers = [label_tier(r) for r, t in zip(labels_raw, labels_translate) if t == buzz_class]
    return max(tiers, key=_TIER_RANK.__getitem__) if tiers else TIER_NONE


def scored_buzz(labels_raw, labels_translate, buzz_class=BUZZ_CLASS):
    """True if this sample's buzz counts toward the headline sensitivity."""
    tier = buzz_tier(labels_raw, labels_translate, buzz_class)
    return tier != TIER_NONE and tier not in TIERS_EXCLUDED_FROM_HEADLINE


def can_write(dir_model):
    if not os.path.exists(os.path.join(dir_model, 'config_model.json')):
        return True

    if os.path.basename(dir_model) == 'test':
        return True

    if not os.listdir(dir_model):
        return True

    return False

