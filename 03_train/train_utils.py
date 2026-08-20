import os
import re
from dataclasses import dataclass

import numpy as np
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



# Frames either side of the target frame handed to the probe alongside it.
# 0 restores the frame-at-a-time baseline.
CONTEXT_FRAMES = 1


def stack_context(embeddings, k=CONTEXT_FRAMES):
    """Widen each frame's embedding with its k neighbours on either side.

    (n_frames, n_dim) -> (n_frames, n_dim * (2k + 1)), frames in time order,
    edges clamped by repeating the first and last frame. k=0 is a no-op.

    Called per snip, never across snips: frames within a pickle are contiguous
    audio, frames in different pickles are not adjacent to anything.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if k == 0:
        return embeddings
    n = len(embeddings)
    idx = np.arange(n)
    return np.concatenate(
        [embeddings[np.clip(idx + offset, 0, n - 1)] for offset in range(-k, k + 1)],
        axis=1,
    )
