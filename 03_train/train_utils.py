import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config as cfg


def build_classes(translation):
    classes = translation['to'].unique().tolist()
    classes = [c for c in classes if not c in  ['ignore', np.nan, '', 'exclude']]
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

