import glob
import os
import pickle
import warnings

import pandas as pd

import config as cfg
from train_utils import build_classes, labels_from_path, Sample


def read_pickle_exhaustive(path_pickle):
    elements = []
    with open(path_pickle, 'rb') as f:
        while True:
            try:
                element = pickle.load(f)
                elements.append(element)
            except EOFError:
                break
    return elements

def build_translation_dict(translation):
    return {translation['from']: translation['to'] for _, translation in translation.iterrows()}

def translate_labels(labels_raw: list, translation_dict: dict):
    """
    Translates a list of raw labels according to a translation dict, as built by build_translation_dict()..

    When a raw label has a "to" value of "ignore"  or no value, no target will be generated for that label.
    If a sample is only labeled with "ignore" or blank labels, it will be excluded from the dataset.

    When a raw label has a "to" value of "exclude", any sample matching that label will be dropped
    from the dataset, no matter what other labels are present for the sample.

    Args:
        labels_raw (list): The raw labels to translate.
        translation (DataFrame): A translation data frame with columns "from" and "to" containing the labels to translate from and to, respectively..

    Returns:
        list: Translated labels with NaN values removed.
    """
    labels_translated = [
        translation_dict.get(l, 'ignore')  # Translate if found, else leave unchanged
        for l in labels_raw
    ]

    return labels_translated


def labels_to_targets(labels_translate, classes):
    return [c in labels_translate for c in classes]


def build_fold_dataset(dir_samples, translation, labels_keep_raw=None, exclusive=False):
    # check if the folder name isn't in train, test, or validate
    if os.path.basename(dir_samples) not in ['train', 'test', 'validate']:
        warnings.warn(f'build_fold_dataset: {dir_samples} is not an expected directory name for a fold; should end in train, test, or validate.')

    translation_dict = build_translation_dict(translation)
    classes = build_classes(translation)

    samples = []
    for p in glob.glob(os.path.join(dir_samples, '**', '*.pickle'), recursive=True):
        sample = Sample(path=p)
        sample.labels_raw = labels_from_path(p)
        samples.append(sample)

    # drop paths outside of classes
    if labels_keep_raw is not None:
        samples = [s for s in samples if any([c in s.labels_raw for c in labels_keep_raw])]
    if labels_keep_raw and exclusive:
        samples = [s for s in samples if all(l in labels_keep_raw for l in s.labels_raw)]

    samples_out = []
    for s in samples:
        s.labels_translate = translate_labels(s.labels_raw, translation_dict)

        # drop samples where any "to" label is set to exclude
        if 'exclude' in s.labels_translate:
            continue

        s.target_array = labels_to_targets(s.labels_translate, classes)

        # drop samples where every label is set to ignore
        if sum(s.target_array) == 0:
            continue

        # else, we're good to go!
        s.embeddings = read_pickle_exhaustive(s.path)
        s.frames = len(s.embeddings)
        samples_out.append(s)

    if len(samples_out) == 0:
        raise ValueError(f'no samples found in {dir_samples}')
    return samples_out


def load_augment_noise(setname, translation, name_noise):
    augment_df = pd.read_csv(os.path.join(cfg.SET_DIR, setname, 'augment_noise_' + name_noise + '.csv'))
    data_augment = []

    props = augment_df['prop'].unique()
    for prop in props:
        classes_keep = augment_df[augment_df['prop'] == prop]['class'].unique()
        augmenttype = 'augment_noise_prop' + str(prop)
        data_augment += build_fold_dataset(setname=setname, fold_keep='train', translation=translation, augmenttype=augmenttype, labels_keep_raw=classes_keep, exclusive=False)

    return data_augment


def load_augment_volume(setname, translation, name_volume):
    augment_df = pd.read_csv(os.path.join(cfg.SET_DIR, setname, 'augment_volume_' + name_volume + '.csv'))
    data_augment = []

    props = augment_df['prop'].unique()
    for prop in props:
        classes_keep = augment_df[augment_df['prop'] == prop]['class'].unique()
        augmenttype = 'augment_volume_prop' + str(prop)
        data_augment += build_fold_dataset(setname=setname, fold_keep='train', translation=translation,
                                           augmenttype=augmenttype, labels_keep_raw=classes_keep, exclusive=False)

    return data_augment
