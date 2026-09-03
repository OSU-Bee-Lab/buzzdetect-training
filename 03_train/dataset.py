import glob
import os
import pickle
import warnings

import pandas as pd

import config as cfg
from train_utils import build_classes, labels_from_path, Sample

# build_fold_dataset() is called once per fold per CV rotation (~11x per
# training run), so a fold that's genuinely and permanently empty (e.g. an
# always-excluded ident) re-triggers an identical warning on every rotation.
# Warn about a given dir_samples once per process rather than once per call.
_warned_empty_dirs = set()


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

    A label with no row at all is left unchanged rather than silently treated as
    "ignore". It still produces no target (it matches no class), but it stays
    visible: survey_untranslated() reports it before training so a missing row
    is caught as an omission rather than mistaken for a deliberate ignore.

    Args:
        labels_raw (list): The raw labels to translate.
        translation (DataFrame): A translation data frame with columns "from" and "to" containing the labels to translate from and to, respectively..

    Returns:
        list: Translated labels with NaN values removed.
    """
    labels_translated = [
        translation_dict.get(l, l)  # translate if found, else leave unchanged
        for l in labels_raw
    ]

    return labels_translated


def survey_untranslated(setname, embeddername, folds, translation):
    """Raw labels present in the extracted embeddings with no row in the
    translation's "from" column, as {label: n_files}.

    Run before training so a missing row surfaces while it's still cheap to
    fix. An untranslated label is indistinguishable from an ignored one once
    training starts — both simply produce no target.
    """
    known = set(translation['from'])
    counts = {}
    for fold in folds:
        dir_fold = cfg.dir_embeddings_fold(setname, embeddername, fold)
        for p in glob.glob(os.path.join(dir_fold, '**', '*.pickle'), recursive=True):
            for label in labels_from_path(p):
                if label not in known:
                    counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def labels_to_targets(labels_translate, classes):
    return [c in labels_translate for c in classes]


def build_fold_dataset(dir_samples, translation, labels_keep_raw=None, exclusive=False):
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

        # drop samples where any "to" label is set to exclude (case-insensitive:
        # translations/build.R writes EXCLUDE, curated rows have used exclude)
        if any(isinstance(l, str) and l.strip().lower() == 'exclude' for l in s.labels_translate):
            continue

        s.target_array = labels_to_targets(s.labels_translate, classes)

        # drop samples where every label is set to ignore
        if sum(s.target_array) == 0:
            continue

        # else, we're good to go!
        s.embeddings = read_pickle_exhaustive(s.path)
        s.frames = len(s.embeddings)
        samples_out.append(s)

    # An empty result is legitimate: every label in this fold may be genuinely
    # excluded or ignored. Warn and hand back nothing; the caller decides
    # whether a fold with no usable frames is fatal for what it's doing.
    if len(samples_out) == 0 and dir_samples not in _warned_empty_dirs:
        _warned_empty_dirs.add(dir_samples)
        labels_raw = sorted({l for s in samples for l in s.labels_raw})
        translated = translate_labels(labels_raw, translation_dict)
        deliberate = all(
            isinstance(l, str) and l.strip().lower() in ('exclude', 'ignore')
            for l in translated
        )
        if not samples:
            warnings.warn(f'no embedding files under {dir_samples}')
        elif not deliberate:
            # A directory whose labels all translate to exclude/ignore is
            # dropped by design — no warning. Warn only when the drop is
            # unexplained (a label with no row, caught again by
            # survey_untranslated at train time).
            warnings.warn(
                f'all {len(samples)} embedding file(s) under {dir_samples} were '
                f'dropped by the translation; raw label(s) present: {labels_raw}'
            )
    return samples_out


def load_augmented(setname, embeddername, aug_dirnames, translation, train_folds):
    """Load augmented embeddings for training.

    aug_dirnames: list of augmentation subdirectory names under embeddings/<embeddername>/,
    e.g. ['augment_noise_0.05', 'augment_volume_2.5', 'augment_combine'].
    Produced by 02_set/augment.py.
    train_folds: fold IDs to load augmented embeddings for (the training pool
    of the current CV iteration).
    """
    data = []
    for aug_dirname in aug_dirnames:
        for fold in train_folds:
            dir_embed = os.path.join(cfg.dir_embeddings_augment(setname, embeddername, aug_dirname), fold)
            if not os.path.isdir(dir_embed):
                raise FileNotFoundError(f'augmented embeddings not found: {aug_dirname} fold {fold} (expected {dir_embed})')
            data += build_fold_dataset(dir_embed, translation)
    return data


ROLE_TRAIN = 'train'
ROLE_ROTATE = 'rotate'
ROLE_HOLDOUT = 'holdout'
ROLE_EXCLUDE = 'exclude'
ROLES = (ROLE_TRAIN, ROLE_ROTATE, ROLE_HOLDOUT, ROLE_EXCLUDE)


def read_fold_roles(setname, embeddername):
    """Map fold id -> role from the set's folds.csv, checked against what's extracted.

    Roles (see README): 'train' always trains and is never scored, 'rotate' is
    the leave-one-fold-out set, 'holdout' is only ever scored, 'exclude' does
    neither. Excluded folds are dropped here rather than upstream — 02_set
    embeds every fold regardless of role, so flipping a role never costs a
    re-extraction.

    folds.csv carries one row per ident, so rows sharing a fold must agree on
    its role.
    """
    path_folds = os.path.join(cfg.dir_set(setname), 'folds.csv')
    folds_df = pd.read_csv(path_folds, dtype=str)

    if 'role' not in folds_df.columns:
        warnings.warn(
            f'{path_folds} has no role column; treating every fold as '
            f'{ROLE_ROTATE!r}. Add a role column to control which folds train '
            f'and which are scored.'
        )
        folds_df['role'] = ROLE_ROTATE

    folds_df['role'] = folds_df['role'].fillna('').str.strip()

    unknown_roles = sorted(set(folds_df['role']) - set(ROLES))
    if unknown_roles:
        raise ValueError(
            f'{path_folds}: unrecognized role(s) {unknown_roles}; '
            f'expected one of {list(ROLES)}'
        )

    n_roles = folds_df.groupby('fold')['role'].nunique()
    conflicted = sorted(n_roles[n_roles > 1].index)
    if conflicted:
        raise ValueError(
            f'{path_folds}: fold(s) {conflicted} have rows with disagreeing '
            f'roles; role is a property of the fold, not the ident'
        )

    roles_all = folds_df.drop_duplicates('fold').set_index('fold')['role'].to_dict()
    roles = {f: r for f, r in sorted(roles_all.items()) if r != ROLE_EXCLUDE}

    # A fold id may itself contain path separators (deployment paths like
    # 'Chia - OSPT/2022/2022-07-26'), so resolve each one as a path rather than
    # matching against a flat listing of dir_raw.
    dir_raw = cfg.dir_embeddings_raw(setname, embeddername)
    missing = sorted(
        f for f in roles
        if not os.path.isdir(cfg.dir_embeddings_fold(setname, embeddername, f))
    )
    if missing:
        raise FileNotFoundError(
            f'no embeddings for fold(s) {missing} under {dir_raw}; '
            f'extraction is incomplete for set {setname!r} / embedder {embeddername!r}'
        )

    # Leftovers from an earlier build: a top-level entry that neither contains
    # nor sits inside any known fold's path.
    unlisted = sorted(
        d for d in os.listdir(dir_raw)
        if os.path.isdir(os.path.join(dir_raw, d))
        and not any(f == d or f.startswith(d + os.sep) for f in roles_all)
    )
    if unlisted:
        warnings.warn(
            f'{dir_raw} holds embeddings for fold(s) {unlisted} that are absent '
            f'from {path_folds}; ignoring them. Likely leftovers from an '
            f'earlier build of this set.'
        )

    return roles


def folds_by_role(roles, role):
    return [f for f, r in roles.items() if r == role]


# Note: no snip-level train/val splitter here, deliberately. Snips from one
# deployment share a recorder, a site and a background, so a within-fold split
# leaks site identity into the early-stopping signal and biases the stopping
# epoch late. Submodels early-stop on their held-out fold instead; the shipped
# model has no monitor and trains to the median of their best epochs.
