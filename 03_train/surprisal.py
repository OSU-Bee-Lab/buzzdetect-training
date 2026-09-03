"""Per-frame surprisal over held-out folds, for finding bad annotations.

For every held-out frame this records -log(sigmoid(logit_k)) for each class k
the frame's annotation actually asserts -- the negative log-likelihood the
model assigns to the annotated label. A genuinely mislabelled annotation should
sit near the top: the model, trained on the other folds, disagrees with the
label. Genuinely hard-but-correct cases (faint buzzes, rare classes) also score
high, so `top_class` -- the model's argmax for the frame -- is written
alongside as a first cut at "and here is what it thinks it should be".

Written during training, one call per held-out fold right after
predictions.csv, and on by default: train_set(..., surprisal=False) or
--no-surprisal turns it off.

Output: <model>/surprisal/<ident>_surprisal.csv, one file per source-audio
ident, columns:
    start       frame start, seconds into the original audio file
    label       the annotated class this row scores (one row per class the
                frame asserts, so a multi-label frame contributes several rows)
    surprisal   -log(sigmoid(logit_label)), nats
    top_class   argmax class over all logits for this frame

Frame -> timestamp comes from the frametimes.csv the extractor writes beside
each ident's embedding pickles (02_set/extract.py). A set extracted before that
file existed has no frame times; its idents are skipped with a warning and the
rest of training is unaffected.
"""

import os
import warnings

import numpy as np
import pandas as pd

import config as cfg
from dataset import build_fold_dataset

FNAME_FRAMETIMES = 'frametimes.csv'
SUBDIR_SURPRISAL = 'surprisal'


def _frametimes_by_stem(dir_pickle):
    """{collapsed-label stem: start[] ordered by embedding row}, or None if the
    ident has no frametimes.csv."""
    path = os.path.join(dir_pickle, FNAME_FRAMETIMES)
    if not os.path.exists(path):
        return None
    ft = pd.read_csv(path)
    return {
        stem: g.sort_values('row')['start'].to_numpy()
        for stem, g in ft.groupby('label')
    }


def write_fold_surprisal(dir_model_full, model, setname, embeddername, fold,
                         translation, classes):
    """Score `fold` with `model` and write its per-ident surprisal CSVs under
    <dir_model_full>/surprisal/. Returns the number of idents written.

    `model` is the fold's own out-of-fold submodel during CV, or the shipped
    model for a holdout fold -- either way it never trained on `fold`.
    """
    dir_fold = cfg.dir_embeddings_fold(setname, embeddername, fold)
    samples = build_fold_dataset(dir_fold, translation)
    if not samples:
        return 0

    classes_arr = np.asarray(classes)
    rows_by_ident = {}
    idents_no_frametimes = set()

    for s in samples:
        dir_pickle = os.path.dirname(s.path)
        stem = os.path.basename(s.path).rsplit('.', 1)[0]
        ident = os.path.relpath(dir_pickle, dir_fold)

        by_stem = _frametimes_by_stem(dir_pickle)
        starts = None if by_stem is None else by_stem.get(stem)
        if starts is None or len(starts) != s.frames:
            idents_no_frametimes.add(ident)
            continue

        logits = model(np.asarray(s.embeddings, dtype=np.float32), training=False).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        top_class = classes_arr[logits.argmax(axis=1)]

        for j in [k for k in range(len(classes)) if s.target_array[k]]:
            d = -np.log(np.clip(probs[:, j], 1e-12, None))
            for i in range(s.frames):
                rows_by_ident.setdefault(ident, []).append(
                    (float(starts[i]), classes[j], float(d[i]), str(top_class[i]))
                )

    for ident, rows in rows_by_ident.items():
        df = pd.DataFrame(rows, columns=['start', 'label', 'surprisal', 'top_class'])
        df = df.sort_values(['start', 'label']).reset_index(drop=True)
        path_out = os.path.join(dir_model_full, SUBDIR_SURPRISAL, ident + '_surprisal.csv')
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        df.to_csv(path_out, index=False)

    if idents_no_frametimes:
        warnings.warn(
            f'surprisal: {len(idents_no_frametimes)} ident(s) in fold {fold!r} have '
            f'no usable {FNAME_FRAMETIMES} (set extracted before frame times were '
            f'recorded, or a row-count mismatch); skipped. Re-extract the set to '
            f'include them. First: {sorted(idents_no_frametimes)[0]!r}'
        )
    return len(rows_by_ident)
