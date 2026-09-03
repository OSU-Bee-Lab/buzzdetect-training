"""Per-frame class activations and loss over held-out folds, for finding bad
annotations and hard negatives.

For every held-out frame this records the model's sigmoid activation for each
class and the multi-label loss of the frame against its annotation. The model
was trained on the other folds, so a frame where it confidently disagrees with
the label -- in either direction -- is a candidate mislabel or a hard case:

    - high activation on a class the frame does not assert -> a missing
      annotation, or a hard negative the model wants to call positive
    - low activation on a class the frame does assert -> a faint/hard positive,
      or an annotation span drawn wider than the sound it covers

`loss` is the mean over all classes of the per-class binary cross-entropy
against hard 0/1 targets (not the label-smoothed targets training uses -- this
is a labeling audit, so it scores disagreement with the annotation as written).
Multi-label frames fall out for free: every asserted class contributes a y=1
term and every other class a y=0 term. Sort by `loss` to rank frames; read the
`activation_*` columns to see which class drove it and which way.

Written during training, one call per held-out fold right after
predictions.csv, and on by default: train_set(..., surprisal=False) or
--no-surprisal turns it off.

Output: <model>/surprisal/<ident>_surprisal.csv, one file per source-audio
ident, one row per held-out frame, ordered by `start`:
    start               frame start, seconds into the original audio file
    label               the classes the frame asserts, ';'-joined
    activation_<class>  sigmoid activation, one column per class (the class set
                        varies between models, so the columns are built from
                        the model's own class list)
    loss                mean per-class BCE of the frame against its annotation

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
    """Score `fold` with `model` and write its per-ident activation/loss CSVs
    under <dir_model_full>/surprisal/. Returns the number of idents written.

    `model` is the fold's own out-of-fold submodel during CV, or the shipped
    model for a holdout fold -- either way it never trained on `fold`.
    """
    dir_fold = cfg.dir_embeddings_fold(setname, embeddername, fold)
    samples = build_fold_dataset(dir_fold, translation)
    if not samples:
        return 0

    act_cols = [f'activation_{c}' for c in classes]
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
        probs_c = np.clip(probs, 1e-12, 1.0 - 1e-12)

        target = np.asarray(s.target_array, dtype=np.float64)
        bce = -(target * np.log(probs_c) + (1.0 - target) * np.log(1.0 - probs_c))
        loss = bce.mean(axis=1)

        label = ';'.join(classes[k] for k in range(len(classes)) if s.target_array[k])

        for i in range(s.frames):
            rows_by_ident.setdefault(ident, []).append(
                (float(starts[i]), label, *probs[i].tolist(), float(loss[i]))
            )

    for ident, rows in rows_by_ident.items():
        df = pd.DataFrame(rows, columns=['start', 'label', *act_cols, 'loss'])
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
