"""Rebuild folds_sx.csv for models already trained, from predictions on disk.

A training run writes this itself. This exists for models trained before
folds_sx.csv did, and for re-reading a run without paying for the CV again —
every input it needs is already in models/<name>/folds/<fold>/predictions.csv.

The training-side columns (frames_val, best_epoch) come from each fold's
summary.json, which is also what makes a rerun resumable. Models trained
before those existed simply get a table without them.

    conda run -n buzzdetect-train python 03_train/resummarize.py           # all
    conda run -n buzzdetect-train python 03_train/resummarize.py <name>... # some

No TensorFlow: nothing here loads a model.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from sx import (
    summarize_folds, format_sx_report, read_fold_predictions, FNAME_SX_SUMMARY,
)

SUBDIR_FOLDS = 'folds'
FNAME_FOLD_SUMMARY = 'summary.json'


def _fold_facts(dir_folds, folds):
    """frames_val and best_epoch per fold, from the summary.json a training run
    leaves in each rotation's directory. Missing files are simply absent."""
    facts = {}
    for fold in folds:
        path = os.path.join(dir_folds, fold, FNAME_FOLD_SUMMARY)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            row = json.load(f)
        facts[fold] = {k: row[k] for k in ('frames_val', 'best_epoch') if k in row}
    return facts


def resummarize(name):
    """Write folds_sx.csv for one model; return it, or None if the model has no
    fold predictions to read."""
    dir_model = os.path.join(cfg.DIR_MODELS, name)
    dir_folds = os.path.join(dir_model, SUBDIR_FOLDS)
    predictions = read_fold_predictions(dir_folds)
    if predictions is None:
        return None
    sx = summarize_folds(predictions, _fold_facts(dir_folds, predictions['fold'].unique()))
    sx.to_csv(os.path.join(dir_model, FNAME_SX_SUMMARY), index=False)
    return sx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='*', metavar='MODEL',
                        help='model names; default is every model with fold predictions')
    args = parser.parse_args()

    names = args.names or sorted(
        d for d in os.listdir(cfg.DIR_MODELS)
        if os.path.isdir(os.path.join(cfg.DIR_MODELS, d, SUBDIR_FOLDS))
    )

    for name in names:
        sx = resummarize(name)
        if sx is None:
            print(f'[{name}] no fold predictions on disk; skipped')
            continue
        print(format_sx_report(name, sx))
