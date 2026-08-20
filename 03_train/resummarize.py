"""Rebuild folds_sx.csv for models already trained, from predictions on disk.

A training run writes this itself. This exists for models trained before
folds_sx.csv did, and for re-reading a run without paying for the CV again —
every input it needs is already in models/<name>/folds/<fold>/predictions.csv.

    conda run -n buzzdetect-train python 03_train/resummarize.py           # all
    conda run -n buzzdetect-train python 03_train/resummarize.py <name>... # some

No TensorFlow: nothing here loads a model.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from sx import (
    summarize_sx, summarize_sx_byfold, format_sx_report, read_fold_predictions,
    FNAME_SX_SUMMARY, FNAME_SX_BYFOLD,
)

SUBDIR_FOLDS = 'folds'


def resummarize(name):
    """Write folds_sx.csv and folds_sx_byfold.csv for one model; return
    (sx, sx_byfold), or None if the model has no fold predictions to read."""
    dir_model = os.path.join(cfg.DIR_MODELS, name)
    predictions = read_fold_predictions(os.path.join(dir_model, SUBDIR_FOLDS))
    if predictions is None:
        return None
    sx = summarize_sx(predictions)
    sx.to_csv(os.path.join(dir_model, FNAME_SX_SUMMARY), index=False)
    sx_byfold = summarize_sx_byfold(predictions)
    sx_byfold.to_csv(os.path.join(dir_model, FNAME_SX_BYFOLD), index=False)
    return sx, sx_byfold


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
        result = resummarize(name)
        if result is None:
            print(f'[{name}] no fold predictions on disk; skipped')
            continue
        sx, sx_byfold = result
        print(format_sx_report(name, sx, sx_byfold))
