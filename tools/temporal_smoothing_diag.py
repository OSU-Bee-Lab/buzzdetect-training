"""Temporal smoothing diagnostic (IDEAS.md item 6). No training, no extraction.

Reads a model's predictions.csv per fold, applies a rolling median/mean to
activation_ins_buzz within each fold (row order == sample order, so a window
can cross a sample boundary near the edges -- see IDEAS.md item 6's caveat:
frames inside one snip nearly always share a label, so this is a *pre-
registered upper bound*, not a candidate for adoption. Its value is the
per-fold shape: which folds have spiky, isolated FPs (smoothing should help)
vs. block FPs (smoothing should hurt, e.g. the 1_95 jet).

Usage: python tools/temporal_smoothing_diag.py <model dir> [--windows 1,3,5,9]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '03_train'))
import sx  # noqa: E402


def smooth_predictions(predictions, window, stat):
    if window == 1:
        return predictions
    out = predictions.copy()
    roll = out.groupby('fold')['activation_ins_buzz'].rolling(
        window=window, center=True, min_periods=1,
    )
    smoothed = roll.median() if stat == 'median' else roll.mean()
    out['activation_ins_buzz'] = smoothed.reset_index(level=0, drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model_dir')
    ap.add_argument('--windows', default='1,3,5,9')
    ap.add_argument('--stat', choices=['mean', 'median'], default='median')
    args = ap.parse_args()

    dir_folds = os.path.join(args.model_dir, 'folds')
    predictions = sx.read_fold_predictions(dir_folds)
    if predictions is None:
        raise SystemExit(f'no predictions.csv found under {dir_folds}')

    windows = [int(w) for w in args.windows.split(',')]

    fold_rows = {}
    total_rows = {}
    for w in windows:
        sm = smooth_predictions(predictions, w, args.stat)
        table = sx.summarize_folds(sm)
        total = table[table['fold'] == sx.TOTAL_ROW].iloc[0]
        total_rows[w] = total
        for _, row in table[table['fold'] != sx.TOTAL_ROW].iterrows():
            fold_rows.setdefault(row['fold'], {})[w] = row

    print(f'\n{args.stat} smoothing, {args.model_dir}\n')
    header = 'fold'.ljust(45) + ''.join(f'w={w:<8}' for w in windows)
    print(header)
    for fold, per_w in fold_rows.items():
        line = fold.ljust(45)
        for w in windows:
            v = per_w[w][sx.SENS_EXCL]
            line += (f'{v:.3f}   '.ljust(9) if pd.notna(v) else 'n/a      ')
        print(line)
    line = 'TOTAL'.ljust(45)
    for w in windows:
        v = total_rows[w][sx.SENS_EXCL]
        line += (f'{v:.3f}   '.ljust(9) if pd.notna(v) else 'n/a      ')
    print(line)


if __name__ == '__main__':
    main()
