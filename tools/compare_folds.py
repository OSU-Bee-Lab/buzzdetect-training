"""Paired per-fold comparison between two trained models' folds_sx.csv.

LOOP.md step 4 asks for this on every experiment: "join the two folds_sx.csv
files on fold and look at the deltas and how many folds moved which way." This
is that join, done the same way every time.

    conda run -n buzzdetect-train python tools/compare_folds.py <baseline> <exp>
    conda run -n buzzdetect-train python tools/compare_folds.py <baseline> <exp> --fpr 0.01

No TensorFlow: reads folds_sx.csv only, nothing here loads a model.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import config as cfg

FNAME_SX_SUMMARY = 'folds_sx.csv'
TOTAL_ROW = 'total'


def resolve_dir_model(name):
    """A model is usually a bare name under this repo's models/, but an
    experiment's model lives under a worktree's own models/ dir (not
    symlinked from main — see setup_worktree.sh), so also accept a direct
    path to a model dir for cross-worktree comparisons."""
    direct = os.path.join(name, FNAME_SX_SUMMARY)
    if os.path.exists(direct):
        return name
    return os.path.join(cfg.DIR_MODELS, name)


def _read_sx(name, fpr):
    """One model's folds_sx.csv, rows for this FPR target, total row dropped."""
    dir_model = resolve_dir_model(name)
    path = os.path.join(dir_model, FNAME_SX_SUMMARY)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found — has {name!r} finished training?')
    sx = pd.read_csv(path)
    here = sx[(sx['fpr'] == fpr) & (sx['fold'] != TOTAL_ROW)]
    if here.empty:
        raise ValueError(f'{path} has no rows at fpr {fpr}; available: '
                         f'{sorted(sx["fpr"].unique())}')
    return here


def read_headline(name, fpr=0.005):
    """The total row's sensitivity — the mean over folds, the headline."""
    dir_model = resolve_dir_model(name)
    path = os.path.join(dir_model, FNAME_SX_SUMMARY)
    if not os.path.exists(path):
        return None
    sx = pd.read_csv(path)
    total = sx[(sx['fpr'] == fpr) & (sx['fold'] == TOTAL_ROW)]
    return total.iloc[0]['sensitivity'] if len(total) else None


def compare_folds(baseline, exp, fpr=0.005):
    base = _read_sx(baseline, fpr)
    other = _read_sx(exp, fpr)

    # frames_val is absent for models resummarized without their summary.json
    cols = ['fold', 'sensitivity'] + (['frames_val'] if 'frames_val' in base else [])
    merged = base[cols].merge(
        other[['fold', 'sensitivity']], on='fold', suffixes=('_base', '_exp'),
    )
    merged['delta'] = merged['sensitivity_exp'] - merged['sensitivity_base']
    merged = merged.sort_values('delta').reset_index(drop=True)

    return merged


def format_report(merged, baseline, exp, fpr):
    lines = [merged.to_string(index=False)]

    n_up = int((merged['delta'] > 0).sum())
    n_down = int((merged['delta'] < 0).sum())
    n_flat = int((merged['delta'] == 0).sum())
    lines.append('')
    lines.append(
        f'{n_up} folds up, {n_down} down, {n_flat} flat '
        f'(mean delta {merged["delta"].mean():+.4f})'
    )

    head_base = read_headline(baseline, fpr)
    head_exp = read_headline(exp, fpr)
    if head_base is not None and head_exp is not None:
        lines.append(
            f'sensitivity (mean over folds) @ fpr{fpr}: {baseline} {head_base:.3f} -> {exp} {head_exp:.3f} '
            f'({head_exp - head_base:+.3f})'
        )

    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', help='model name under models/, e.g. cv-baseline\'s cv_baseline')
    parser.add_argument('exp', help='model name under models/ to compare against baseline')
    parser.add_argument('--fpr', type=float, default=0.005, choices=[0.001, 0.005, 0.01])
    args = parser.parse_args()

    merged = compare_folds(args.baseline, args.exp, args.fpr)
    print(format_report(merged, args.baseline, args.exp, args.fpr))
