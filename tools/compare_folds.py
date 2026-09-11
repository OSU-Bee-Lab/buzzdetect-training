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

# The era's headline (LOOP.md "Goal"): sensitivity with the `_quiet`-only buzz
# frames dropped from the scored positives. `sensitivity` — every buzz frame —
# is the companion, reported beside it at the same threshold.
SENS_COL = 'sensitivity_exclquiet'
SENS_COL_INCL = 'sensitivity'


def sens_col(df, name):
    """Which sensitivity column to read, and warn if it is the fallback.

    Models trained before 2026-09-11 have no quiet flag in predictions.csv, so
    their folds_sx.csv either lacks the column or carries NaN. Those are
    previous-era models and comparing them to this era's is already invalid —
    the warning is there so a fallback never passes silently as the headline.
    """
    if name in df and df[name].notna().any():
        return name
    return None


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


def read_headline(name, fpr=0.005, column=SENS_COL):
    """The total row's sensitivity — the mean over folds, the headline.

    Reads sensitivity_exclquiet by default and falls back to the inclusive
    column, loudly, for a model that predates the quiet flag.
    """
    dir_model = resolve_dir_model(name)
    path = os.path.join(dir_model, FNAME_SX_SUMMARY)
    if not os.path.exists(path):
        return None
    sx = pd.read_csv(path)
    total = sx[(sx['fpr'] == fpr) & (sx['fold'] == TOTAL_ROW)]
    if not len(total):
        return None
    col = sens_col(total, column)
    if col is None:
        col = SENS_COL_INCL
        if column != SENS_COL_INCL:
            print(f'WARNING: {name} has no {column}; falling back to '
                  f'{SENS_COL_INCL}. It predates the _quiet split, so it is '
                  f'not a valid comparator for this era.', file=sys.stderr)
    return total.iloc[0][col]


def read_headline_pair(name, fpr=0.005):
    """(excl-quiet, inclusive) headline, for reporting both."""
    return read_headline(name, fpr, SENS_COL), read_headline(name, fpr, SENS_COL_INCL)


def compare_folds(baseline, exp, fpr=0.005):
    base = _read_sx(baseline, fpr)
    other = _read_sx(exp, fpr)

    # Pair on the headline column where both sides carry it; a pre-2026-09-11
    # model has no quiet flag, and sens_col returning None is the signal.
    col = SENS_COL if (sens_col(base, SENS_COL) and sens_col(other, SENS_COL)) else SENS_COL_INCL
    if col == SENS_COL_INCL:
        print(f'WARNING: pairing on {SENS_COL_INCL} — one side predates the '
              f'_quiet split and is not a valid comparator for this era.',
              file=sys.stderr)

    # frames_val is absent for models resummarized without their summary.json
    cols = ['fold', col] + (['frames_val'] if 'frames_val' in base else [])
    merged = base[cols].merge(
        other[['fold', col]], on='fold', suffixes=('_base', '_exp'),
    )
    merged['delta'] = merged[f'{col}_exp'] - merged[f'{col}_base']
    merged = merged.sort_values('delta').reset_index(drop=True)
    merged.attrs['sens_col'] = col

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

    col = merged.attrs.get('sens_col', SENS_COL)
    head_base = read_headline(baseline, fpr, col)
    head_exp = read_headline(exp, fpr, col)
    if head_base is not None and head_exp is not None:
        lines.append(
            f'{col} (mean over folds) @ fpr{fpr}: {baseline} {head_base:.3f} -> {exp} {head_exp:.3f} '
            f'({head_exp - head_base:+.3f})   <- the headline'
        )
    if col == SENS_COL:
        incl_base = read_headline(baseline, fpr, SENS_COL_INCL)
        incl_exp = read_headline(exp, fpr, SENS_COL_INCL)
        if incl_base is not None and incl_exp is not None:
            lines.append(
                f'{SENS_COL_INCL} (incl. _quiet buzz), same threshold: '
                f'{incl_base:.3f} -> {incl_exp:.3f} ({incl_exp - incl_base:+.3f})'
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
