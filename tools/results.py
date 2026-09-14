"""The Results section of an experiment's notes.md, built from disk (LOOP.md
step 5), so no number is copied by hand.

    python tools/results.py <matched control model> <experiment model>

Prints markdown: the per-fold table in notes.md's template (control, exp,
delta, ± SD, buzz events), the headline with its delta SD, the inclusive
figure, the folds that missed the target FPR, and the loudness-tier deltas.
Sensitivities come from folds_sx.csv (compare_folds.py); the SDs from
eval_sampling_sd.py's paired event bootstrap. The interpretation under the
table is still yours to write.
"""
# TensorFlow must be imported before pandas/pyarrow: see eval_sampling_sd.py's
# header, which this imports. Keep this import first.
import tensorflow  # noqa: F401  -- imported for load-order side effect only

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_folds import (compare_folds, resolve_dir_model, FNAME_SX_SUMMARY,  # noqa: E402
                           SENS_COL, SENS_COL_INCL, TOTAL_ROW)
from eval_sampling_sd import fold_rows, headline_sd  # noqa: E402

TIERS = ('loud', 'normal', 'untagged', 'background', 'quiet', 'faint')


def total_row(name, fpr):
    sx = pd.read_csv(os.path.join(resolve_dir_model(name), FNAME_SX_SUMMARY))
    return sx[(sx['fpr'] == fpr) & (sx['fold'] == TOTAL_ROW)].iloc[0]


def fmt(x, spec='.3f'):
    return '' if x is None or pd.isna(x) else format(x, spec)


def results(control, exp, fpr=0.005, n_boot=2000):
    merged = compare_folds(control, exp, fpr).sort_values('fold')
    col = merged.attrs['sens_col']
    boot = {r['fold']: r for r in fold_rows(resolve_dir_model(control), resolve_dir_model(exp), fpr, n_boot)}

    # the last path component names a deployment (1_29, 1_150); fall back to
    # the full fold if that ever stops being unique
    labels = [f.rsplit('/', 1)[-1] for f in merged['fold']]
    if len(set(labels)) < len(labels):
        labels = list(merged['fold'])

    out = [f'| fold | baseline sens@fpr{fpr} | this exp | delta | ± SD | buzz events |',
           '|---|---|---|---|---|---|']
    for label, (_, r) in zip(labels, merged.iterrows()):
        b = boot.get(r['fold'], {})
        events = r['buzz_events_exclquiet'] if 'buzz_events_exclquiet' in r else b.get('events')
        out.append(f"| {label} | {fmt(r[f'{col}_base'])} | {fmt(r[f'{col}_exp'])} | "
                   f"{fmt(r['delta'], '+.3f')} | {fmt(b.get('delta_sd'))} | {fmt(events, '.0f')} |")

    tb, te = total_row(control, fpr), total_row(exp, fpr)
    sds = [r['delta_sd'] for r in boot.values() if 'delta_sd' in r]
    # a blank ± SD means the two models' frame grids differ (eval_sampling_sd.py warns)
    dsd = f'{headline_sd(sds):.3f}' if len(sds) == len(merged) else 'n/a: frame grids differ'
    out += ['', f'- mean sens@fpr{fpr} ({col}): baseline {tb[col]:.3f} → this {te[col]:.3f} '
                f'({te[col] - tb[col]:+.3f} ± {dsd})']
    if col == SENS_COL:
        out.append(f'- inclusive ({SENS_COL_INCL}), same thresholds: {tb[SENS_COL_INCL]:.3f} → '
                   f'{te[SENS_COL_INCL]:.3f} ({te[SENS_COL_INCL] - tb[SENS_COL_INCL]:+.3f})')
    missed = [lab for lab, (_, r) in zip(labels, merged.iterrows())
              if pd.isna(r[f'{col}_base']) or pd.isna(r[f'{col}_exp'])]
    out.append(f"- folds that missed fpr {fpr}: {', '.join(missed) or 'none'}")

    out += ['', '| tier | baseline | this exp | delta | frames |', '|---|---|---|---|---|']
    for t in TIERS:
        c = f'sensitivity_{t}'
        if c not in tb or (pd.isna(tb[c]) and pd.isna(te[c])):
            continue
        out.append(f"| {t} | {fmt(tb[c])} | {fmt(te[c])} | {fmt(te[c] - tb[c], '+.3f')} | "
                   f"{fmt(tb.get(f'{t}_frames'), '.0f')} |")

    out += ['', '± SD is eval sampling only (tools/eval_sampling_sd.py); training '
                'stochasticity is larger per fold and not in it.']
    return '\n'.join(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('control', help='matched control: a name under models/, or a model dir')
    ap.add_argument('exp', help='experiment model: a name under models/, or a model dir')
    ap.add_argument('--fpr', type=float, default=0.005, choices=[0.001, 0.005, 0.01])
    ap.add_argument('--n-boot', type=int, default=2000)
    args = ap.parse_args()
    print(results(args.control, args.exp, args.fpr, args.n_boot))
