"""Threshold sweeps over frame-level predictions.

metrics_by_group turns (activation, correct) pairs into a full ROC/PR sweep;
metrics_at_fpr and metrics_at_precision read one operating point off it. 03_train
scores every held-out fold through these, and sx.py aggregates the results — see
its docstring for which reading is the endpoint and why.

Formerly 04_test/metrics.py. The functions that read stage 4's on-disk inference
results went with stage 4; these three never depended on it.
"""

import numpy as np
import pandas as pd


def metrics_by_group(results_join, groups=None):
    if groups is None:
        groups = []

    group_cols = groups + ['activation_ins_buzz']

    agg = results_join.groupby(group_cols).agg(
        frames_here=('correct', 'count'),
        buzzes_here=('correct', 'sum'),
    ).reset_index()
    agg['nonbuzzes_here'] = agg['frames_here'] - agg['buzzes_here']
    agg = agg.sort_values(groups + ['activation_ins_buzz'], ascending=[True] * len(groups) + [False])

    def cumstats(g):
        g = g.copy()
        g['frames_above'] = g['frames_here'].cumsum()
        g['buzzes_above'] = g['buzzes_here'].cumsum()
        g['nonbuzzes_above'] = g['nonbuzzes_here'].cumsum()
        total_buzzes = g['buzzes_here'].sum()
        total_nonbuzzes = g['nonbuzzes_here'].sum()
        g['precision'] = g['buzzes_above'] / g['frames_above']
        g['sensitivity'] = g['buzzes_above'] / total_buzzes
        g['fpr'] = g['nonbuzzes_above'] / total_nonbuzzes
        return g

    result = agg.groupby(groups, group_keys=False).apply(cumstats) if groups else cumstats(agg)

    return result.rename(columns={'activation_ins_buzz': 'threshold'})[
        groups + ['threshold', 'precision', 'sensitivity', 'fpr']
    ]


PRECISION_TARGETS = (0.90, 0.95, 0.99)


def metrics_at_precision(metrics_df, precisions=PRECISION_TARGETS):
    """Threshold, sensitivity, and FPR at each target precision via linear interpolation.

    Returns a DataFrame with columns: precision, threshold, sensitivity, fpr.
    Rows where the model never reaches the target precision have NaN values.
    """
    rows = []
    for prec_target in precisions:
        delta = metrics_df['precision'] - prec_target
        under = metrics_df[delta <= 0]
        over = metrics_df[delta >= 0]

        if under.empty or over.empty:
            rows.append({'precision': prec_target, 'threshold': np.nan,
                         'sensitivity': np.nan, 'fpr': np.nan})
            continue

        u = under.loc[under['precision'].sub(prec_target).abs().idxmin()]
        o = over.loc[over['precision'].sub(prec_target).abs().idxmin()]

        if u['threshold'] == o['threshold']:
            rows.append({'precision': prec_target, 'threshold': u['threshold'],
                         'sensitivity': u['sensitivity'], 'fpr': u['fpr']})
            continue

        prec_range = o['precision'] - u['precision']
        d = (prec_target - u['precision']) / prec_range
        rows.append({
            'precision': prec_target,
            'threshold': u['threshold'] + (o['threshold'] - u['threshold']) * d,
            'sensitivity': u['sensitivity'] + (o['sensitivity'] - u['sensitivity']) * d,
            'fpr': u['fpr'] + (o['fpr'] - u['fpr']) * d,
        })

    return pd.DataFrame(rows).round(3)


FPR_TARGETS = (0.001, 0.005, 0.01)


def metrics_at_fpr(metrics_df, fprs=FPR_TARGETS):
    """Threshold, sensitivity, and precision at each target FPR via linear interpolation.

    FPR and sensitivity are properties of the classifier, invariant to a
    deployment's base rate — unlike precision, which mixes in how much of the
    deployment's audio actually contains the target class. This is the
    base-rate-free counterpart to metrics_at_precision.

    Returns a DataFrame with columns: fpr, threshold, sensitivity, precision.
    Rows where the model never reaches the target FPR have NaN values.
    """
    rows = []
    for fpr_target in fprs:
        delta = metrics_df['fpr'] - fpr_target
        under = metrics_df[delta <= 0]
        over = metrics_df[delta >= 0]

        if under.empty or over.empty:
            rows.append({'fpr': fpr_target, 'threshold': np.nan,
                         'sensitivity': np.nan, 'precision': np.nan})
            continue

        u = under.loc[under['fpr'].sub(fpr_target).abs().idxmin()]
        o = over.loc[over['fpr'].sub(fpr_target).abs().idxmin()]

        if u['threshold'] == o['threshold']:
            rows.append({'fpr': fpr_target, 'threshold': u['threshold'],
                         'sensitivity': u['sensitivity'], 'precision': u['precision']})
            continue

        fpr_range = o['fpr'] - u['fpr']
        d = (fpr_target - u['fpr']) / fpr_range
        rows.append({
            'fpr': fpr_target,
            'threshold': u['threshold'] + (o['threshold'] - u['threshold']) * d,
            'sensitivity': u['sensitivity'] + (o['sensitivity'] - u['sensitivity']) * d,
            'precision': u['precision'] + (o['precision'] - u['precision']) * d,
        })

    return pd.DataFrame(rows).round(3)
