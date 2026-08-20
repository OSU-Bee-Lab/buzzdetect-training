"""sx — sensitivity at a fixed FPR, on held-out deployments.

sens@FPR is not a property of a model on its own. It is what the model catches
once a threshold is drawn, and where that threshold goes is a deployment
decision. buzzdetect ships no threshold — operators are told to find their own
— so every fold is scored at a threshold set on its own audio, and the primary
number is the plain mean across folds: what a deployment gets, whatever
deployment it happens to be.

That is an oracle. Putting a fold at exactly 0.5% FPR uses that fold's labels,
which an operator does not have. Read it as the ceiling on operator tuning, not
as a promise.

Two things this deliberately does not report.

  A buzz-weighted mean — total buzzes found over total buzzes present. It
  weights each fold by annotated hours times buzz density, so a full-bloom
  mustard field speaks over a quiet soybean patch, and the mix of effort and
  ecology in that weighting makes it a clean answer to no question. It is the
  lower-variance estimator, which matters only if the folds share one true
  sensitivity — and the whole premise here is that they don't.

  A pooled read, one global threshold across every fold at once. It answers
  "one shipped threshold, everywhere", which is not how the tool is used, and
  it confounds discrimination with how portable a model's score scale is. It
  once made a strictly better model look half as good. The full pooled sweep
  still lands in folds_pooled_metrics.csv for ROC plots and
  metrics_at_precision; it is not an endpoint.

Only fpr 0.005 is reported. 0.01 is too loose to be operationally useful, and
0.001 is not measurable on sets this size — it rests on a handful of negative
frames per fold, and some folds cannot reach it at all.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '04_test'))
from metrics import metrics_by_group, metrics_at_fpr

# The training path's target, deliberately one number. 04_test/metrics.py keeps
# its own three-target default for the stale stage-4 readers; everything under
# 03_train passes this instead.
FPR_TARGETS = (0.005,)

FNAME_SX_SUMMARY = 'folds_sx.csv'
FNAME_PREDICTIONS = 'predictions.csv'


def _fold_sens(df, fprs):
    """Per-fold sensitivity at each target FPR, threshold set within the fold,
    plus how many negative frames each target actually rests on.

    NaN where the fold cannot reach the target. Two ways that happens: no buzz
    frames to be sensitive about, and — the one metrics_at_fpr will not tell
    you — too few negative frames for the target FPR to correspond to even one
    of them. In that second case metrics_at_fpr still returns a number, by
    interpolating between "no negatives above the line" and "one negative above
    the line". That is a value invented inside a single frame, so it is
    discarded here rather than averaged into a headline. A fold's own sx.csv
    keeps metrics_at_fpr's answer untouched.
    """
    correct = df['correct'].astype(bool)
    n_neg = int((~correct).sum())
    sens = metrics_at_fpr(metrics_by_group(df[['activation_ins_buzz', 'correct']]), fprs)
    sens = sens.set_index('fpr')['sensitivity'].copy()
    neg_frames = {f: int(np.floor(f * n_neg)) for f in fprs}
    for f in fprs:
        if neg_frames[f] < 1:
            sens[f] = np.nan
    return sens, neg_frames


def summarize_sx(predictions, fprs=FPR_TARGETS):
    """One row per target FPR.

    `predictions` is every rotating fold's held-out predictions concatenated,
    with a 'fold' column — what _collect_fold_results returns.

    Columns:
      sens_persite            mean over folds, each tuned to the target FPR on
                              its own audio. The number.
      folds_scored            folds that could reach this FPR, of folds_total
      buzz_frames_scored      buzz frames those folds hold, of buzz_frames_total
      neg_frames_persite_median  negative frames a typical fold's own threshold
                              rests on — how much evidence the row has
    """
    by_fold = {}
    for fold, df in predictions.groupby('fold'):
        sens, neg_frames = _fold_sens(df, fprs)
        by_fold[fold] = (sens, int(df['correct'].sum()), neg_frames)

    rows = []
    for f in fprs:
        sens = np.array([s[f] for s, _, _ in by_fold.values()], dtype=float)
        buzz = np.array([b for _, b, _ in by_fold.values()], dtype=float)
        neg = np.array([n[f] for _, _, n in by_fold.values()], dtype=float)
        ok = ~np.isnan(sens)
        rows.append({
            'fpr': f,
            'sens_persite': round(sens[ok].mean(), 3) if ok.any() else np.nan,
            'folds_scored': int(ok.sum()),
            'folds_total': len(by_fold),
            'buzz_frames_scored': int(buzz[ok].sum()),
            'buzz_frames_total': int(buzz.sum()),
            'neg_frames_persite_median': int(np.median(neg[ok])) if ok.any() else 0,
        })
    return pd.DataFrame(rows)


def _num(v):
    return 'n/a' if pd.isna(v) else f'{v:.3f}'


def format_sx_report(name, sx):
    """The headline block. One line per target FPR, plus what it rests on."""
    lines = []
    for _, r in sx.iterrows():
        folds = int(r['folds_total'])
        lines.append(f'\n[{name}] sens@fpr{r["fpr"]:.1%} per deployment: {_num(r["sens_persite"])}')
        lines.append(f'  mean over {int(r["folds_scored"])}/{folds} held-out fold(s), each '
                     f'tuned to {r["fpr"]:.1%} FPR on its own audio '
                     f'(~{int(r["neg_frames_persite_median"])} negative frames set each threshold)')
        if r['folds_scored'] < folds:
            lines.append(f'  {folds - int(r["folds_scored"])} fold(s) had no buzz or too few '
                         f'negative frames to reach this FPR; covers '
                         f'{int(r["buzz_frames_scored"])} of {int(r["buzz_frames_total"])} buzz frames')

    lines.append("  an oracle: each threshold is placed using that fold's labels, so read this "
                 "as the ceiling on operator tuning")
    lines.append('  per-fold spread understates uncertainty about a new deployment '
                 '(training pools overlap heavily)\n')
    return '\n'.join(lines)


def read_fold_predictions(dir_folds):
    """Every rotating fold's predictions on disk, concatenated with a 'fold'
    column. Walks rather than globbing a fixed depth — fold names are
    path-like and of no fixed depth."""
    frames = []
    for path in sorted(glob.glob(os.path.join(dir_folds, '**', FNAME_PREDICTIONS), recursive=True)):
        df = pd.read_csv(path)
        df['fold'] = os.path.relpath(os.path.dirname(path), dir_folds)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None
