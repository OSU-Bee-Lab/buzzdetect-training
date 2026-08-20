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

import numpy as np
import pandas as pd

from metrics import metrics_by_group, metrics_at_fpr

# The training path's target, deliberately one number. metrics.py keeps a
# three-target default of its own; everything here passes this instead.
FPR_TARGETS = (0.005,)

FNAME_SX_SUMMARY = 'folds_sx.csv'
FNAME_SX_BYFOLD = 'folds_sx_byfold.csv'
FNAME_PREDICTIONS = 'predictions.csv'


def _fold_sens(df, fprs):
    """Per-fold sensitivity and threshold at each target FPR, threshold set
    within the fold, plus how many negative frames each target actually rests
    on.

    sensitivity and threshold are NaN where the fold cannot reach the target.
    Two ways that happens: no buzz frames to be sensitive about, and — the one
    metrics_at_fpr will not tell you — too few negative frames for the target
    FPR to correspond to even one of them. In that second case metrics_at_fpr
    still returns a number, by interpolating between "no negatives above the
    line" and "one negative above the line". That is a value invented inside a
    single frame, so it is discarded here rather than averaged into a
    headline. A fold's own sx.csv keeps metrics_at_fpr's answer untouched.
    """
    correct = df['correct'].astype(bool)
    n_neg = int((~correct).sum())
    at_fpr = metrics_at_fpr(metrics_by_group(df[['activation_ins_buzz', 'correct']]), fprs)
    sens = at_fpr.set_index('fpr')['sensitivity'].copy()
    thresh = at_fpr.set_index('fpr')['threshold'].copy()
    neg_frames = {f: int(np.floor(f * n_neg)) for f in fprs}
    for f in fprs:
        if neg_frames[f] < 1:
            sens[f] = np.nan
            thresh[f] = np.nan
    return sens, thresh, neg_frames


def summarize_sx(predictions, fprs=FPR_TARGETS):
    """One row per target FPR.

    `predictions` is every rotating fold's held-out predictions concatenated,
    with a 'fold' column — what _collect_fold_results returns.

    Columns:
      sensitivity_mean        mean over folds, each tuned to the target FPR on
                              its own audio. The number.
      threshold_mean/median   the per-fold thresholds that number rests on —
                              what to actually set, and how much folds disagree
      folds_scored            folds that could reach this FPR, of folds_total
      buzz_frames_scored      buzz frames those folds hold, of buzz_frames_total
      neg_frames_fold_median  negative frames a typical fold's own threshold
                              rests on — how much evidence the row has
    """
    by_fold = {}
    for fold, df in predictions.groupby('fold'):
        sens, thresh, neg_frames = _fold_sens(df, fprs)
        by_fold[fold] = (sens, thresh, int(df['correct'].sum()), neg_frames)

    rows = []
    for f in fprs:
        sens = np.array([s[f] for s, _, _, _ in by_fold.values()], dtype=float)
        thresh = np.array([t[f] for _, t, _, _ in by_fold.values()], dtype=float)
        buzz = np.array([b for _, _, b, _ in by_fold.values()], dtype=float)
        neg = np.array([n[f] for _, _, _, n in by_fold.values()], dtype=float)
        ok = ~np.isnan(sens)
        rows.append({
            'fpr': f,
            'sensitivity_mean': round(sens[ok].mean(), 3) if ok.any() else np.nan,
            'threshold_mean': round(thresh[ok].mean(), 3) if ok.any() else np.nan,
            'threshold_median': round(np.median(thresh[ok]), 3) if ok.any() else np.nan,
            'folds_scored': int(ok.sum()),
            'folds_total': len(by_fold),
            'buzz_frames_scored': int(buzz[ok].sum()),
            'buzz_frames_total': int(buzz.sum()),
            'neg_frames_fold_median': int(np.median(neg[ok])) if ok.any() else 0,
        })
    return pd.DataFrame(rows)


def summarize_sx_byfold(predictions, fprs=FPR_TARGETS):
    """One row per (fold, target FPR): the threshold and sensitivity that
    fold's own audio sets, for operators who want the breakdown rather than
    the mean — e.g. to see how much thresholds disagree across deployments."""
    rows = []
    for fold, df in predictions.groupby('fold'):
        sens, thresh, neg_frames = _fold_sens(df, fprs)
        for f in fprs:
            rows.append({
                'fold': fold,
                'fpr': f,
                'threshold': thresh[f],
                'sensitivity': sens[f],
                'neg_frames': neg_frames[f],
            })
    return pd.DataFrame(rows)


def _num(v):
    return 'n/a' if pd.isna(v) else f'{v:.3f}'


def format_sx_report(name, sx, sx_byfold=None):
    """The headline block. One line per target FPR, plus what it rests on.

    If `sx_byfold` (summarize_sx_byfold's output) is given, also lists each
    fold's own threshold and sensitivity — there is no single threshold to
    ship, only a distribution of them, and this is that distribution.
    """
    lines = []
    for _, r in sx.iterrows():
        folds = int(r['folds_total'])
        lines.append(f'\n[{name}] sens@fpr{r["fpr"]:.1%} per deployment: {_num(r["sensitivity_mean"])}')
        lines.append(f'  mean over {int(r["folds_scored"])}/{folds} held-out fold(s), each '
                     f'tuned to {r["fpr"]:.1%} FPR on its own audio '
                     f'(~{int(r["neg_frames_fold_median"])} negative frames set each threshold)')
        lines.append(f'  threshold: mean {_num(r["threshold_mean"])}, median {_num(r["threshold_median"])} '
                     f'— what each fold\'s own audio set, not one number to ship')
        if r['folds_scored'] < folds:
            lines.append(f'  {folds - int(r["folds_scored"])} fold(s) had no buzz or too few '
                         f'negative frames to reach this FPR; covers '
                         f'{int(r["buzz_frames_scored"])} of {int(r["buzz_frames_total"])} buzz frames')

        if sx_byfold is not None:
            rows = sx_byfold[sx_byfold['fpr'] == r['fpr']].sort_values('fold')
            for _, fr in rows.iterrows():
                lines.append(f'    {fr["fold"]}: threshold {_num(fr["threshold"])}, '
                             f'sensitivity {_num(fr["sensitivity"])} '
                             f'(~{int(fr["neg_frames"])} negative frames)')

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
