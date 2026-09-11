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
  once made a strictly better model look half as good. It used to be written
  out anyway, as folds_pooled_metrics.csv; it no longer is. Rebuild it from
  read_fold_predictions() + metrics_by_group() if you want a pooled ROC.

Only fpr 0.005 is reported. 0.01 is too loose to be operationally useful, and
0.001 is not measurable on sets this size — it rests on a handful of negative
frames per fold, and some folds cannot reach it at all.

Every sensitivity is reported twice, at one identical threshold: `sensitivity`
counts every annotated buzz frame, and `sensitivity_exclquiet` drops the frames
whose buzz is only `_quiet`-tagged. A `_quiet` buzz is really there but needs
filtering and an expert ear to hear, so failing to catch it is not a failure of
the tool and catching it is not a credit; it simply leaves the equation. Those
frames still train, as ordinary positives. **The headline is `sensitivity_exclquiet`** (LOOP.md "Goal"): quiet buzz leaves
the equation, and the inclusive figure is its companion at the same threshold.
This module computes both and imposes neither; the policy lives in LOOP.md and
in the two tools that read this file (`tools/compare_folds.py`,
`tools/log_entry.py`).

summarize_folds writes all of this to one file, folds_sx.csv: a row per fold,
then a total. Three files used to carry the same per-fold sensitivity under
three different NaN policies; see _fold_sens for the policy that survived.
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
FNAME_PREDICTIONS = 'predictions.csv'


QUIET_COL = 'quiet'
SENS_EXCL = 'sensitivity_exclquiet'


def _has_quiet(df):
    """Whether this prediction table carries the quiet flag at all.

    Runs from before 2026-09-11 have no such column, and predictions.csv is the
    only per-fold artifact kept, so resummarize.py has to read them. Those
    models simply get NaN in the excl-quiet column rather than a number that
    pretends the split was made.
    """
    return QUIET_COL in df.columns


def _fold_sens(df, fprs):
    """One fold's threshold, sensitivity and precision at each target FPR, the
    threshold set within the fold, plus how many negative frames it rests on.

    Two sensitivities are reported, over one identical threshold:

      `sensitivity`     — every annotated buzz frame counts as a positive.
      `sensitivity_exclquiet` — frames whose buzz is only `_quiet`-tagged are
                          dropped from the scored set entirely. Those buzzes
                          need audio filtering and an expert ear to perceive,
                          which is below what an operator could reasonably ask
                          of the tool: missing one is not a false negative, and
                          catching one is not a credit. They still train, as
                          ordinary positives — they are really there, and
                          teaching the model they are background would be worse
                          than either scoring choice.

    The threshold does not move between the two. Quiet frames are positives, so
    dropping them changes neither the negative pool nor the FPR sweep; only the
    sensitivity numerator and denominator change. That is what makes the pair
    readable side by side — it is one operating point scored against two
    definitions of the target, not two models.

    All three are NaN where the fold cannot reach the target. Two ways that
    happens: no buzz frames to be sensitive about, and — the one metrics_at_fpr
    will not tell you — too few negative frames for the target FPR to
    correspond to even one of them. In that second case metrics_at_fpr still
    returns a number, by interpolating between "no negatives above the line"
    and "one negative above the line". That is a value invented inside a single
    frame, so it is discarded here rather than reported.

    precision is included because it is what an operator sees in the output,
    but read it against the fold's own buzz density, not a deployment's: a
    fold's base rate is a property of what got annotated.
    """
    correct = df['correct'].astype(bool)
    n_neg = int((~correct).sum())
    at_fpr = metrics_at_fpr(metrics_by_group(df[['activation_ins_buzz', 'correct']]), fprs)
    cols = at_fpr.set_index('fpr')[['threshold', 'sensitivity', 'precision']]

    # The excl-quiet reading: same negatives, same sweep, quiet positives gone.
    if _has_quiet(df):
        loud = df[~df[QUIET_COL].astype(bool)]
        excl = metrics_at_fpr(
            metrics_by_group(loud[['activation_ins_buzz', 'correct']]), fprs,
        ).set_index('fpr')['sensitivity']
    else:
        excl = pd.Series({f: np.nan for f in fprs})
    cols[SENS_EXCL] = [excl.get(f, np.nan) for f in cols.index]

    neg_frames = {f: int(np.floor(f * n_neg)) for f in fprs}
    for f in fprs:
        # Blank the whole row together, not just the column that went NaN on
        # its own. A fold with no buzz frames gets sensitivity NaN but
        # precision 0.0 from metrics_at_fpr, and the total row would then
        # average precision over folds whose sensitivity it could not use.
        if neg_frames[f] < 1 or pd.isna(cols.loc[f, 'sensitivity']):
            cols.loc[f] = np.nan
    return cols, neg_frames


TOTAL_ROW = 'total'


def summarize_folds(predictions, fold_facts=None, fprs=FPR_TARGETS):
    """The results table: one row per (fold, target FPR), then a `total` row.

    This is the whole metrics summary. It used to be three files —
    folds_sx.csv, folds_sx_byfold.csv and folds_summary.csv — which reported
    the same per-fold sensitivity under three different NaN policies. One
    table, one policy.

    `predictions` is every rotating fold's held-out predictions concatenated,
    with a 'fold' column. `fold_facts`, if given, maps fold -> dict of
    training-side facts (frames_val, best_epoch) to carry as extra columns;
    everything else about a rotation stays in that fold's summary.json.

    In the `total` row, counts are summed and threshold/sensitivity/precision
    are the plain mean over the folds that could reach the target. That mean is
    the headline: each deployment counts once, because the question is what a
    new deployment gets, and a new deployment is one fold. A fold that could
    not reach the target keeps its row, with NaNs, and is not averaged in —
    count the non-NaN rows to see how many folds the total rests on.
    """
    rows = []
    for fold, df in predictions.groupby('fold'):
        cols, neg_frames = _fold_sens(df, fprs)
        buzz = df['correct'].astype(bool)
        quiet = df[QUIET_COL].astype(bool) if _has_quiet(df) else None
        for f in fprs:
            rows.append({
                'fold': fold,
                'fpr': f,
                **cols.loc[f].to_dict(),
                'buzz_frames': int(buzz.sum()),
                'quiet_frames': int((buzz & quiet).sum()) if quiet is not None else np.nan,
                'neg_frames': neg_frames[f],
                **(fold_facts or {}).get(fold, {}),
            })

    table = pd.DataFrame(rows)
    means = ['threshold', 'sensitivity', SENS_EXCL, 'precision']
    sums = [c for c in ('buzz_frames', 'quiet_frames', 'neg_frames', 'frames_val') if c in table]

    totals = []
    for f in fprs:
        here = table[table['fpr'] == f]
        total = {'fold': TOTAL_ROW, 'fpr': f}
        total.update(here[means].mean().round(3).to_dict())
        # A column that is NaN everywhere (quiet_frames on a pre-2026-09-11
        # run) must stay NaN in the total rather than summing to 0.
        total.update({c: (np.nan if here[c].isna().all() else int(here[c].sum()))
                      for c in sums})
        if 'best_epoch' in here:
            # the median is what the shipped model's fixed epoch count is set to
            total['best_epoch'] = int(round(here['best_epoch'].median()))
        totals.append(total)

    table[means] = table[means].round(3)
    return pd.concat([table, pd.DataFrame(totals)], ignore_index=True)[table.columns]


def _num(v):
    return 'n/a' if pd.isna(v) else f'{v:.3f}'


def format_sx_report(name, table):
    """The headline block: the total, what it rests on, then each fold."""
    lines = []
    for fpr, here in table.groupby('fpr'):
        folds = here[here['fold'] != TOTAL_ROW]
        total = here[here['fold'] == TOTAL_ROW].iloc[0]
        scored = folds['sensitivity'].notna().sum()

        lines.append(f'\n[{name}] sens@fpr{fpr:.1%} per deployment: '
                     f'{_num(total.get(SENS_EXCL))}'
                     f'  (incl. _quiet buzz: {_num(total["sensitivity"])})')
        lines.append(f'  mean over {scored}/{len(folds)} held-out fold(s), each tuned to '
                     f'{fpr:.1%} FPR on its own audio '
                     f'(~{int(folds["neg_frames"].median())} negative frames set each threshold)')
        lines.append(f'  threshold: mean {_num(total["threshold"])} — what each fold\'s own '
                     f'audio set, not one number to ship')
        if scored < len(folds):
            lines.append(f'  {len(folds) - scored} fold(s) had no buzz or too few negative '
                         f'frames to reach this FPR; covers '
                         f'{int(folds.loc[folds["sensitivity"].notna(), "buzz_frames"].sum())} '
                         f'of {int(folds["buzz_frames"].sum())} buzz frames')
        for _, r in folds.sort_values('fold').iterrows():
            lines.append(f'    {r["fold"]}: threshold {_num(r["threshold"])}, '
                         f'sensitivity {_num(r.get(SENS_EXCL))} '
                         f'(incl. quiet {_num(r["sensitivity"])}) '
                         f'(~{int(r["neg_frames"])} negative frames)')
        quiet_total = total.get('quiet_frames')
        if pd.notna(quiet_total) and quiet_total:
            lines.append(f'  {int(quiet_total)} of {int(total["buzz_frames"])} buzz frames are '
                         f'_quiet-only: audible to an expert ear with filtering, below what an '
                         f'operator can ask for. They train as positives; the headline '
                         f'drops them from the score at the same threshold.')

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
