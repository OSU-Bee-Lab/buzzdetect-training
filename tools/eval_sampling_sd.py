"""Bootstrap SD of sens@fpr on a trained model's own eval frames.

The headline metric places each fold's threshold on ~0.5% of that fold's
negative frames -- between 1 and 30 frames, depending on the fold -- and scores
it against that fold's buzz. This asks how much of the run-to-run noise floor is
that sampling, rather than training stochasticity: it resamples each fold's
evidence (with replacement) and recomputes the metric. No training and no model
is loaded; it reads predictions.csv off disk and runs in seconds.

**It resamples buzz events, not frames.** A buzz event spans many frames -- a
26-frame run is one bee heard for 26 seconds -- so frames within an event are
nothing like independent, and a frame bootstrap reads a fold as having ten times
the evidence it has. This script used to do exactly that, and understated the
per-fold SD by up to ~8x (`1_29`: 0.015 reported against 0.116 actual, because
its 1972 buzz frames are 14 events). Every number it printed before 2026-09-13
was optimistic; the headline row was least wrong, the per-fold rows most.

Two readings, and the gap between them is the point:

  headline SD   ~0.01 -- eight folds' independent wobbles partly cancel in the
                mean, which is why the mean is the number to read.
  per-fold SD   ~0.03-0.13 -- one fold is single-digit-to-low-tens of events.
                A per-fold delta smaller than about twice this is not evidence
                of anything, however clean the story it tells.

And this is only half the uncertainty. It holds the model fixed, so it says
nothing about training stochasticity, which is the larger term per fold (~0.07
on 1_150, against ~0.045 here). Only repeat draws measure that.

    python tools/eval_sampling_sd.py models/cv_baseline
    python tools/eval_sampling_sd.py models/cv_baseline --other models/exp  # paired

Nothing here loads a model, but it does import sx (for the event blocks and the
tier policy), which reaches train_utils and so TensorFlow -- hence the load-order
import below. It is still seconds, not minutes.
"""
# TensorFlow must be imported before pandas/pyarrow. pandas eagerly imports
# pyarrow, and pyarrow + TF each bundle their own statically-linked abseil; the
# shared lib that loads first wins abseil's weak synchronization symbols
# process-wide. If libarrow wins, TF's in-graph FFT (ducc0) threadpool ends up
# waiting on libarrow's incompatible semaphore impl and deadlocks. Keep this
# import first, ahead of anything that pulls in pandas.
import tensorflow  # noqa: F401  -- imported for load-order side effect only

import argparse
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)                          # config.py
sys.path.insert(0, os.path.join(_ROOT, '03_train'))  # sx.py, train_utils.py

from metrics import sens_at_fpr as _sens_at_fpr  # noqa: E402
from sx import buzz_event_blocks, TIERS_EXCLUDED_FROM_HEADLINE  # noqa: E402

BLOCK_NEG = 30  # negatives resample in blocks too; adjacent frames share their audio


def read_fold_predictions(dir_folds):
    """Each fold's predictions.csv, keyed by fold path."""
    import glob
    out = {}
    for path in glob.glob(os.path.join(dir_folds, '**', 'predictions.csv'), recursive=True):
        fold = os.path.relpath(os.path.dirname(path), dir_folds)
        out[fold] = pd.read_csv(path)
    return out


def fold_units(df):
    """(activation, positive event blocks, negative blocks, FP budget) for one fold.

    Positives are the scored ones -- `_quiet`-only buzz leaves the headline
    equation, so it is not part of the n either. Negatives are blocked by row
    run rather than by event: they carry no event structure, but adjacent
    negative frames are still the same audio, and the threshold is an order
    statistic over the top ~0.5% of them.
    """
    correct = df['correct'].astype(bool).to_numpy()
    tier = (df['loudness'].fillna('').astype(str).to_numpy()
            if 'loudness' in df.columns else np.full(len(df), ''))
    scored = correct & ~np.isin(tier, list(TIERS_EXCLUDED_FROM_HEADLINE))

    pos = [b[scored[b]] for b in buzz_event_blocks(df)]
    pos = [b for b in pos if len(b)]
    neg_idx = np.flatnonzero(~correct)
    neg = [neg_idx[i:i + BLOCK_NEG] for i in range(0, len(neg_idx), BLOCK_NEG)]
    return df['activation_ins_buzz'].to_numpy(), pos, neg, len(neg_idx)


def sens(activation, pos_idx, neg_idx, fpr):
    """The headline read over one resampled fold.

    Delegates to metrics.sens_at_fpr rather than taking the k-th negative
    directly: that function interpolates between bracketing thresholds and
    handles an unreachable target as NaN, and it is what folds_sx.csv is built
    from (tools/check_sens_at_fpr.py asserts the agreement). Rolling our own
    order statistic here read a couple of frames low on some folds, which is
    small but is exactly the sort of quiet disagreement between two tools that
    costs an afternoon later.

    Restricting the positives to the scored ones is sx._sens_over's move: the
    negatives are untouched, so the threshold is the same one the headline
    uses.
    """
    if not len(pos_idx) or not len(neg_idx):
        return np.nan
    act = np.concatenate([activation[pos_idx], activation[neg_idx]])
    correct = np.concatenate([np.ones(len(pos_idx), bool), np.zeros(len(neg_idx), bool)])
    return _sens_at_fpr(act, correct, (fpr,))[fpr]


def headline_sd(sds):
    """Folds are averaged unweighted, so their sampling SDs add in quadrature."""
    return float(np.sqrt(np.nansum(np.square(sds))) / len(sds))


def fold_rows(model_dir, other=None, fpr=0.005, n_boot=2000, seed=0):
    """One dict per fold: fold, events, sens, sd, neg_at_fpr, and delta/delta_sd
    when `other` is given and has the fold. tools/results.py reads these."""
    rng = np.random.default_rng(seed)
    preds = read_fold_predictions(os.path.join(model_dir, 'folds'))
    if not preds:
        raise FileNotFoundError(f'no predictions.csv under {model_dir}/folds')
    preds_b = read_fold_predictions(os.path.join(other, 'folds')) if other else {}

    rows = []
    for fold, df in sorted(preds.items()):
        activation, pos, neg, n_neg = fold_units(df)
        act_b = preds_b[fold]['activation_ins_buzz'].to_numpy() if fold in preds_b else None
        if act_b is not None and len(act_b) != len(df):
            # predictions.csv carries no frame key, so pairing is by row; two
            # frame grids (e.g. an AVES-hop embedder against YAMNet) can't pair,
            # and a longer `other` would otherwise pair the wrong frames silently
            print(f'WARNING: {fold}: {len(df)} vs {len(act_b)} frames -- different '
                  f'frame grids, no paired delta SD', file=sys.stderr)
            act_b = None

        boot, dboot = [], []
        for _ in range(n_boot):
            p = np.concatenate([pos[i] for i in rng.integers(0, len(pos), len(pos))]) if pos else np.array([], int)
            n = np.concatenate([neg[i] for i in rng.integers(0, len(neg), len(neg))])
            boot.append(sens(activation, p, n, fpr))
            if act_b is not None:
                # the same resampled evidence scored by both models: the shared
                # part of the wobble cancels, which is why a paired delta is
                # tighter than either model's own SD
                dboot.append(sens(act_b, p, n, fpr) - sens(activation, p, n, fpr))

        s = sens(activation, np.concatenate(pos) if pos else np.array([], int),
                 np.flatnonzero(~df['correct'].astype(bool).to_numpy()), fpr)
        row = dict(fold=fold, events=len(pos), sens=s, sd=float(np.nanstd(boot)),
                   neg_at_fpr=int(np.floor(fpr * n_neg)))
        if act_b is not None:
            s_b = sens(act_b, np.concatenate(pos) if pos else np.array([], int),
                       np.flatnonzero(~df['correct'].astype(bool).to_numpy()), fpr)
            row.update(delta=s_b - s, delta_sd=float(np.nanstd(dboot)))
        rows.append(row)
    return rows


def main(model_dir, other=None, fpr=0.005, n_boot=2000, seed=0):
    try:
        rows = fold_rows(model_dir, other, fpr, n_boot, seed)
    except FileNotFoundError as e:
        sys.exit(str(e))

    head = f"{'fold':<56}{'events':>7}{'sens':>7}{'boot_sd':>9}{'neg@fpr':>9}"
    if other:
        head += f"{'delta':>8}{'delta_sd':>10}"
    print(head)
    for r in rows:
        label = r['fold'] if len(r['fold']) <= 55 else '...' + r['fold'][-52:]
        line = f"{label:<56}{r['events']:>7}{r['sens']:>7.3f}{r['sd']:>9.3f}{r['neg_at_fpr']:>9}"
        if 'delta' in r:
            line += f"{r['delta']:>+8.3f}{r['delta_sd']:>10.3f}"
        print(line)

    paired = [r for r in rows if 'delta' in r]
    print(f"\n{'mean sens (the headline)':<56}{np.nanmean([r['sens'] for r in rows]):>7.3f}")
    print(f"{'headline SD, eval sampling alone':<56}{headline_sd([r['sd'] for r in rows]):>7.3f}")
    if paired:
        print(f"{'headline delta':<56}{np.nanmean([r['delta'] for r in paired]):>+7.3f}")
        print(f"{'headline delta SD, eval sampling alone':<56}{headline_sd([r['delta_sd'] for r in paired]):>7.3f}")
    print('\ntraining stochasticity is NOT in these numbers and is the larger term '
          'per fold; only repeat draws measure it.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model_dir', nargs='?', default='models/cv_baseline')
    ap.add_argument('--other', help='second model dir; report the paired per-fold delta SD too')
    ap.add_argument('--fpr', type=float, default=0.005)
    ap.add_argument('--n-boot', type=int, default=2000)
    args = ap.parse_args()
    main(args.model_dir, args.other, args.fpr, args.n_boot)
