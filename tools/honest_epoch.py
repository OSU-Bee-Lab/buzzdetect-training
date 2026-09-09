"""Re-score every run's folds at an epoch chosen without seeing that fold.

Each rotation's held-out fold doubles as its early-stopping monitor
(README.md, "There is no `validate` role"), so a fold's reported sensitivity was
measured on the fold that picked its epoch. Under `--monitor val_loss` the
selection statistic differs from the reported one and README:642 argues the
optimism is small. Under `--monitor val_sens` they are the *same statistic*, and
the reported number is `max` over epochs of the thing being reported.

README:651 prescribes the remedy and notes it needs no retraining: score each
fold at an epoch derived from the *other* folds. `val_sens_fpr0.005_curve` is
persisted per fold, so that is a lookup.

Rules, all scoring fold f off f's own curve but choosing the epoch differently:

  own-peak     max of f's own curve                      (what a val_sens run reports)
  shipped      f's own best_epoch                        (what the run reported)
  xfold-median median of the other folds' peak epochs     (README:651, literally)
  xfold-pooled argmax of the MEAN of the other folds' curves — one shared epoch,
               which is how the shipped model's epoch is actually chosen, and the
               only rule an operator with no labels for the new site could use.

No TensorFlow, no training. Reads models/*/folds/**/summary.json.
"""
import json
import os
import sys
import glob
import statistics


def load_run(model_dir):
    folds = {}
    pattern = os.path.join(model_dir, 'folds', '**', 'summary.json')
    for path in glob.glob(pattern, recursive=True):
        summary = json.load(open(path))
        curve = summary.get('val_sens_fpr0.005_curve')
        if not curve:
            continue
        rel = os.path.relpath(os.path.dirname(path), os.path.join(model_dir, 'folds'))
        folds[rel.split('/')[-1]] = {
            'curve': curve,
            'best_epoch': summary['best_epoch'],
            'at_best': summary['val_sens_fpr0.005_at_best'],
            'peak_epoch': summary['val_sens_fpr0.005_peak_epoch'],
        }
    return folds


def at(curve, epoch):
    """Sensitivity at a 1-indexed epoch, clamped to the curve's length.

    A curve ends at n_epochs, which is the peak plus the patience tail, so a
    clamp lands on a real measurement rather than an extrapolation.
    """
    return curve[min(epoch, len(curve)) - 1]


def score(folds):
    rows = []
    for name, f in sorted(folds.items()):
        others = [o for k, o in folds.items() if k != name]
        # The other folds' curves end where their own early stopping ended, so
        # the latest testable shared epoch is capped by the shortest of them.
        # A rule that lands ON the cap is marked: its epoch may be too early.
        span = min(len(o['curve']) for o in others)
        pooled = [
            sum(at(o['curve'], e) for o in others) / len(others)
            for e in range(1, span + 1)
        ]
        e_pooled = max(range(1, span + 1), key=lambda e: pooled[e - 1])
        e_median = int(statistics.median(o['peak_epoch'] for o in others))
        rows.append({
            'fold': name,
            'own-peak': max(f['curve']),
            'shipped': f['at_best'],
            'xfold-median': at(f['curve'], e_median),
            'xfold-pooled': at(f['curve'], e_pooled),
            'e_shipped': f['best_epoch'],
            'e_median': e_median,
            'e_pooled': e_pooled,
            'span': span,
            'n_epochs': len(f['curve']),
        })
    return rows


RULES = ('own-peak', 'shipped', 'xfold-median', 'xfold-pooled')


def main(argv):
    roots = argv or ['models'] + glob.glob('.local/worktrees/*/models')
    seen = {}
    for root in roots:
        for csv in sorted(glob.glob(os.path.join(root, '*', 'folds_sx.csv'))):
            model_dir = os.path.dirname(csv)
            seen.setdefault(os.path.basename(model_dir), model_dir)

    print(f'{"run":24} ' + ' '.join(f'{r:>13}' for r in RULES))
    summary = {}
    for name, model_dir in sorted(seen.items()):
        folds = load_run(model_dir)
        if len(folds) < 3:
            continue
        rows = score(folds)
        means = {r: sum(row[r] for row in rows) / len(rows) for r in RULES}
        summary[name] = (means, rows)
        print(f'{name:24} ' + ' '.join(f'{means[r]:13.3f}' for r in RULES))

    for name, (means, rows) in summary.items():
        print(f'\n== {name}')
        print(f'   {"fold":>6} ' + ' '.join(f'{r:>13}' for r in RULES)
              + '   epochs shipped/median/pooled of n  (span, * = capped)')
        for row in rows:
            print(f'   {row["fold"]:>6} '
                  + ' '.join(f'{row[r]:13.3f}' for r in RULES)
                  + f'   {row["e_shipped"]}/{row["e_median"]}/{row["e_pooled"]}'
                    f' of {row["n_epochs"]}'
                    f'  ({row["span"]}{"*" if row["e_pooled"] == row["span"] else ""})')


if __name__ == '__main__':
    main(sys.argv[1:])
