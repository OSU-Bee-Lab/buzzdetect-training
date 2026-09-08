"""Bootstrap SD of sens@fpr on a trained model's own eval frames.

The headline metric places each fold's threshold on ~0.5% of that fold's
negative frames -- between 1 and 30 frames, depending on the fold. This asks how
much of the run-to-run noise floor is that sampling, rather than training
stochasticity: it resamples frames within each fold (with replacement) and
recomputes the metric. No training, no TensorFlow.

    python tools/eval_sampling_sd.py models/cv_baseline
"""
import csv
import glob
import os
import sys

import numpy as np


def read_fold_predictions(dir_folds):
    """(activation, is_buzz) per fold. `correct` in predictions.csv is the
    ground-truth buzz label, not whether the prediction was right."""
    out = {}
    for path in glob.glob(os.path.join(dir_folds, '**', 'predictions.csv'), recursive=True):
        activation, is_buzz = [], []
        with open(path) as f:
            rows = csv.reader(f)
            next(rows)
            for row in rows:
                activation.append(float(row[0]))
                is_buzz.append(row[1] == 'True')
        fold = os.path.relpath(os.path.dirname(path), dir_folds)
        out[fold] = (np.array(activation), np.array(is_buzz))
    return out


def sens_at_fpr(activation, is_buzz, fpr=0.005):
    negatives = np.sort(activation[~is_buzz])[::-1]
    k = int(np.floor(fpr * len(negatives)))
    if k < 1:
        return np.nan
    return float((activation[is_buzz] > negatives[k - 1]).mean())


def main(model_dir, fpr=0.005, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    preds = read_fold_predictions(os.path.join(model_dir, 'folds'))
    if not preds:
        sys.exit(f'no predictions.csv under {model_dir}/folds')

    print(f"{'fold':<58}{'sens':>7}{'boot_sd':>9}{'neg@fpr':>9}")
    point, spread = [], []
    for fold, (activation, is_buzz) in sorted(preds.items()):
        boot = [
            sens_at_fpr(activation[i], is_buzz[i], fpr)
            for i in (rng.integers(0, len(activation), len(activation)) for _ in range(n_boot))
        ]
        sens, sd = sens_at_fpr(activation, is_buzz, fpr), np.nanstd(boot)
        n_neg = int(np.floor(fpr * (~is_buzz).sum()))
        print(f'{fold:<58}{sens:>7.3f}{sd:>9.3f}{n_neg:>9}')
        point.append(sens)
        spread.append(sd)

    # folds are averaged unweighted, so their sampling SDs add in quadrature
    headline_sd = np.sqrt(np.nansum(np.square(spread))) / len(spread)
    print(f"\n{'mean sens':<58}{np.nanmean(point):>7.3f}")
    print(f"{'headline SD from eval sampling alone':<58}{headline_sd:>7.3f}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'models/cv_baseline')
