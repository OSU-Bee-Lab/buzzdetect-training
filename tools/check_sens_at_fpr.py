"""Assert metrics.sens_at_fpr agrees with metrics_by_group -> metrics_at_fpr.

sens_at_fpr is a fast numpy restatement of that pandas pair, run once per epoch
by 03_train/callbacks.py instead of once per fold. Two implementations of one
definition drift, so this pins them together on random and adversarial inputs
(heavy score ties, one-sided labels, targets outside the reachable FPR range).

    conda run -n buzzdetect-train python tools/check_sens_at_fpr.py

No TensorFlow, no repo data.
"""

import os
import sys

import numpy as np
import pandas as pd

# absolute, so this can't silently pick up main's metrics.py from a worktree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '03_train'))

from metrics import metrics_by_group, metrics_at_fpr, sens_at_fpr  # noqa: E402

TARGETS = (0.001, 0.005, 0.01, 0.05, 0.5)


def reference(activation, correct, fprs):
    df = pd.DataFrame({'activation_ins_buzz': activation, 'correct': correct})
    sweep = metrics_at_fpr(metrics_by_group(df), fprs)
    return dict(zip(sweep['fpr'], sweep['sensitivity']))


def check(name, activation, correct, tol=1e-3):
    """tol is 1e-3 because metrics_at_fpr rounds its output to 3 decimals."""
    got = sens_at_fpr(activation, correct, TARGETS)
    want = reference(activation, correct, TARGETS)
    for f in TARGETS:
        a, b = got[f], want[f]
        if np.isnan(a) and np.isnan(b):
            continue
        assert not (np.isnan(a) or np.isnan(b)), f'{name}: fpr={f} nan mismatch {a} vs {b}'
        assert abs(a - b) <= tol, f'{name}: fpr={f} {a} vs {b}'
    print(f'  ok  {name}: ' + ', '.join(f'{f:g}->{got[f]:.3f}' for f in TARGETS))


def main():
    rng = np.random.default_rng(0)

    for n, prevalence in ((20000, 0.02), (5000, 0.1), (300, 0.5)):
        correct = rng.random(n) < prevalence
        # separable-ish logits, the shape a trained probe actually produces
        activation = rng.normal(np.where(correct, 1.5, -1.5), 1.0)
        check(f'random n={n} prev={prevalence}', activation, correct)

    # heavy ties: many frames share one score, so distinct-threshold grouping
    # and the choice of bracketing rows are what the two paths could disagree on
    correct = rng.random(4000) < 0.05
    check('coarse ties', np.round(rng.normal(0, 1, 4000), 1), correct)
    check('near-constant', np.where(rng.random(4000) < 0.5, 0.0, 1.0), correct)

    # degenerate label sets -- both paths must return NaN, not divide by zero
    check('all positive', rng.normal(0, 1, 500), np.ones(500, dtype=bool))
    check('all negative', rng.normal(0, 1, 500), np.zeros(500, dtype=bool))

    # too few negatives for fpr=0.001 to correspond to even one frame
    correct = np.array([True] * 50 + [False] * 50)
    check('tiny negatives', rng.normal(0, 1, 100), correct)

    print('\nsens_at_fpr matches metrics_at_fpr on all cases.')


if __name__ == '__main__':
    main()
