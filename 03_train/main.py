# TensorFlow must be imported before pandas/pyarrow. pandas eagerly imports
# pyarrow, and pyarrow + TF each bundle their own statically-linked abseil; the
# shared lib that loads first wins abseil's weak synchronization symbols
# process-wide. If libarrow wins, TF's in-graph FFT (ducc0) threadpool ends up
# waiting on libarrow's incompatible semaphore impl and deadlocks. Keep this
# import first, ahead of anything (incl. train.py) that pulls in pandas.
import tensorflow  # noqa: F401  -- imported for load-order side effect only

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train_set

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--set', default='medium', dest='setname')
    parser.add_argument('--embedder', default='yamnet')
    parser.add_argument('--translation', default='general')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--patience', type=int, default=50,
                        help='EarlyStopping patience for the per-fold submodels')
    parser.add_argument('--learning-rate', type=float, default=0.002, dest='learning_rate',
                        help='Adam learning rate (std-convergence sweep knob)')
    parser.add_argument('--min-delta', type=float, default=0.002, dest='min_delta',
                        help='EarlyStopping min_delta (std-convergence sweep knob)')
    parser.add_argument('--clipnorm', type=float, default=None,
                        help='Adam clipnorm; optimizer-level safeguard, not a structural '
                             'change (std-convergence sweep knob)')
    parser.add_argument('--only-folds', nargs='*', dest='only_folds', metavar='FOLD',
                        help='restrict which rotate folds take a turn as held-out '
                             '(cheap diagnosis only; does not touch folds.csv)')
    parser.add_argument('--skip-shipped', action='store_true', dest='skip_shipped',
                        help='skip training the final shipped model (cheap diagnosis only)')
    parser.add_argument('--augment', nargs='*', dest='aug_dirnames', metavar='AUG_DIRNAME')
    parser.add_argument('-y', '--yes', action='store_true', dest='assume_yes',
                        help='accept untranslated labels without confirming')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    train_set(
        name=args.name,
        embeddername=args.embedder,
        setname=args.setname,
        name_translation=args.translation,
        epochs_max=args.epochs,
        aug_dirnames=args.aug_dirnames,
        verbose=args.verbose,
        patience=args.patience,
        assume_yes=args.assume_yes,
        learning_rate=args.learning_rate,
        min_delta=args.min_delta,
        only_folds=args.only_folds,
        skip_shipped=args.skip_shipped,
    )
