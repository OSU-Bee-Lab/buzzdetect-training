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
    parser.add_argument('--stop-tol', type=float, default=0.01, dest='stop_tol',
                        help='shipped-model epoch count: fraction of the consensus '
                             'val_loss curve span to stop short of its floor '
                             '(default 0.01). Larger = fewer epochs. See '
                             'train._consensus_epoch.')
    parser.add_argument('--skip-cv', action='store_true', dest='skip_cv',
                        help='train no rotations; go straight to the shipped model, '
                             'taking its epoch count from the fold results already '
                             'on disk. Errors if there are none.')
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
        stop_tol=args.stop_tol,
        skip_cv=args.skip_cv,
    )
