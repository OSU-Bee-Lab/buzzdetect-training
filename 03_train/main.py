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

# tensorflow-metal (Apple GPU) produces non-finite training loss within a few
# epochs on this data; CPU does not, and is ~GPU speed for the 1024-d probe.
# CUDA_VISIBLE_DEVICES does not touch the Metal pluggable device, so hide the
# GPU explicitly. Opt-in via BUZZDETECT_NO_GPU=1.
if os.environ.get('BUZZDETECT_NO_GPU'):
    tensorflow.config.set_visible_devices([], 'GPU')

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
                             'on disk. Errors if there are none. Implies '
                             '--train-shipped; the inverse of the default.')
    parser.add_argument('--only-folds', nargs='+', default=None, dest='only_folds',
                        metavar='FOLD',
                        help='diagnostic: hold out & score only these rotating '
                             'folds (still trains on the full pool). folds_sx.csv '
                             'is then a subset, not comparable to a full CV, and '
                             'cannot supply a shipped epoch count.')
    parser.add_argument('--train-shipped', action='store_true', dest='train_shipped',
                        help='also train the shipped model after the rotations. '
                             'Off by default: it contributes nothing to '
                             'folds_sx.csv, so an experiment never needs it. Its '
                             'epoch count is read from the fold curves on disk, so '
                             'a later run with the same --name plus --skip-cv '
                             'produces the same model. Implied by --skip-cv.')
    parser.add_argument('--augment', nargs='*', dest='aug_dirnames', metavar='AUG_DIRNAME')
    parser.add_argument('-y', '--yes', action='store_true', dest='assume_yes',
                        help='accept untranslated labels without confirming')
    parser.add_argument('--margin-lambda', type=float, default=0.0,
                        dest='margin_lambda',
                        help='weight of the class-conditional hinge that pushes '
                             'the ins_buzz logit at least --margin-m below zero '
                             'on frames labelled --margin-class and not '
                             'ins_buzz. 0 (default) leaves the loss identical to '
                             'every run logged before this flag existed.')
    parser.add_argument('--margin-m', type=float, default=2.0, dest='margin_m',
                        help='margin in logits for --margin-lambda (default 2.0)')
    parser.add_argument('--margin-class', default='mech_auto',
                        dest='margin_class',
                        help='conditioning class for --margin-lambda, named in '
                             'the translation\'s output space (default mech_auto, '
                             'the plurality of cv_baseline\'s threshold-setting '
                             'false positives)')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-surprisal', action='store_false', dest='surprisal',
                        help='skip the per-frame surprisal CSVs under '
                             '<model>/surprisal/ (written by default, one file '
                             'per source ident, for finding bad annotations)')
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
        # --skip-cv means 'shipped model only', so it has to turn it on.
        train_shipped=args.train_shipped or args.skip_cv,
        only_folds=args.only_folds,
        surprisal=args.surprisal,
        margin_lambda=args.margin_lambda,
        margin_m=args.margin_m,
        margin_class=args.margin_class,
    )
