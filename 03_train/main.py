import argparse
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--set', default=None, dest='setname',
                        help="training set (default: inherited from --name's "
                             "existing config_model.json, else 'medium')")
    parser.add_argument('--embedder', default=None,
                        help="embedder (default: inherited from --name's "
                             "existing config_model.json, else 'yamnet')")
    parser.add_argument('--translation', default=None,
                        help="translation table (default: inherited from "
                             "--name's existing config_model.json, else "
                             "'general')")
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='input dropout rate before the class logits '
                             '(default 0.0 = none). The era baseline is the '
                             'bare linear probe; dropout was 0.2 and hardcoded '
                             'through the 2026-09 era, and is now something an '
                             'experiment turns on rather than a premise.')
    parser.add_argument('--epochs', type=int, default=400,
                        help='THE EPOCH BUDGET (default 400). Train every '
                             'rotation for exactly this many epochs, with no '
                             'early stopping and no restore-best, and ship the '
                             'final weights. No per-fold epoch selection of '
                             'any kind happens, so every arm of a comparison '
                             'is scored at one identical epoch. Vary it to run '
                             'a budget ladder; 400 is provisional — see '
                             '03_train/CLAUDE.md. The shipped model (see '
                             '--train-shipped) does not use this budget '
                             'directly: it reads a data-driven epoch count off '
                             'the rotation val_loss curves instead.')
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
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-surprisal', action='store_false', dest='surprisal',
                        help='skip the per-frame surprisal CSVs under '
                             '<model>/surprisal/ (written by default, one file '
                             'per source ident, for finding bad annotations)')
    parser.add_argument('--cpu', action='store_true',
                        help='hide the GPU from training (sets BUZZDETECT_NO_GPU=1). '
                             'tensorflow-metal (Apple GPU) produces non-finite training '
                             'loss within a few epochs on this data; CPU does not, and is '
                             '~GPU speed for the 1024-d probe. CUDA_VISIBLE_DEVICES does '
                             'not touch the Metal pluggable device, so this is the only '
                             'way to hide it.')
    args = parser.parse_args()

    # TensorFlow must be imported before pandas/pyarrow. pandas eagerly imports
    # pyarrow, and pyarrow + TF each bundle their own statically-linked abseil; the
    # shared lib that loads first wins abseil's weak synchronization symbols
    # process-wide. If libarrow wins, TF's in-graph FFT (ducc0) threadpool ends up
    # waiting on libarrow's incompatible semaphore impl and deadlocks. Keep this
    # import first, ahead of anything (incl. train.py) that pulls in pandas.
    # Deferred to here, after parse_args(), so `--help` and argparse errors are
    # instant rather than paying TF's tens-of-seconds cold import.
    import tensorflow  # noqa: F401  -- imported for load-order side effect only

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from train import train_set

    # tensorflow-metal (Apple GPU) produces non-finite training loss within a
    # few epochs on this data; CPU does not, and is ~GPU speed for the 1024-d
    # probe. CUDA_VISIBLE_DEVICES does not touch the Metal pluggable device,
    # so hide the GPU explicitly. Opt-in via --cpu or BUZZDETECT_NO_GPU=1.
    if args.cpu or os.environ.get('BUZZDETECT_NO_GPU'):
        os.environ['BUZZDETECT_NO_GPU'] = '1'
        tensorflow.config.set_visible_devices([], 'GPU')

    train_set(
        name=args.name,
        embeddername=args.embedder,
        setname=args.setname,
        name_translation=args.translation,
        epochs=args.epochs,
        aug_dirnames=args.aug_dirnames,
        verbose=args.verbose,
        assume_yes=args.assume_yes,
        stop_tol=args.stop_tol,
        skip_cv=args.skip_cv,
        dropout=args.dropout,
        # --skip-cv means 'shipped model only', so it has to turn it on.
        train_shipped=args.train_shipped or args.skip_cv,
        only_folds=args.only_folds,
        surprisal=args.surprisal,
    )
