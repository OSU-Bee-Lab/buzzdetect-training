"""
Run the training pipeline (stages 2–3) for a model: extract the set's
embeddings, then train leave-one-fold-out CV plus the shipped model.

There is no stage 4. It scored a fixed model against a hand-curated corpus and
expected the `<name>_v1…_vN` repeated-run layout, both of which the CV rework
replaced; a CV run scores every held-out fold and writes folds_sx.csv itself.
"""
# TensorFlow must be imported before pandas/pyarrow. pandas eagerly imports
# pyarrow, and pyarrow + TF each bundle their own statically-linked abseil; the
# shared lib that loads first wins abseil's weak synchronization symbols
# process-wide. If libarrow wins, TF's in-graph FFT (ducc0) threadpool ends up
# waiting on libarrow's incompatible semaphore impl and deadlocks — YAMNet
# extraction freezes mid-run. Keep this import first, ahead of the stage
# modules (which pull in pandas).
import tensorflow  # noqa: F401  -- imported for load-order side effect only

import argparse
import importlib.util
import multiprocessing
import os
import shutil
import sys

import config


def load_stage(path, module_name):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    stage_dir = os.path.dirname(os.path.abspath(path))
    for d in (root_dir, stage_dir):
        if d not in sys.path:
            sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main(modelname, setname, embeddername, name_translation, epochs_max, clear,
         aug_dirnames=None, verbose=False, n_workers=2, snip_workers=4, patience=50,
         overlap_event_prop=None, framehop_prop=None, assume_yes=False,
         stop_tol=0.01, skip_cv=False, surprisal=True):
    model_dir = os.path.join(config.DIR_MODELS, modelname)

    if clear and os.path.exists(model_dir):
        print(f'clearing {model_dir}')
        shutil.rmtree(model_dir)

    print('=== 02 extract set ===')
    multiprocessing.set_start_method('fork', force=True)
    stage2 = load_stage('02_set/main.py', 'stage2_main')
    stage2.extract_set(
        setname=setname,
        embeddername=embeddername,
        overlap_event_prop=overlap_event_prop,
        framehop_prop=framehop_prop,
        n_workers=n_workers,
        snip_workers=snip_workers,
        verbose=verbose,
    )

    print('\n=== 03 train ===')
    stage3 = load_stage('03_train/train.py', 'stage3_train')
    stage3.train_set(
        name=modelname,
        embeddername=embeddername,
        setname=setname,
        name_translation=name_translation,
        epochs_max=epochs_max,
        aug_dirnames=aug_dirnames,
        verbose=verbose,
        patience=patience,
        assume_yes=assume_yes,
        stop_tol=stop_tol,
        skip_cv=skip_cv,
        surprisal=surprisal,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults match 03_train/main.py — only --model is required.
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--set', dest='setname', default='medium')
    parser.add_argument('--embedder', default='yamnet')
    parser.add_argument('--translation', default='general')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--patience', type=int, default=50,
                        help='EarlyStopping patience for the per-fold submodels')
    parser.add_argument('--stop-tol', type=float, default=0.01, dest='stop_tol',
                        help='shipped-model epoch count: fraction of the consensus '
                             'val_loss curve span to stop short of its floor '
                             '(default 0.01). Larger = fewer epochs.')
    parser.add_argument('--skip-cv', action='store_true', dest='skip_cv',
                        help='train no rotations; go straight to the shipped model '
                             'using the fold results already on disk')
    parser.add_argument('--workers', type=int, default=2, dest='n_workers',
                        help='framing+embedding workers; 0 runs in-process')
    parser.add_argument('--snip-workers', type=int, default=4, dest='snip_workers',
                        help='threads for the snip-sync phase (source-drive I/O); 1 = serial')
    # Only consulted when the set has no config_extract.json yet; see 02_set/main.py.
    parser.add_argument('--overlap-event-prop', type=float, default=None, dest='overlap_event_prop')
    parser.add_argument('--framehop-prop', type=float, default=None, dest='framehop_prop')
    # Opt-in rather than opt-out: 03_train skips folds that already have a
    # config_model.json, so a rerun resumes by default. Clearing throws that
    # away, which should take an explicit flag rather than being the default.
    parser.add_argument('--clear', action='store_true',
                        help='Delete the existing model dir before training')
    parser.add_argument('--augment', nargs='*', dest='aug_dirnames', metavar='AUG_DIRNAME',
                        help='Augmented embedding dirs to include in training (must be built via 02_set/augment.py first)')
    parser.add_argument('-y', '--yes', action='store_true', dest='assume_yes',
                        help='accept untranslated labels without confirming')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-surprisal', action='store_false', dest='surprisal',
                        help='skip the per-frame surprisal CSVs under '
                             '<model>/surprisal/ (written by default)')
    args = parser.parse_args()

    main(
        modelname=args.model,
        setname=args.setname,
        embeddername=args.embedder,
        name_translation=args.translation,
        epochs_max=args.epochs,
        clear=args.clear,
        aug_dirnames=args.aug_dirnames,
        verbose=args.verbose,
        n_workers=args.n_workers,
        snip_workers=args.snip_workers,
        patience=args.patience,
        stop_tol=args.stop_tol,
        skip_cv=args.skip_cv,
        surprisal=args.surprisal,
        overlap_event_prop=args.overlap_event_prop,
        framehop_prop=args.framehop_prop,
        assume_yes=args.assume_yes,
    )
