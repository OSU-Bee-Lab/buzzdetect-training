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
# extraction freezes mid-run. The import therefore has to happen before
# load_stage() pulls in any stage module (all of which import pandas) — but it
# does NOT need to be at module scope, so it lives at the top of main() instead.
# That keeps `--help` and argparse errors instant (a cold `import tensorflow` is
# tens of seconds) without weakening the ordering guarantee.

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

# Same fallback 03_train/train.py::_RUN_CONFIG_DEFAULTS uses when --set/
# --embedder/--translation are left as None (inherit from an existing
# config_model.json). Stage 2 needs a concrete set/embedder up front, before
# train_set ever gets a chance to resolve against disk, so this is the value
# it falls back to for a brand-new model. Keep in sync with train.py.
_EXTRACT_DEFAULTS = {'setname': 'medium', 'embeddername': 'yamnet'}


def main(modelname, setname, embeddername, name_translation, epochs, clear,
         aug_dirnames=None, verbose=False, n_workers=2, snip_workers=2,
         overlap_event_prop=None, framehop_prop=None, assume_yes=False,
         stop_tol=0.01, skip_cv=False, surprisal=True, dropout=0.0,
         train_shipped=False, only_folds=None):
    import tensorflow  # noqa: F401  -- load-order side effect; see header comment

    # tensorflow-metal (Apple GPU) produces non-finite training loss within a
    # few epochs on this data; CPU does not, and is ~GPU speed for the 1024-d
    # probe. CUDA_VISIBLE_DEVICES does not touch the Metal pluggable device, so
    # hide the GPU explicitly. Opt-in via BUZZDETECT_NO_GPU=1.
    if os.environ.get('BUZZDETECT_NO_GPU'):
        tensorflow.config.set_visible_devices([], 'GPU')

    model_dir = os.path.join(config.DIR_MODELS, modelname)

    if clear and os.path.exists(model_dir):
        print(f'clearing {model_dir}')
        shutil.rmtree(model_dir)

    print('=== 02 extract set ===')
    multiprocessing.set_start_method('fork', force=True)
    stage2 = load_stage('02_set/extract.py', 'stage2_extract')
    stage2.extract_set(
        setname=setname if setname is not None else _EXTRACT_DEFAULTS['setname'],
        embeddername=embeddername if embeddername is not None else _EXTRACT_DEFAULTS['embeddername'],
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
        epochs=epochs,
        aug_dirnames=aug_dirnames,
        verbose=verbose,
        assume_yes=assume_yes,
        stop_tol=stop_tol,
        skip_cv=skip_cv,
        surprisal=surprisal,
        dropout=dropout,
        # --skip-cv means 'shipped model only', so it has to turn it on.
        train_shipped=train_shipped or skip_cv,
        only_folds=only_folds,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    # Flag names and defaults match the stage scripts (03_train/main.py in
    # particular), except the model name is --model here vs --name there.
    # --set/--embedder/--translation default to None, not a hardcoded value:
    # None means "inherit from this model's existing config_model.json if one
    # exists" (see train.py::_resolve_run_config) — a resumed run must be able
    # to omit them, so they can't be pinned to a default here.
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--set', dest='setname', default=None,
                        help="training set (default: inherited from --model's "
                             "existing config_model.json, else 'medium')")
    parser.add_argument('--embedder', default=None,
                        help="embedder (default: inherited from --model's "
                             "existing config_model.json, else 'yamnet')")
    parser.add_argument('--translation', default=None,
                        help="translation table (default: inherited from "
                             "--model's existing config_model.json, else "
                             "'general')")
    parser.add_argument('--epochs', type=int, default=400,
                        help='epoch budget: every rotation trains this many '
                             'epochs, fixed, no early stopping. See '
                             '03_train/CLAUDE.md.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='input dropout rate before the class logits '
                             '(default 0.0 = none)')
    parser.add_argument('--stop-tol', type=float, default=0.01, dest='stop_tol',
                        help='shipped-model epoch count: fraction of the consensus '
                             'val_loss curve span to stop short of its floor '
                             '(default 0.01). Larger = fewer epochs.')
    parser.add_argument('--skip-cv', action='store_true', dest='skip_cv',
                        help='train no rotations; go straight to the shipped model '
                             'using the fold results already on disk. Implies '
                             '--train-shipped.')
    parser.add_argument('--train-shipped', action='store_true', dest='train_shipped',
                        help='also train the shipped model after the rotations '
                             '(off by default)')
    parser.add_argument('--only-folds', nargs='+', default=None, dest='only_folds',
                        metavar='FOLD',
                        help='diagnostic: hold out & score only these rotating '
                             'folds; cannot supply a shipped epoch count')
    parser.add_argument('--workers', type=int, default=2, dest='n_workers',
                        help='framing+embedding workers; 0 runs in-process')
    parser.add_argument('--snip-workers', type=int, default=2, dest='snip_workers',
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
    parser.add_argument('--cpu', action='store_true',
                        help='hide the GPU from stage 2 extraction and stage 3 training '
                             '(sets BUZZDETECT_NO_GPU=1)')
    parser.add_argument('--no-surprisal', action='store_false', dest='surprisal',
                        help='skip the per-frame surprisal CSVs under '
                             '<model>/surprisal/ (written by default)')
    args = parser.parse_args()

    if args.cpu:
        os.environ['BUZZDETECT_NO_GPU'] = '1'

    main(
        modelname=args.model,
        setname=args.setname,
        embeddername=args.embedder,
        name_translation=args.translation,
        epochs=args.epochs,
        clear=args.clear,
        aug_dirnames=args.aug_dirnames,
        verbose=args.verbose,
        n_workers=args.n_workers,
        snip_workers=args.snip_workers,
        stop_tol=args.stop_tol,
        skip_cv=args.skip_cv,
        surprisal=args.surprisal,
        dropout=args.dropout,
        train_shipped=args.train_shipped,
        only_folds=args.only_folds,
        overlap_event_prop=args.overlap_event_prop,
        framehop_prop=args.framehop_prop,
        assume_yes=args.assume_yes,
    )
