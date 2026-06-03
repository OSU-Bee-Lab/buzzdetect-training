"""
Run the full training pipeline (stages 2–4) for a model.
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

def main(modelname, setname, embeddername, name_translation, epochs_max, clear, aug_dirnames=None, verbose=False):
    model_dir = os.path.join(config.DIR_MODELS, modelname)

    if clear and os.path.exists(model_dir):
        print(f'Clearing {model_dir}')
        shutil.rmtree(model_dir)

    print('\n=== 02 extract set ===')
    multiprocessing.set_start_method('fork', force=True)
    stage2 = load_stage('02_set/main.py', 'stage2_main')
    stage2.extract_set(
        setname=setname,
        embeddername=embeddername,
        n_workers=4,
        verbose=verbose,
    )

    print('\n=== 03 train ===')
    stage3 = load_stage('03_train/main.py', 'stage3_main')
    stage3.train_model(
        modelname=modelname,
        embeddername=embeddername,
        setname=setname,
        name_translation=name_translation,
        epochs_max=epochs_max,
        aug_dirnames=aug_dirnames,
    )

    print('\n=== 04 test ===')
    stage4 = load_stage('04_test/main.py', 'stage4_main')
    stage4.test_model(modelname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--set', required=True, dest='setname')
    parser.add_argument('--embedder', required=True)
    parser.add_argument('--translation', required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--no-clear', action='store_false', dest='clear',
                        help='Skip clearing existing model dir')
    parser.add_argument('--augment', nargs='*', dest='aug_dirnames', metavar='AUG_DIRNAME',
                        help='Augmented embedding dirs to include in training (must be built via 02_set/augment.py first)')
    parser.add_argument('--verbose', action='store_true')
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
    )
