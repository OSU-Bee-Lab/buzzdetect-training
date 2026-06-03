"""
Run the full training pipeline (stages 2–4) for a model.
Default: modelname='test', setname='lite', cleared before running.
"""
import argparse
import importlib.util
import multiprocessing
import os
import shutil
import sys

import config

# NOTE: these defaults are for a quick run through; they are not optimal for frontier models
TEST_MODEL = 'test'
TEST_DATASET = 'lite'
TEST_EMBEDDER = 'yamnet'
TEST_TRANSLATION = 'general'
TEST_EPOCHS = 50 # this value, in particular, needs to be cranked up. Early stopping rules mean that there can't be too many epochs; set to at least 300


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


modelname = TEST_MODEL
setname = TEST_DATASET
embeddername = TEST_EMBEDDER
name_translation = TEST_TRANSLATION
epochs_max = TEST_EPOCHS
clear=True

def main(modelname, setname, embeddername, name_translation, epochs_max, clear):
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
    )

    print('\n=== 03 train ===')
    stage3 = load_stage('03_train/main.py', 'stage3_main')
    stage3.train_model(
        modelname=modelname,
        embeddername=embeddername,
        setname=setname,
        name_translation=name_translation,
        epochs_max=epochs_max,
    )

    print('\n=== 04 test ===')
    stage4 = load_stage('04_test/main.py', 'stage4_main')
    stage4.test_model(modelname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=TEST_MODEL)
    parser.add_argument('--set', default=TEST_DATASET, dest='setname')
    parser.add_argument('--embedder', default=TEST_EMBEDDER)
    parser.add_argument('--translation', default=TEST_TRANSLATION)
    parser.add_argument('--epochs', type=int, default=TEST_EPOCHS)
    parser.add_argument('--no-clear', action='store_false', dest='clear',
                        help='Skip clearing existing model dir')
    args = parser.parse_args()

    main(
        modelname=args.model,
        setname=args.setname,
        embeddername=args.embedder,
        name_translation=args.translation,
        epochs_max=args.epochs,
        clear=args.clear,
    )
