# TensorFlow must be imported before pandas/pyarrow. pandas eagerly imports
# pyarrow, and pyarrow + TF each bundle their own statically-linked abseil; the
# shared lib that loads first wins abseil's weak synchronization symbols
# process-wide. If libarrow wins, TF's in-graph FFT (ducc0) threadpool ends up
# waiting on libarrow's incompatible semaphore impl and deadlocks. Keep this
# import first, ahead of pandas.
import tensorflow  # noqa: F401  -- imported for load-order side effect only

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import config as cfg
from inference import ensure_test_embeddings, run_inference_for_model
from metrics import compute_model_metrics
from plots import plot_tradeoff, plot_tradeoff_zoomed, plot_metrics, plot_metrics_zoomed


def test_set(name, runs):
    # Determine embeddername from first model's config.
    first_model = f'{name}_v1'
    with open(os.path.join(cfg.DIR_MODELS, first_model, 'config_model.json')) as f:
        embeddername = json.load(f)['embeddername']

    embeddings_by_ident, framelength_s = ensure_test_embeddings(embeddername)

    annotations = pd.read_csv(os.path.join(cfg.TEST_DIR, 'annotations.csv'))

    for i in range(1, runs + 1):
        modelname = f'{name}_v{i}'
        run_inference_for_model(modelname, embeddings_by_ident, framelength_s)

        path_metrics = os.path.join(cfg.DIR_MODELS, modelname, cfg.SUBDIR_TESTS, cfg.FNAME_METRICS)
        if not os.path.exists(path_metrics):
            metrics = compute_model_metrics(modelname, annotations, framelength=framelength_s)
            metrics.to_csv(path_metrics, index=False)

        metrics = pd.read_csv(path_metrics)
        plot_tradeoff(metrics, modelname)
        plot_tradeoff_zoomed(metrics, modelname)
        plot_metrics(metrics, modelname)
        plot_metrics_zoomed(metrics, modelname)
        print(f'  [{modelname}] done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--runs', type=int, default=5)
    args = parser.parse_args()

    test_set(args.name, args.runs)
