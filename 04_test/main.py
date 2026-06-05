# TensorFlow must be imported before pandas/pyarrow. pandas eagerly imports
# pyarrow, and pyarrow + TF each bundle their own statically-linked abseil; the
# shared lib that loads first wins abseil's weak synchronization symbols
# process-wide. If libarrow wins, TF's in-graph FFT (ducc0) threadpool ends up
# waiting on libarrow's incompatible semaphore impl and deadlocks. Keep this
# import first, ahead of pandas.
import tensorflow  # noqa: F401  -- imported for load-order side effect only

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import config as cfg
from inference import run_inference
from metrics import compute_model_metrics
from plots import plot_tradeoff, plot_tradeoff_zoomed, plot_metrics, plot_metrics_zoomed


def test_model(modelname):
    framelength = run_inference(modelname)

    path_metrics = os.path.join(cfg.DIR_MODELS, modelname, cfg.SUBDIR_TESTS, cfg.FNAME_METRICS)
    path_tradeoff = os.path.join(cfg.DIR_MODELS, modelname, cfg.SUBDIR_TESTS, cfg.FNAME_PLOT_TRADEOFF)
    path_metrics_plot = os.path.join(cfg.DIR_MODELS, modelname, cfg.SUBDIR_TESTS, cfg.FNAME_PLOT_METRICS)
    path_tradeoff_zoomed = os.path.join(cfg.DIR_MODELS, modelname, cfg.SUBDIR_TESTS, cfg.FNAME_PLOT_TRADEOFF_ZOOMED)
    path_metrics_zoomed = os.path.join(cfg.DIR_MODELS, modelname, cfg.SUBDIR_TESTS, cfg.FNAME_PLOT_METRICS_ZOOMED)

    if os.path.exists(path_metrics):
        print(f'  metrics exist, skipping; loading for plot')
        metrics = pd.read_csv(path_metrics)
    else:
        annotations = pd.read_csv(os.path.join(cfg.TEST_DIR, 'annotations.csv'))
        metrics = compute_model_metrics(modelname, annotations, framelength=framelength)
        metrics.to_csv(path_metrics, index=False)
        print(f'  saved {path_metrics}')

    plot_tradeoff(metrics, modelname)
    plot_tradeoff_zoomed(metrics, modelname)
    plot_metrics(metrics, modelname)
    plot_metrics_zoomed(metrics, modelname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()

    test_model(args.model)
