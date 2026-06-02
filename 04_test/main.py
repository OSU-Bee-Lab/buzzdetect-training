import os

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

    if os.path.exists(path_tradeoff):
        print(f'  tradeoff plot exists, skipping')
    else:
        plot_tradeoff(metrics, modelname)

    if os.path.exists(path_tradeoff_zoomed):
        print(f'  tradeoff zoomed plot exists, skipping')
    else:
        plot_tradeoff_zoomed(metrics, modelname)

    if os.path.exists(path_metrics_plot):
        print(f'  metrics plot exists, skipping')
    else:
        plot_metrics(metrics, modelname)

    if os.path.exists(path_metrics_zoomed):
        print(f'  metrics zoomed plot exists, skipping')
    else:
        plot_metrics_zoomed(metrics, modelname)


if __name__ == '__main__':
    test_model('test')
