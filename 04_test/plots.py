import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import config as cfg

COLORS = {
    'sensitivity':        '#00b9e3',
    'precision':          '#e88526',
    'false positive rate': '#c73030',
}

_pct = mticker.FuncFormatter(lambda x, _: f'{x:.0%}')
_raw = mticker.FuncFormatter(lambda x, _: f'{x:.2f}')


def plot_tradeoff(metrics, modelname):
    """Sensitivity vs precision (trade-off space)."""
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(metrics['precision'], metrics['sensitivity'],
            color=COLORS['sensitivity'], linewidth=1.5)
    ax.set_xlabel('Precision')
    ax.set_ylabel('Sensitivity')
    ax.xaxis.set_major_formatter(_pct)
    ax.yaxis.set_major_formatter(_pct)
    ax.grid(axis='y', alpha=0.4)

    fig.suptitle(modelname, fontsize=11)
    plt.tight_layout()

    path_out = os.path.join(cfg.DIR_MODELS, modelname, cfg.SUBDIR_TESTS, cfg.FNAME_PLOT_TRADEOFF)
    fig.savefig(path_out)
    plt.close(fig)
    print(f'  saved {path_out}')


def plot_metrics(metrics, modelname):
    """Sensitivity, precision, and FPR each in their own panel vs threshold."""
    metric_cols = [
        ('sensitivity',        'sensitivity'),
        ('precision',          'precision'),
        ('false positive rate', 'fpr'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    for (label, col), ax in zip(metric_cols, axes):
        ax.plot(metrics['threshold'], metrics[col],
                color=COLORS[label], linewidth=1.5)
        ax.set_xlabel('Threshold')
        ax.set_title(label)
        ax.xaxis.set_major_formatter(_raw)
        ax.yaxis.set_major_formatter(_pct)
        ax.grid(axis='y', alpha=0.4)

    fig.suptitle(modelname, fontsize=11)
    plt.tight_layout()

    path_out = os.path.join(cfg.DIR_MODELS, modelname, cfg.SUBDIR_TESTS, cfg.FNAME_PLOT_METRICS)
    fig.savefig(path_out)
    plt.close(fig)
    print(f'  saved {path_out}')


def plot_tradeoff_zoomed(metrics, modelname):
    """Same as plot_tradeoff but filtered to precision > PRECISION_ZOOM_MIN."""
    metrics_zoomed = metrics[metrics['precision'] > cfg.PRECISION_ZOOM_MIN]
    if metrics_zoomed.empty:
        print(f'  no thresholds with precision > {cfg.PRECISION_ZOOM_MIN:.0%}; skipping zoomed tradeoff plot')
        return

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(metrics_zoomed['precision'], metrics_zoomed['sensitivity'],
            color=COLORS['sensitivity'], linewidth=1.5)
    ax.set_xlabel('Precision')
    ax.set_ylabel('Sensitivity')
    ax.xaxis.set_major_formatter(_pct)
    ax.yaxis.set_major_formatter(_pct)
    ax.grid(axis='y', alpha=0.4)

    fig.suptitle(f'{modelname} (precision > {cfg.PRECISION_ZOOM_MIN:.0%})', fontsize=11)
    plt.tight_layout()

    path_out = os.path.join(cfg.DIR_MODELS, modelname, cfg.SUBDIR_TESTS, cfg.FNAME_PLOT_TRADEOFF_ZOOMED)
    fig.savefig(path_out)
    plt.close(fig)
    print(f'  saved {path_out}')


def plot_metrics_zoomed(metrics, modelname):
    """Same as plot_metrics but filtered to thresholds where precision > PRECISION_ZOOM_MIN."""
    metrics_zoomed = metrics[metrics['precision'] > cfg.PRECISION_ZOOM_MIN]
    if metrics_zoomed.empty:
        print(f'  no thresholds with precision > {cfg.PRECISION_ZOOM_MIN:.0%}; skipping zoomed plot')
        return

    metric_cols = [
        ('sensitivity',        'sensitivity'),
        ('precision',          'precision'),
        ('false positive rate', 'fpr'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    for (label, col), ax in zip(metric_cols, axes):
        ax.plot(metrics_zoomed['threshold'], metrics_zoomed[col],
                color=COLORS[label], linewidth=1.5)
        ax.set_xlabel('Threshold')
        ax.set_title(label)
        ax.xaxis.set_major_formatter(_raw)
        ax.yaxis.set_major_formatter(_pct)
        ax.grid(axis='y', alpha=0.4)

    fig.suptitle(f'{modelname} (precision > {cfg.PRECISION_ZOOM_MIN:.0%})', fontsize=11)
    plt.tight_layout()

    path_out = os.path.join(cfg.DIR_MODELS, modelname, cfg.SUBDIR_TESTS, cfg.FNAME_PLOT_METRICS_ZOOMED)
    fig.savefig(path_out)
    plt.close(fig)
    print(f'  saved {path_out}')
