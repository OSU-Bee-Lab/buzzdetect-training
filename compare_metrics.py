"""
Compare model performance across all evaluated models.

Usage:
  python compare_metrics.py               — top 5 models by sensitivity at 95% precision
  python compare_metrics.py <model>       — top 5 models, always including the given model
  python compare_metrics.py --top N       — show top N models

Reads models/<modelname>/tests/metrics.csv for each model found.
Reports sensitivity at 95% precision (averaged over the 94.5–95.5% band).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import config as cfg

PRECISION_TARGET = 0.95
PRECISION_BAND = 0.005  # ±0.5%


def sensitivity_at_95(metrics_path):
    df = pd.read_csv(metrics_path).dropna(subset=['precision', 'sensitivity'])
    band = df[
        (df['precision'] >= PRECISION_TARGET - PRECISION_BAND) &
        (df['precision'] <= PRECISION_TARGET + PRECISION_BAND)
    ]
    if not band.empty:
        return band['sensitivity'].mean(), 'band'

    # Interpolate when precision values don't land in the band
    df_sorted = df.sort_values('precision')
    below = df_sorted[df_sorted['precision'] < PRECISION_TARGET]
    above = df_sorted[df_sorted['precision'] > PRECISION_TARGET]
    if below.empty or above.empty:
        return None, None
    p0, s0 = below.iloc[-1][['precision', 'sensitivity']]
    p1, s1 = above.iloc[0][['precision', 'sensitivity']]
    t = (PRECISION_TARGET - p0) / (p1 - p0)
    return float(s0 + t * (s1 - s0)), 'interpolated'


def find_models():
    models = []
    for name in os.listdir(cfg.DIR_MODELS):
        path = os.path.join(cfg.DIR_MODELS, name, cfg.SUBDIR_TESTS, cfg.FNAME_METRICS)
        if os.path.isfile(path):
            models.append((name, path))
    return models


def main():
    parser = argparse.ArgumentParser(description='Compare model sensitivity at 95% precision.')
    parser.add_argument('model', nargs='?', help='Model to highlight (shown even if outside top N)')
    parser.add_argument('--top', type=int, default=5, metavar='N', help='Number of top models to show (default: 5)')
    args = parser.parse_args()

    current = args.model
    top_n = args.top

    print("WARNING: interpolated results may be unreliable — occurs when few data points fall near 95% precision")

    models = find_models()
    if not models:
        print('No models with metrics found.')
        sys.exit(1)

    rows = []
    for name, path in models:
        s, method = sensitivity_at_95(path)
        rows.append({'model': name, 'sensitivity': s, 'method': method})

    df = pd.DataFrame(rows).sort_values('sensitivity', ascending=False)

    if current is not None and current not in df['model'].values:
        print(f"Warning: '{current}' not found among evaluated models.")
        current = None

    top = df.head(top_n)
    if current is not None and current not in top['model'].values:
        cur_row = df[df['model'] == current]
        top = pd.concat([top, cur_row], ignore_index=True)

    def fmt(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 'N/A'
        return f'{val:.4f}'

    col_w = max(len(m) for m in top['model']) + 2
    header = f"{'Model':<{col_w}}  Sensitivity @ 95% Precision"
    print(header)
    print('-' * len(header))

    for _, row in top.iterrows():
        tags = []
        if row['model'] == current:
            tags.append('current')
        if row['method'] == 'interpolated':
            tags.append('interpolated')
        tag_str = f"  [{', '.join(tags)}]" if tags else ''
        print(f"{row['model']:<{col_w}}  {fmt(row['sensitivity'])}{tag_str}")


if __name__ == '__main__':
    main()
