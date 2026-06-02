"""
Evaluate and compare model performance.

Usage: python eval.py <current_model>

Reads models/<modelname>/tests/metrics.csv for each model found.
Reports sensitivity at 95% precision (averaged over the 94.5–95.5% band).
The specified current model appears first and is marked as baseline.
"""

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
    if len(sys.argv) != 2:
        print(f'Usage: python {sys.argv[0]} <current_model>')
        sys.exit(1)

    current = sys.argv[1]

    models = find_models()
    if not models:
        print('No models with metrics found.')
        sys.exit(1)

    rows = []
    for name, path in models:
        s, method = sensitivity_at_95(path)
        rows.append({'model': name, 'sensitivity': s, 'method': method})

    df = pd.DataFrame(rows)

    if current not in df['model'].values:
        print(f"Warning: '{current}' not found among evaluated models.")
        current_found = False
    else:
        current_found = True

    def fmt(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 'N/A'
        return f'{val:.4f}'

    # Build display rows: current first, then others sorted by sensitivity desc
    if current_found:
        cur_row = df[df['model'] == current].iloc[0]
        others = df[df['model'] != current].sort_values('sensitivity', ascending=False)
        ordered = pd.concat([cur_row.to_frame().T, others], ignore_index=True)
    else:
        ordered = df.sort_values('sensitivity', ascending=False)

    col_w = max(len(m) for m in ordered['model']) + 2
    header = f"{'Model':<{col_w}}  Sensitivity @ 95% Precision"
    print(header)
    print('-' * len(header))

    for _, row in ordered.iterrows():
        tags = []
        if row['model'] == current:
            tags.append('current')
        if row['method'] == 'interpolated':
            tags.append('interpolated')
        tag_str = f"  [{', '.join(tags)}]" if tags else ''
        print(f"{row['model']:<{col_w}}  {fmt(row['sensitivity'])}{tag_str}")


if __name__ == '__main__':
    main()
