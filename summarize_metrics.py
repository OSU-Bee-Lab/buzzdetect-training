#!/usr/bin/env python3
"""Print sensitivity at 90/95/99% precision for one or more models."""

import os
import sys

import pandas as pd

import config as cfg

PRECISION_TARGETS = [0.90, 0.95, 0.99]


def sensitivity_at_precision(df, target):
    eligible = df[df['precision'] >= target]
    if eligible.empty:
        return None
    return eligible['sensitivity'].max()


def summarize(modelnames):
    rows = []
    for name in modelnames:
        path = os.path.join(cfg.DIR_MODELS, name, cfg.SUBDIR_TESTS, cfg.FNAME_METRICS)
        if not os.path.exists(path):
            print(f"metrics.csv not found for {name!r} (expected {path})", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        row = {'model': name}
        for t in PRECISION_TARGETS:
            row[f'sens@{int(t*100)}prec'] = sensitivity_at_precision(df, t)
        rows.append(row)

    if not rows:
        return

    result = pd.DataFrame(rows).set_index('model')

    def fmt(v):
        return f'{v:.1%}' if pd.notna(v) else '—'

    for col in result.columns:
        result[col] = result[col].map(fmt)

    print(result.to_string())


if __name__ == '__main__':
    if len(sys.argv) < 2:
        available = [
            d for d in os.listdir(cfg.DIR_MODELS)
            if os.path.isfile(os.path.join(cfg.DIR_MODELS, d, cfg.SUBDIR_TESTS, cfg.FNAME_METRICS))
        ]
        print('Usage: python summarize_metrics.py <model> [<model> ...]')
        if available:
            print(f'Available: {", ".join(sorted(available))}')
        sys.exit(1)

    summarize(sys.argv[1:])
