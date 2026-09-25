"""Shared helpers for per-set metrics scripts."""

import os

import numpy as np
import pandas as pd
from scipy import stats

import config as cfg

PRECISION_BAND = 0.005  # ±0.5% band around target for direct reads


def sensitivity_at_precision(metrics_path, target=0.95):
    """Return (value, method) where method is 'band' or 'interpolated', or (None, None)."""
    df = pd.read_csv(metrics_path).dropna(subset=['precision', 'sensitivity'])
    band = df[
        (df['precision'] >= target - PRECISION_BAND) &
        (df['precision'] <= target + PRECISION_BAND)
    ]
    if not band.empty:
        return band['sensitivity'].mean(), 'band'

    df_sorted = df.sort_values('precision')
    below = df_sorted[df_sorted['precision'] < target]
    above = df_sorted[df_sorted['precision'] > target]
    if below.empty or above.empty:
        return None, None
    p0, s0 = below.iloc[-1][['precision', 'sensitivity']]
    p1, s1 = above.iloc[0][['precision', 'sensitivity']]
    t = (target - p0) / (p1 - p0)
    return float(s0 + t * (s1 - s0)), 'interpolated'


def model_metrics_path(modelname):
    return os.path.join(cfg.DIR_MODELS, modelname, cfg.SUBDIR_TESTS, cfg.FNAME_METRICS)


def ci_t(values):
    """95% CI via t-distribution. Returns (lo, hi) or None if n < 2."""
    n = len(values)
    if n < 2:
        return None
    mean = np.mean(values)
    sem = np.std(values, ddof=1) / np.sqrt(n)
    lo, hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
    return float(lo), float(hi)


def load_set(base, n=None):
    """
    Load sensitivity values for models <base>_v1 through <base>_vN.
    If n is None, auto-detects all <base>_v<int> models in models/.
    Returns (values, any_interpolated) where None entries are dropped.
    """
    import re
    if n is None:
        pattern = re.compile(rf'^{re.escape(base)}_v(\d+)$')
        indices = sorted(
            int(m.group(1))
            for name in os.listdir(cfg.DIR_MODELS)
            if (m := pattern.match(name))
        )
    else:
        indices = range(1, n + 1)

    values = []
    any_interpolated = False
    for i in indices:
        path = model_metrics_path(f'{base}_v{i}')
        if not os.path.isfile(path):
            continue
        val, method = sensitivity_at_precision(path)
        if val is not None:
            values.append(val)
            if method == 'interpolated':
                any_interpolated = True
    return values, any_interpolated
