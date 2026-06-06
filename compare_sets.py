"""
Rank experiment sets by mean sensitivity at 95% precision, with 95% CI.

Usage:
  python compare_sets.py                  — all sets found in models/
  python compare_sets.py <set> [<set>...] — specific sets
  python compare_sets.py --top N          — show top N (default: 10)
  python compare_sets.py --precision 0.99 — different operating point
  python compare_sets.py --min-runs 3     — require at least N valid runs (default: 2)

A "set" is a group of models named <base>_v1 ... <base>_vN.
Individual models without the _v<N> pattern are ignored; use compare_metrics.py for those.
"""

import argparse
import os
import re
import sys

import numpy as np

import config as cfg
from metrics_utils import load_set, ci_t


def discover_sets(min_runs=2):
    """Scan models/ for groups of <base>_v1 ... <base>_vN with at least min_runs members."""
    pattern = re.compile(r'^(.+)_v(\d+)$')
    bases = {}
    for name in os.listdir(cfg.DIR_MODELS):
        m = pattern.match(name)
        if not m:
            continue
        base = m.group(1)
        bases[base] = bases.get(base, 0) + 1
    return [b for b, count in bases.items() if count >= min_runs]


def fmt_ci(ci):
    if ci is None:
        return '—'
    return f'[{ci[0]:.3f}, {ci[1]:.3f}]'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sets', nargs='*', metavar='set_base')
    parser.add_argument('--top', type=int, default=10, metavar='N')
    parser.add_argument('--n', type=int, default=None, help='Max runs per set (default: auto-detect)')
    parser.add_argument('--precision', type=float, default=0.95)
    parser.add_argument('--min-runs', type=int, default=2, dest='min_runs',
                        help='Min valid runs to include a set (default: 2)')
    args = parser.parse_args()

    bases = args.sets if args.sets else discover_sets(min_runs=args.min_runs)
    if not bases:
        print('No experiment sets found.')
        sys.exit(1)

    rows = []
    any_interp = False
    for base in bases:
        values, interp = load_set(base, n=args.n)
        if len(values) < args.min_runs:
            continue
        if interp:
            any_interp = True
        mean = np.mean(values)
        ci = ci_t(values)
        rows.append({
            'base': base,
            'n': len(values),
            'mean': mean,
            'ci': ci,
            'interp': interp,
        })

    if not rows:
        print('No sets with sufficient runs found.')
        sys.exit(1)

    rows.sort(key=lambda r: r['mean'], reverse=True)
    rows = rows[:args.top]

    best_ci = rows[0]['ci']

    col_w = max(len(r['base']) for r in rows) + 2
    header = f"{'Set':<{col_w}}  {'n':>3}  {'mean':>7}  {'95% CI':<20}  vs best"
    print(header)
    print('-' * (len(header) + 4))
    for i, r in enumerate(rows):
        tag = ''
        if i == 0:
            tag = '—'
        elif r['ci'] is None or best_ci is None:
            tag = 'n/a'
        elif r['ci'][1] >= best_ci[0]:
            tag = 'overlap'
        else:
            tag = 'below'

        interp_mark = ' *' if r['interp'] else ''
        print(
            f"{r['base']:<{col_w}}{interp_mark}  "
            f"{r['n']:>3}  "
            f"{r['mean']:>7.3f}  "
            f"{fmt_ci(r['ci']):<20}  "
            f"{tag}"
        )

    if any_interp:
        print('* contains interpolated values — may be unreliable')


if __name__ == '__main__':
    main()
