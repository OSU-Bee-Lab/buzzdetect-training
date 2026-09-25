"""
Evaluate one or more multi-run experiment sets, reporting CI.

Usage:
  python evaluate_set.py <set_base> [<set_base> ...]   — summary table
  python evaluate_set.py <set_base> --detail            — per-run values + summary
  python evaluate_set.py <set_base> --n 8              — expect up to 8 runs (default: 5)
  python evaluate_set.py <set_base> --precision 0.99   — different operating point
"""

import argparse
import sys

import numpy as np

from metrics_utils import load_set, ci_t


def fmt_ci(ci):
    if ci is None:
        return '—'
    return f'[{ci[0]:.3f}, {ci[1]:.3f}]'


def summarize_set(base, n, precision, detail):
    values, any_interp = load_set(base, n=n)
    if not values:
        return None, f'{base}: no runs found'

    warn = ' *' if any_interp else ''
    mean = np.mean(values)
    median = np.median(values)
    std = np.std(values, ddof=1) if len(values) > 1 else float('nan')
    ci = ci_t(values)

    if detail:
        run_str = ', '.join(f'{v:.3f}' for v in sorted(values))
        print(f'\nSet: {base}{warn} (n={len(values)})')
        print(f'  Runs:   {run_str}')
        print(f'  mean={mean:.3f}  median={median:.3f}  std={std:.3f}  95% CI={fmt_ci(ci)}')

    return {
        'base': base,
        'n': len(values),
        'mean': mean,
        'median': median,
        'std': std,
        'ci': ci,
        'any_interp': any_interp,
    }, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sets', nargs='+', metavar='set_base')
    parser.add_argument('--n', type=int, default=None, help='Max runs to look for (default: auto-detect)')
    parser.add_argument('--precision', type=float, default=0.95)
    parser.add_argument('--detail', action='store_true', help='Print individual run values')
    args = parser.parse_args()

    results = []
    errors = []
    for base in args.sets:
        row, err = summarize_set(base, args.n, args.precision, args.detail)
        if err:
            errors.append(err)
        else:
            results.append(row)

    for e in errors:
        print(e, file=sys.stderr)

    if not results:
        sys.exit(1)

    if args.detail:
        return

    col_w = max(len(r['base']) for r in results) + 2
    header = f"{'Set':<{col_w}}  {'n':>3}  {'median':>7}  {'mean':>7}  {'std':>6}  {'95% CI'}"
    print(header)
    print('-' * (len(header) + 10))
    for r in results:
        interp = ' *' if r['any_interp'] else ''
        print(
            f"{r['base']:<{col_w}}{interp}  "
            f"{r['n']:>3}  "
            f"{r['median']:>7.3f}  "
            f"{r['mean']:>7.3f}  "
            f"{r['std']:>6.3f}  "
            f"{fmt_ci(r['ci'])}"
        )
    if any(r['any_interp'] for r in results):
        print('* contains interpolated values — may be unreliable')


if __name__ == '__main__':
    main()
