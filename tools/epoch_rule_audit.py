"""Compare epoch-selection rules on every CV run's stored per-fold curves.

The evidence behind NEXT_ERA.md's scoring change (2026-10-21). Every rotation
saves `val_sens_fpr0.005_curve` (held-out sensitivity at every epoch, quiet buzz
INCLUDED) and `val_loss_curve` in its summary.json, so any epoch rule can be
re-scored offline, with no retraining. The headline (quiet excluded), the tiers
and the event-bootstrap SD can't be: per-epoch predictions are not stored.

    python tools/epoch_rule_audit.py runs [--json out.json]     # every run, every rule
    python tools/epoch_rule_audit.py log                        # 03_train/log.jsonl deltas re-scored
    python tools/epoch_rule_audit.py draws <model> <model> ... [--epochs N]

Rules, each a mean over a run's folds:
  final    the last epoch (the current rule)
  own      each fold at its own sensitivity peak (selection on the scored fold)
  xfold    each fold at the argmax of the mean of the OTHER folds' curves
           (honest_epoch.py's xfold-pooled; the proposed rule)
  vl_own   each fold at its own val_loss argmin (old early stopping, no patience)
  vl_cons  one epoch for all folds from train._consensus_epoch (the shipped model's rule)
  jitter   median per-fold SD of the curve's second differences / sqrt(6) over the
           last half: epoch-to-epoch noise, assuming a smooth underlying curve

`draws` takes repeat runs of one recipe. For each draw and fold it picks the
fold's own peak epoch, then scores that epoch on the mean of the other draws:
own minus that is training-noise inflation; that minus the other draws' final
epoch is the real gain from per-fold timing. It also prints the pooled draw-mean
curve, which is what sets an epoch budget.

Models are found under models/ and .local/worktrees/*/models/; test_* are skipped,
and the first directory found wins for a name. No TensorFlow.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg

KEY_SENS = 'val_sens_fpr0.005_curve'
RULES = ('final', 'own', 'xfold', 'vl_own', 'vl_cons')
# log.jsonl names whose model directory is named differently
ALIASES = {'ps-sharedtiled': 'v4-ft-ps-sharedtiled', 'ps-untiled': 'v4-ft-ps-untiled',
           'ps-fast': 'v4-ft-ps-fast', 'ps-d8-ktails': 'ps-d8-ktails3'}


def model_dirs():
    found = {}
    pattern_wt = os.path.join(cfg.ROOT, '.local', 'worktrees', '*', 'models', '*', '')
    for d in sorted(glob.glob(os.path.join(cfg.DIR_MODELS, '*', ''))) + sorted(glob.glob(pattern_wt)):
        name = os.path.basename(d.rstrip('/'))
        if not name.startswith('test_'):
            found.setdefault(name, d)
    return found


def load_folds(model_dir):
    """{fold: summary} if every fold has full, equal-length curves, else None."""
    rows = {}
    for path in glob.glob(os.path.join(model_dir, 'folds', '**', 'summary.json'), recursive=True):
        s = json.load(open(path))
        curve = s.get(KEY_SENS)
        if not curve or any(x is None for x in curve) or not s.get('val_loss_curve'):
            return None
        rows[path.split('/')[-2]] = s
    if len(rows) < 5 or len({len(r[KEY_SENS]) for r in rows.values()}) != 1:
        return None
    return rows


def consensus_epoch(rows, tol=0.01):
    """train._consensus_epoch, copied so this tool needs no TensorFlow import."""
    curves = [(r['val_loss_curve'], r.get('frames_val', 1)) for r in rows]
    length = max(len(c) for c, _ in curves)
    stack, weights = [], []
    for curve, frames in curves:
        arr = np.asarray(curve, float)
        arr = np.minimum.accumulate(np.concatenate([arr, np.full(length - len(arr), arr.min())]))
        span = arr.max() - arr.min()
        stack.append((arr - arr.min()) / span if span else np.zeros(length))
        weights.append(max(frames, 1))
    mean = np.average(stack, axis=0, weights=weights)
    return int(np.argmax(mean <= mean.min() + tol * (mean.max() - mean.min()))) + 1


def audit(rows):
    folds = list(rows)
    sens = {f: np.asarray(rows[f][KEY_SENS]) for f in folds}
    loss = {f: np.asarray(rows[f]['val_loss_curve']) for f in folds}
    n = len(next(iter(sens.values())))
    ce = consensus_epoch(list(rows.values()))
    out = {r: [] for r in RULES}
    jitter, ep_peak, ep_xfold, ep_loss = [], [], [], []
    for f in folds:
        c = sens[f]
        e_x = int(np.argmax(np.mean([sens[g] for g in folds if g != f], axis=0)))
        e_l = int(np.argmin(loss[f]))
        out['final'].append(c[-1])
        out['own'].append(c.max())
        out['xfold'].append(c[e_x])
        out['vl_own'].append(c[e_l])
        out['vl_cons'].append(c[ce - 1])
        if n - n // 2 >= 6:
            jitter.append(np.std(np.diff(c[n // 2:], 2)) / np.sqrt(6))
        ep_peak.append(int(np.argmax(c)) + 1)
        ep_xfold.append(e_x + 1)
        ep_loss.append(e_l + 1)
    res = {r: float(np.mean(v)) for r, v in out.items()}
    res.update(n=n, cons_ep=ce, jitter=float(np.median(jitter)) if jitter else float('nan'),
               peak_ep=int(np.median(ep_peak)), xf_ep=int(np.median(ep_xfold)), vl_ep=int(np.median(ep_loss)))
    return res


def audit_all():
    res = {}
    for name, d in model_dirs().items():
        rows = load_folds(d)
        if rows:
            res[name] = audit(rows)
    return res


def cmd_runs(args):
    res = audit_all()
    print(f'{"run":34}{"n":>4}' + ''.join(f'{r:>8}' for r in RULES) + f'{"jitter":>8}'
          '   median epochs: peak/xfold/val_loss, consensus')
    for name, r in sorted(res.items(), key=lambda kv: (kv[1]['n'], kv[0])):
        print(f'{name[:34]:34}{r["n"]:4}' + ''.join(f'{r[k]:8.3f}' for k in RULES) + f'{r["jitter"]:8.3f}'
              f'   {r["peak_ep"]}/{r["xf_ep"]}/{r["vl_ep"]}, {r["cons_ep"]}')
    if args.json:
        json.dump(res, open(args.json, 'w'), indent=1)


def cmd_log(args):
    res = audit_all()
    rows, missing = [], []
    for line in open(cfg.TRAIN_LOG):
        e = json.loads(line)
        name, base = ALIASES.get(e['name'], e['name']), e['baseline']['model']
        if name not in res or base not in res:
            missing.append(e['name'])
            continue
        if name != base:
            old = res[name]['final'] - res[base]['final']
            new = res[name]['xfold'] - res[base]['xfold']
            rows.append((e['name'], base, old, new))
    print(f'{"experiment":26}{"baseline":24}{"final":>9}{"xfold":>9}{"change":>9}   (deltas, quiet included)')
    for name, base, old, new in sorted(rows, key=lambda r: -abs(r[3] - r[2])):
        print(f'{name[:26]:26}{base[:24]:24}{old:+9.3f}{new:+9.3f}{new - old:+9.3f}')
    print(f'\nnot re-scorable ({len(missing)}): ' + ', '.join(missing))


def cmd_draws(args):
    dirs = model_dirs()
    draws = []
    for name in args.models:
        rows = load_folds(dirs[name]) if name in dirs else None
        if rows is None:
            sys.exit(f'{name}: not found, or curves missing/uneven')
        draws.append({f: np.asarray(s[KEY_SENS], float) for f, s in rows.items()})
    n = args.epochs or min(len(c) for d in draws for c in d.values())
    folds = sorted(set.intersection(*[set(d) for d in draws]))
    draws = [{f: d[f][:n] for f in folds} for d in draws]
    k = len(draws)
    own, honest, final = [], [], []
    for i in range(k):
        for f in folds:
            c = draws[i][f]
            rest = np.mean([draws[j][f] for j in range(k) if j != i], axis=0)
            e = int(np.argmax(c))
            own.append(c[e]); honest.append(rest[e]); final.append(rest[-1])
    own, honest, final = map(np.mean, (own, honest, final))
    print(f'{k} draws, {len(folds)} folds, epochs 1..{n} (sensitivity, quiet included)')
    print(f'  own peak, scored on the draw that picked it   {own:.3f}')
    print(f'  same epoch, scored on the other draws         {honest:.3f}   inflation {own - honest:+.3f}')
    print(f'  final epoch, scored on the other draws        {final:.3f}   real timing gain {honest - final:+.3f}')
    per_fold = np.array([[np.mean([d[f][e] for d in draws]) for e in range(n)] for f in folds])
    spread = np.array([[np.std([d[f][e] for d in draws], ddof=1) for e in range(n)] for f in folds])
    pooled = per_fold.mean(0)
    print('  pooled draw-mean curve: ' + ' '.join(f'e{e}:{pooled[e - 1]:.3f}' for e in range(1, n + 1)
                                                   if e in (1, 3, 5) or e % 5 == 0 or e == n))
    print(f'  pooled peak e{int(np.argmax(pooled)) + 1} ({pooled.max():.3f}); per-fold peaks of the draw mean: '
          f'{[int(np.argmax(r)) + 1 for r in per_fold]}')
    print(f'  draw-to-draw SD per fold per epoch, median over the last half: {np.median(spread[:, n // 2:]):.3f}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('runs'); r.add_argument('--json'); r.set_defaults(fn=cmd_runs)
    sub.add_parser('log').set_defaults(fn=cmd_log)
    d = sub.add_parser('draws'); d.add_argument('models', nargs='+'); d.add_argument('--epochs', type=int)
    d.set_defaults(fn=cmd_draws)
    args = p.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
