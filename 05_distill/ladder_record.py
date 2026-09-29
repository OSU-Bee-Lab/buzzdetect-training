"""Ladder bookkeeping: one jsonl line per trained run, the comparison table, and
the variant-qualification decision.

    ladder_record.py record --rung A --seed 1 --steps 7000 --name lad_A_s1 --wall SECONDS [--arch a0.50] [--loader mem]
    ladder_record.py table
    ladder_record.py decide          # prints the qualifying rung-D variant arch key, or 'none'
    ladder_record.py gate            # exit 0 if rung C beat B by more than the A spread, else 1

Reads `.local/distill/runs/<name>/curve.json` (final val flips vs the teacher on
the V pool), `.local/distill/eval/<name>/folds_sx.csv` (eval_folds.py) and, if
present, `.local/distill/models/<name>/speed_{20,200}.json`. Train env.
"""
import argparse
import json
import os
import sys

import pandas as pd

MAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('/.claude/worktrees/')[0]
LOCAL = os.path.join(MAIN, '.local', 'distill')
LADDER = os.path.join(LOCAL, 'ladder.jsonl')
COL = 'sensitivity_exclquiet'
TIERS = ['faint', 'quiet', 'background', 'untagged', 'normal', 'loud']
HEADLINE_TOL = 0.03
REFS = [('baseline cv-baseline-v4-moderate', os.path.join(MAIN, 'models', 'cv-baseline-v4-moderate', 'folds_sx.csv')),
        ('teacher honest rotation (v4-ft-ps-e60-moderate)', os.path.join(MAIN, 'models', 'v4-ft-ps-e60-moderate', 'folds_sx.csv')),
        ('teacher ONNX via harness (trained on folds, inflated)', os.path.join(LOCAL, 'eval', 'teacher', 'folds_sx.csv'))]


def sx(path):
    t = pd.read_csv(path)
    per = t[t['fold'] != 'total'].set_index('fold')[COL]
    tot = t[t['fold'] == 'total'].iloc[0]
    return (float(tot[COL]), [round(float(v), 3) for v in per],
            {k: (None if pd.isna(tot.get('sensitivity_' + k)) else float(tot['sensitivity_' + k])) for k in TIERS},
            float(tot['sensitivity']))


def speed(name):
    out = {}
    for s in (20, 200):
        p = os.path.join(LOCAL, 'models', name, f'speed_{s}.json')
        if os.path.exists(p):
            j = json.load(open(p))
            out[f'gpu{s}'] = round(j['gpu'])
            out[f'x_yamnet{s}'] = round(j['x_yamnet'], 3)
    return out


def rows():
    return [json.loads(l) for l in open(LADDER)] if os.path.exists(LADDER) else []


def is_ladder(r):
    """The alpha-0.5 ladder: in-memory rungs A-C, streamed rung D."""
    arch = r.get('arch', 'a0.50')
    loader = r.get('loader', 'mem')
    return arch == 'a0.50' and (loader == 'mem' or r['rung'] == 'D')


def levels():
    """rung -> (mean lost_pct, mean headline) over the ladder rows."""
    out = {}
    for rung in 'ABCD':
        rs = [r for r in rows() if is_ladder(r) and r['rung'] == rung]
        if rs:
            out[rung] = (sum(r['lost_pct'] for r in rs) / len(rs), sum(r['headline'] for r in rs) / len(rs))
    return out


def a_spread():
    a = [r for r in rows() if is_ladder(r) and r['rung'] == 'A']
    return abs(a[0]['lost_pct'] - a[1]['lost_pct']) if len(a) >= 2 else None


def record(a):
    cur = json.load(open(os.path.join(LOCAL, 'runs', a.name, 'curve.json')))
    v = [x for x in cur['val'] if x.get('final')][-1]
    head, per, tiers, incl = sx(os.path.join(LOCAL, 'eval', a.name, 'folds_sx.csv'))
    row = {'rung': a.rung, 'seed': a.seed, 'steps': a.steps, 'name': a.name, 'arch': a.arch, 'loader': a.loader,
           'val_frames': v['frames'], 'buzz_teacher_pos': v['buzz_teacher'], 'buzz_student_pos': v['buzz_student'],
           'buzz_gained': v['buzz_gained'], 'buzz_lost': v['buzz_lost'],
           'lost_pct': round(100 * v['buzz_lost'] / max(1, v['buzz_teacher']), 2),
           'gained_pct': round(100 * v['buzz_gained'] / max(1, v['buzz_teacher']), 2),
           'other_gained': v['other_gained'], 'other_lost': v['other_lost'],
           'mae_live': round(v['mae_live'], 4), 'mae_buzz': round(v['mae_buzz'], 4),
           'headline': head, 'headline_inclusive': incl, 'per_fold': per, 'tiers': tiers,
           'wall_s': round(a.wall), **speed(a.name)}
    with open(LADDER, 'a') as f:
        f.write(json.dumps(row) + '\n')
    print('[ladder]', json.dumps(row))


def table(a):
    rs = rows()
    print('\nrun                    steps  V buzz lost%  gained%  | other g/l   | mae   | headline (excl-quiet @0.005)  per fold                 | GPU x YAMNet 20s/200s | wall')
    for r in rs:
        tag = f'{r["rung"]}/{r.get("arch", "a0.50")}/s{r["seed"]}' + ('/stream' if r.get('loader') == 'stream' else '')
        sp = f'{r.get("x_yamnet20", float("nan")):.2f}/{r.get("x_yamnet200", float("nan")):.2f}'
        print(f'{tag:22s} {r["steps"]:>5}  {r["lost_pct"]:>11.1f}  {r["gained_pct"]:>6.1f}  | '
              f'{r["other_gained"]:>5}/{r["other_lost"]:<5} | {r["mae_live"]:.3f} | {r["headline"]:.3f}   {r["per_fold"]}  | {sp:>10} | {r["wall_s"] / 60:.0f} min')
    print('\ncomparison points (same column, total row, same 5 rotating folds in every file):')
    base = None
    for name, p in REFS:
        if os.path.exists(p):
            h, per, _, incl = sx(p)
            base = h if base is None else base
            print(f'  {name:55s} {h:.3f}  (inclusive {incl:.3f})  {per}')
    if base:
        print(f'  floors: headline >= {base / 2:.3f} (50% of baseline {base:.3f}); GPU >= 1.5x YAMNet at 200 s')
        for r in rs:
            print(f'  {r["name"]}: {r["headline"]:.3f} = {100 * r["headline"] / base:.0f}% of baseline')
    sp, lv = a_spread(), levels()
    if sp is not None:
        print(f'\nstopping rule: A repeat spread of lost% = {sp:.2f}')
        prev = None
        for rung in 'ABCD':
            if rung in lv:
                if prev:
                    gain = lv[prev][0] - lv[rung][0]
                    print(f'  {prev}->{rung}: lost% {lv[prev][0]:.2f} -> {lv[rung][0]:.2f}, improvement {gain:.2f} -> '
                          f'{"ADVANCE" if gain > sp else "STOP (not more than the spread)"}')
                prev = rung


def gate(a):
    sp, lv = a_spread(), levels()
    ok = sp is not None and 'B' in lv and 'C' in lv and lv['B'][0] - lv['C'][0] > sp
    print(f'[gate] C vs B: lost% {lv.get("B", (None,))[0]} -> {lv.get("C", (None,))[0]}, spread {sp}: '
          f'{"advance to D" if ok else "STOP: rung C does not beat B by more than the A spread"}')
    sys.exit(0 if ok else 1)


def streamcheck(a):
    """Rung B through the streaming loader must reproduce the in-memory rung B within the A spread."""
    sp = a_spread()
    mem = [r for r in rows() if r['rung'] == 'B' and r.get('arch', 'a0.50') == 'a0.50'
           and r.get('loader', 'mem') == 'mem' and r['seed'] == 1][0]
    st_ = [r for r in rows() if r['rung'] == 'B' and r.get('arch', 'a0.50') == 'a0.50' and r.get('loader') == 'stream'][-1]
    d = abs(mem['lost_pct'] - st_['lost_pct'])
    ok = d <= sp
    print(f'[streamcheck] B mem lost% {mem["lost_pct"]} (headline {mem["headline"]}) vs stream {st_["lost_pct"]} '
          f'(headline {st_["headline"]}): diff {d:.2f} vs A spread {sp:.2f}: {"OK" if ok else "STREAM LOADER DOES NOT REPRODUCE"}')
    sys.exit(0 if ok else 1)


def decide(a):
    """Which rung-D variant (if any) qualifies: within the A spread of a0.50 on lost% (rung B, seed 1)
    AND headline within HEADLINE_TOL of it; the faster (GPU, 200 s) if several."""
    sp = a_spread()
    b = [r for r in rows() if is_ladder(r) and r['rung'] == 'B' and r['seed'] == 1][0]
    ok = []
    for r in rows():
        if r['rung'] == 'B' and r.get('arch', 'a0.50') != 'a0.50' and r.get('loader', 'mem') == 'mem':
            good = (r['lost_pct'] - b['lost_pct'] <= sp) and (r['headline'] >= b['headline'] - HEADLINE_TOL)
            print(f'[decide] {r["arch"]}: lost% {r["lost_pct"]} vs {b["lost_pct"]} (spread {sp:.2f}), headline '
                  f'{r["headline"]} vs {b["headline"]} (tol {HEADLINE_TOL}), gpu200 {r.get("gpu200")}: '
                  f'{"QUALIFIES" if good else "no"}', file=sys.stderr)
            if good:
                ok.append(r)
    print(max(ok, key=lambda r: r.get('gpu200', 0))['arch'] if ok else 'none')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['record', 'table', 'decide', 'gate', 'streamcheck'])
    ap.add_argument('--rung'), ap.add_argument('--seed', type=int), ap.add_argument('--steps', type=int)
    ap.add_argument('--name'), ap.add_argument('--wall', type=float, default=0)
    ap.add_argument('--arch', default='a0.50'), ap.add_argument('--loader', default='mem')
    a = ap.parse_args()
    {'record': record, 'table': table, 'decide': decide, 'gate': gate,
     'streamcheck': streamcheck}[a.phase](a)
