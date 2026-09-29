"""Ladder bookkeeping: one jsonl line per trained rung, and the comparison table.

    ladder_record.py record --rung A --seed 1 --steps 12000 --name lad_A_s1 --wall SECONDS
    ladder_record.py table

Reads `.local/distill/runs/<name>/curve.json` (final val flips vs the teacher on
the V pool) and `.local/distill/eval/<name>/folds_sx.csv` (eval_folds.py). Plain
pandas/json, run in the train env.
"""
import argparse
import json
import os

import pandas as pd

MAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('/.claude/worktrees/')[0]
LOCAL = os.path.join(MAIN, '.local', 'distill')
LADDER = os.path.join(LOCAL, 'ladder.jsonl')
COL = 'sensitivity_exclquiet'
TIERS = ['faint', 'quiet', 'background', 'untagged', 'normal', 'loud']
REFS = [('baseline cv-baseline-v4-moderate', os.path.join(MAIN, 'models', 'cv-baseline-v4-moderate', 'folds_sx.csv')),
        ('teacher honest rotation (v4-ft-ps-e60-moderate)', os.path.join(MAIN, 'models', 'v4-ft-ps-e60-moderate', 'folds_sx.csv')),
        ('teacher ONNX via harness (trained on folds, inflated)', os.path.join(LOCAL, 'eval', 'teacher', 'folds_sx.csv'))]


def sx(path):
    t = pd.read_csv(path)
    per = t[t['fold'] != 'total'].set_index('fold')[COL]
    tot = t[t['fold'] == 'total'].iloc[0]
    return float(tot[COL]), [round(float(v), 3) for v in per], {k: (None if pd.isna(tot.get('sensitivity_' + k)) else float(tot['sensitivity_' + k])) for k in TIERS}, float(tot['sensitivity'])


def record(a):
    cur = json.load(open(os.path.join(LOCAL, 'runs', a.name, 'curve.json')))
    v = [x for x in cur['val'] if x.get('final')][-1]
    head, per, tiers, incl = sx(os.path.join(LOCAL, 'eval', a.name, 'folds_sx.csv'))
    row = {'rung': a.rung, 'seed': a.seed, 'steps': a.steps, 'name': a.name,
           'val_frames': v['frames'], 'buzz_teacher_pos': v['buzz_teacher'], 'buzz_student_pos': v['buzz_student'],
           'buzz_gained': v['buzz_gained'], 'buzz_lost': v['buzz_lost'],
           'lost_pct': round(100 * v['buzz_lost'] / max(1, v['buzz_teacher']), 2),
           'gained_pct': round(100 * v['buzz_gained'] / max(1, v['buzz_teacher']), 2),
           'other_gained': v['other_gained'], 'other_lost': v['other_lost'],
           'mae_live': round(v['mae_live'], 4), 'mae_buzz': round(v['mae_buzz'], 4),
           'headline': head, 'headline_inclusive': incl, 'per_fold': per, 'tiers': tiers,
           'wall_s': round(a.wall)}
    with open(LADDER, 'a') as f:
        f.write(json.dumps(row) + '\n')
    print('[ladder]', json.dumps(row))


def table(a):
    rows = [json.loads(l) for l in open(LADDER)] if os.path.exists(LADDER) else []
    print('\nrun        steps  V buzz: teacher+  lost%  gained%  | other g/l | mae   | headline (excl-quiet @0.005)  per fold           | wall')
    for r in rows:
        print(f'{r["rung"]}/s{r["seed"]:<7} {r["steps"]:>5}  {r["buzz_teacher_pos"]:>14}  {r["lost_pct"]:>5.1f}  {r["gained_pct"]:>6.1f}  | '
              f'{r["other_gained"]:>5}/{r["other_lost"]:<5} | {r["mae_live"]:.3f} | {r["headline"]:.3f}   {r["per_fold"]}  | {r["wall_s"] / 60:.0f} min')
    print('\ncomparison points (same column, total row, same 5 rotating folds in every file):')
    base = None
    for name, p in REFS:
        if os.path.exists(p):
            h, per, _, incl = sx(p)
            base = h if base is None else base
            print(f'  {name:55s} {h:.3f}  (inclusive {incl:.3f})  {per}')
    if base:
        print(f'  Luke floor: 50% of baseline = {base / 2:.3f}')
        for r in rows:
            print(f'  {r["rung"]}/s{r["seed"]}: {r["headline"]:.3f} = {100 * r["headline"] / base:.0f}% of baseline')
    a_rows = [r for r in rows if r['rung'] == 'A']
    if len(a_rows) >= 2:
        spread = abs(a_rows[0]['lost_pct'] - a_rows[1]['lost_pct'])
        a_mean = sum(r['lost_pct'] for r in a_rows) / len(a_rows)
        print(f'\nstopping rule: A repeat spread of lost% = {spread:.2f} (A mean {a_mean:.2f})')
        for r in rows:
            if r['rung'] != 'A':
                gain = a_mean - r['lost_pct']
                print(f'  {r["rung"]}: lost% {r["lost_pct"]:.2f}, improvement over A {gain:.2f} -> '
                      f'{"ADVANCE to next rung" if gain > spread else "STOP (not more than the spread)"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['record', 'table'])
    ap.add_argument('--rung'), ap.add_argument('--seed', type=int), ap.add_argument('--steps', type=int)
    ap.add_argument('--name'), ap.add_argument('--wall', type=float, default=0)
    a = ap.parse_args()
    {'record': record, 'table': table}[a.phase](a)
