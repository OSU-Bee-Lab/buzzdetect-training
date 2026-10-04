"""Ladder bookkeeping: one jsonl line per trained run, the comparison table, and
the variant-qualification decision.

    ladder_record.py record --rung A --seed 1 --steps 7000 --name lad_A_s1 --wall SECONDS [--arch a0.50] [--loader mem]
    ladder_record.py table
    ladder_record.py decide          # prints the qualifying rung-D variant arch key, or 'none'
    ladder_record.py gate            # exit 0 if rung C beat B by more than the A spread, else 1
    ladder_record.py wsd --name <trunk>   # step-budget curve: a WSD trunk's decayed branches (+ cosine reference)
    ladder_record.py proxy           # how well each V readout (hit at 0, hit@K, mae) tracks the fold headline
    ladder_record.py spread          # prints the rung-A hit@K spread (the --wsd-stop tolerance); exit 1 if unknown

Reads `05_distill/data/runs/<name>/curve.json` (final val flips vs the teacher on
the V pool), `05_distill/data/eval/<name>/folds_sx.csv` (eval_folds.py) and, if
present, `05_distill/data/models/<name>/speed_{20,200}.json`. Train env.
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpaths as D  # noqa: E402

MAIN = D.MAIN
LOCAL = D.LOCAL                # 05_distill/data/<teacher>
LADDER = D.LADDER
COL = 'sensitivity_exclquiet'
TIERS = ['faint', 'quiet', 'background', 'untagged', 'normal', 'loud']
HEADLINE_TOL = 0.03
REFS = [(f'baseline {D.BASELINE}', os.path.join(MAIN, 'models', D.BASELINE, 'folds_sx.csv')),
        (f'teacher honest rotation ({D.TEACHER})', os.path.join(MAIN, 'models', D.TEACHER, 'folds_sx.csv')),
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


def is_cosine(r):
    """Rows from before the schedule field are cosine. WSD branches stay out of every ladder/frontier
    comparison: they are points on a step-budget curve (`wsd`), not alternatives to a 7k cosine run."""
    return r.get('schedule', 'cosine') == 'cosine'


def is_ladder(r):
    """The alpha-0.5 ladder: in-memory rungs A-C, streamed rung D."""
    if not is_cosine(r):
        return False
    arch = r.get('arch', 'a0.50')
    loader = r.get('loader', 'mem')
    return arch == 'a0.50' and (loader == 'mem' or r['rung'] == 'D') and r.get('frontend', 'yamnet') == 'yamnet' \
        and not r.get('init') and not r.get('classes') and r.get('lam') in (None, 0.1)


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


def final_val(name):
    """curve.json's final V readout of a finished run, else None."""
    d = os.path.join(LOCAL, 'runs', name)
    if not os.path.exists(os.path.join(d, 'TRAIN_DONE')):
        return None
    return [x for x in json.load(open(os.path.join(d, 'curve.json')))['val'] if x.get('final')][-1]


def hitk_pct(v):
    """Teacher-hit at K: of the teacher's K buzz detections on V, the share among the student's K highest
    buzz scores (distill_train.hit_at_k). Blind to the student's calibration, unlike hit% at logit 0. None
    for a run that predates it (backfill_hitk.py)."""
    return None if v is None or 'buzz_hitk' not in v else 100 * v['buzz_hitk'] / max(1, v['buzz_teacher'])


def hitk_spread():
    """|difference| of hit@K between the two rung-A repeats (the noise floor --wsd-stop is set against)."""
    a = [hitk_pct(final_val(r['name'])) for r in rows() if is_ladder(r) and r['rung'] == 'A']
    a = [x for x in a if x is not None]
    return abs(a[0] - a[1]) if len(a) >= 2 else None


def record(a):
    cur = json.load(open(os.path.join(LOCAL, 'runs', a.name, 'curve.json')))
    v = [x for x in cur['val'] if x.get('final')][-1]
    head, per, tiers, incl = sx(os.path.join(LOCAL, 'eval', a.name, 'folds_sx.csv'))
    args = cur.get('args', {})
    row = {'rung': a.rung, 'seed': a.seed, 'steps': a.steps, 'name': a.name, 'arch': a.arch, 'loader': a.loader,
           'frontend': a.frontend, 'init': a.init, 'classes': a.classes, 'lam': a.lam,
           'schedule': args.get('schedule', 'cosine'), 'lr': args.get('lr'), 'warmup': args.get('warmup', 0),
           'decay_from': args.get('decay_from', ''),
           'passes': round(cur['passes'], 3) if 'passes' in cur else None,
           'val_frames': v['frames'], 'buzz_teacher_pos': v['buzz_teacher'], 'buzz_student_pos': v['buzz_student'],
           'buzz_gained': v['buzz_gained'], 'buzz_lost': v['buzz_lost'],
           'lost_pct': round(100 * v['buzz_lost'] / max(1, v['buzz_teacher']), 2),
           'hitk_pct': None if hitk_pct(v) is None else round(hitk_pct(v), 2),
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
        tag = f'{r["rung"]}/{r.get("arch", "a0.50")}/s{r["seed"]}' + ('/stream' if r.get('loader') == 'stream' else '') \
            + (f'/{r["frontend"]}' if r.get('frontend', 'yamnet') != 'yamnet' else '') + ('/' + r['init'] if r.get('init') else '') \
            + (f'/c{len(r["classes"].split(","))}' if r.get('classes') else '') + (f'/lam{r["lam"]:g}' if r.get('lam') not in (None, 0.1) else '') \
            + ('' if is_cosine(r) else '/' + r['schedule'])
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
           and r.get('loader', 'mem') == 'mem' and r['seed'] == 1 and is_cosine(r)][0]
    st_ = [r for r in rows() if r['rung'] == 'B' and r.get('arch', 'a0.50') == 'a0.50' and r.get('loader') == 'stream'][-1]
    d = abs(mem['lost_pct'] - st_['lost_pct'])
    ok = d <= sp
    print(f'[streamcheck] B mem lost% {mem["lost_pct"]} (headline {mem["headline"]}) vs stream {st_["lost_pct"]} '
          f'(headline {st_["headline"]}): diff {d:.2f} vs A spread {sp:.2f}: {"OK" if ok else "STREAM LOADER DOES NOT REPRODUCE"}')
    sys.exit(0 if ok else 1)


def frontier(a):
    """Speed / sensitivity of every rung-B run, any front end, sorted by 200 s speed.
    Speed is x YAMNet (GPU, same harness); sensitivity is the ladder headline and % of the baseline's."""
    base = sx(REFS[0][1])[0]
    rs = [r for r in rows() if r['rung'] == 'B' and r.get('loader', 'mem') == 'mem' and is_cosine(r)]
    rs.sort(key=lambda r: -(r.get('x_yamnet200') or 0))
    print(f'\nrung B, seed-1 runs by 200 s speed (baseline headline {base:.3f}; 50% of it = {base / 2:.3f})')
    print(f'{"run":44s} {"frontend":9s} {"arch":10s} {"init":7s} {"x YAM 20s":>9s} {"x YAM 200s":>10s} '
          f'{"headline":>8s} {"% base":>6s} {"incl-quiet":>10s} {"lost%":>6s}')
    for r in rs:
        print(f'{r["name"]:44s} {r.get("frontend", "yamnet"):9s} {r.get("arch", "a0.50"):10s} {r.get("init", "") or "-":7s} '
              f'{r.get("x_yamnet20", float("nan")):9.2f} {r.get("x_yamnet200", float("nan")):10.2f} '
              f'{r["headline"]:8.3f} {100 * r["headline"] / base:5.0f}% {r["headline_inclusive"]:10.3f} {r["lost_pct"]:6.1f}')


def wsd(a):
    """A WSD trunk's step-budget curve: one line per decayed branch `<trunk><B>` (held-out V readouts from
    curve.json, headline where it was evaluated), with the cosine run of the same student (`<trunk>` minus
    `_wsd`, plus --ref names) as reference. hit% = teacher-hit %: of the teacher's buzz detections on V, the
    share the student also makes at logit 0 (100 - lost%); hit@K = the same share among the student's K top
    scores (K = the teacher's count; blind to calibration); dK = hit@K's change from the previous branch (what
    --wsd-stop reads). Not the headline, which is sensitivity against labels on the folds."""
    import glob
    runs = os.path.join(LOCAL, 'runs')
    base = a.name[:-len('_wsd')] if a.name.endswith('_wsd') else a.name
    branches = sorted((p for p in glob.glob(os.path.join(runs, a.name + '*'))
                       if os.path.basename(p)[len(a.name):].isdigit()), key=lambda p: int(p.rsplit('_wsd', 1)[1]))
    print(f'\nstep-budget curve of {a.name} (V pool vs teacher; headline = {COL} @0.005 where evaluated)')
    print(f'{"run":58s} {"steps":>6s} {"passes":>6s} {"loss":>6s} {"mae":>6s} {"hit%":>6s} {"dhit":>6s} '
          f'{"hit@K":>6s} {"dK":>6s} {"gain%":>6s} {"headline":>8s}')
    prev = prevk = None
    refs = [base] + [r for r in (a.ref or '').split(',') if r]
    for i, p in enumerate([os.path.join(runs, r) for r in refs] + branches):
        n = os.path.basename(p)
        if not os.path.exists(os.path.join(p, 'TRAIN_DONE')):
            if i >= len(refs):
                print(f'{n:58s} (not finished)')
            continue
        cur = json.load(open(os.path.join(p, 'curve.json')))
        v = [x for x in cur['val'] if x.get('final')][-1]
        hit = 100 - 100 * v['buzz_lost'] / max(1, v['buzz_teacher'])
        loss = sum(t['loss'] for t in cur['train'][-5:]) / max(1, len(cur['train'][-5:]))
        ev = os.path.join(LOCAL, 'eval', n, 'folds_sx.csv')
        head = f'{sx(ev)[0]:8.3f}' if os.path.exists(ev) else f'{"-":>8s}'
        ref = i < len(refs)
        d = '' if ref or prev is None else f'{hit - prev:+6.2f}'
        hk = hitk_pct(v)
        dk = '' if ref or prevk is None or hk is None else f'{hk - prevk:+6.2f}'
        hks = f'{hk:6.2f}' if hk is not None else f'{"-":>6s}'
        sched = cur['args'].get('schedule', 'cosine')
        print(f'{(n + (" (" + sched + " ref)" if ref else "")):58s} {cur["args"]["steps"]:>6d} '
              f'{cur.get("passes", float("nan")):6.2f} {loss:6.3f} {v["mae_live"]:6.3f} {hit:6.2f} {d:>6s} '
              f'{hks} {dk:>6s} {100 * v["buzz_gained"] / max(1, v["buzz_teacher"]):6.2f} {head}')
        if not ref:
            prev, prevk = hit, hk
    sp, spk = a_spread(), hitk_spread()
    if sp is not None:
        print(f'noise: rung-A repeat spread of teacher-hit % = {sp:.2f}, of hit@K = '
              f'{"-" if spk is None else f"{spk:.2f}"}; headline seed noise ~0.01-0.02')


def spread(a):
    sp = hitk_spread()
    if sp is None:
        sys.exit('no hit@K on both rung-A repeats yet (backfill_hitk.py)')
    print(f'{sp:.2f}')


def proxy(a):
    """Pearson r of each V readout with the fold headline over every evaluated run (cosine and WSD), and for
    the buzz+rain+human students alone: which readout a stop rule should trust."""
    import numpy as np
    pts = []
    for r in rows():
        v = final_val(r['name'])
        if v is None or r.get('headline') is None:
            continue
        pts.append((r['name'], 'ambient_rain' in (r.get('classes') or ''), r['headline'], 100 - r['lost_pct'],
                    hitk_pct(v), -v['mae_live']))
    print(f'{"run":62s} {"headline":>8s} {"hit%":>6s} {"hit@K":>6s} {"mae":>6s}')
    for n, _, h, h0, hk, m in pts:
        print(f'{n:62s} {h:8.3f} {h0:6.2f} {"-" if hk is None else f"{hk:6.2f}":>6s} {-m:6.3f}')
    for tag, sub in (('all', pts), ('buzz+rain+human', [p for p in pts if p[1]])):
        full = [p for p in sub if p[4] is not None]
        if len(full) < 3:
            print(f'{tag}: {len(full)} runs with hit@K, too few for r')
            continue
        h = np.array([p[2] for p in full])
        rs = {k: np.corrcoef(h, [p[i] for p in full])[0, 1] for k, i in (('hit%', 3), ('hit@K', 4), ('-mae', 5))}
        print(f'{tag} ({len(full)} runs): r with headline  ' + '  '.join(f'{k} {x:+.3f}' for k, x in rs.items()))


def decide(a):
    """Which rung-D variant (if any) qualifies: within the A spread of a0.50 on lost% (rung B, seed 1)
    AND headline within HEADLINE_TOL of it; the faster (GPU, 200 s) if several."""
    sp = a_spread()
    b = [r for r in rows() if is_ladder(r) and r['rung'] == 'B' and r['seed'] == 1][0]
    ok = []
    for r in rows():
        if r['rung'] == 'B' and r.get('arch', 'a0.50') != 'a0.50' and r.get('loader', 'mem') == 'mem' and is_cosine(r):
            good = (r['lost_pct'] - b['lost_pct'] <= sp) and (r['headline'] >= b['headline'] - HEADLINE_TOL)
            print(f'[decide] {r["arch"]}: lost% {r["lost_pct"]} vs {b["lost_pct"]} (spread {sp:.2f}), headline '
                  f'{r["headline"]} vs {b["headline"]} (tol {HEADLINE_TOL}), gpu200 {r.get("gpu200")}: '
                  f'{"QUALIFIES" if good else "no"}', file=sys.stderr)
            if good:
                ok.append(r)
    print(max(ok, key=lambda r: r.get('gpu200', 0))['arch'] if ok else 'none')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['record', 'table', 'decide', 'gate', 'streamcheck', 'frontier', 'wsd', 'proxy', 'spread'])
    ap.add_argument('--ref', help='wsd: extra comma-separated run names to show as references (e.g. a 28k cosine run)')
    ap.add_argument('--rung'), ap.add_argument('--seed', type=int), ap.add_argument('--steps', type=int)
    ap.add_argument('--name'), ap.add_argument('--wall', type=float, default=0)
    ap.add_argument('--arch', default='a0.50'), ap.add_argument('--loader', default='mem')
    ap.add_argument('--frontend', default='yamnet'), ap.add_argument('--init', default='',
                                                                     help='non-default init tag, e.g. select (a control)')
    ap.add_argument('--classes', default='', help='comma-separated classes the student was distilled on ("" = all)')
    ap.add_argument('--lam', type=float, default=0.1, help='code-regression loss weight the run used')
    a = ap.parse_args()
    {'record': record, 'table': table, 'decide': decide, 'gate': gate,
     'streamcheck': streamcheck, 'frontier': frontier, 'wsd': wsd, 'proxy': proxy, 'spread': spread}[a.phase](a)
