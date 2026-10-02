"""Stage 5: distil a teacher into lite students, end to end and resumable.

    tools/launch_job.sh <log> -- conda run -n buzzdetect-train python 05_distill/main.py \\
        --teacher v4-ft-ps-e60-moderate --rung B --runs "yamnet:a0.50:select fast32:a0.50 fast32h16:a0.375"

    ... main.py --runs fast32:a0.50 --dry-run     # what is done, what would run, nothing executed

Run it in the train env (the interpreter that runs main.py trains, exports and scores); the
onnxruntime steps run in `.local/venv-onnx` and the speed steps in buzzdetect's engine venv
(`distill_onnx_python` in paths.local.json overrides the first; the second is
`<buzzdetect_dest>/../.venv`). Everything is a subprocess of the stage script named below, so
each stage can also be run by hand with `DISTILL_TEACHER=<teacher>`.

Teacher stages (once per teacher, shared by every student)
  frontend  bench_arch.py export --candidates frontend_only    YAMNet mel graph (teacher-independent)
  plan      plan.py                  slices, rungs, blacklist (the teacher's own training deployments)
  teacher   teacher_onnx.py          teacher + `code` output, verified on the fixture
  cache     cache.py V, <rung>       teacher targets (+ the shared YAMNet mel)
  cache_fe  cache_fe.py V, <rung>    student-input spectrograms for the non-YAMNet front ends, all at once
  pack      shards.py pack --rung D  only for rung D (streamed from shards)

Per student (`--runs`: <frontend>:<arch>[:<init>][:classes=a+b+c][:lam=<x>][:lr=<x>] ...;
name fe_<rung>_<frontend>_<arch>_s<seed>[_<init>][_c-<classes>][_lam<x>][_lr<x>]; `classes=` distils only those
teacher classes (the head has only those outputs; must include ins_buzz), `lam=0` drops the code-regression loss)
  train     distill_train.py         resumes from its last checkpoint; TRAIN_DONE marks the end
  export    export_student.py export ONNX with the teacher's centers folded in, parity-checked
  eval      eval_folds.py run        headline sensitivity_exclquiet @ fpr 0.005 on the rotating folds
  probe     eval_folds.py probe      1_95 (jet fold, a training fold) scored apart from the headline: threshold, sens, jet share of FPs
  speed     export_student.py time   x YAMNet on the GPU, 20 s and 200 s chunks
  record    ladder_record.py record  one row in 05_distill/data/<teacher>/ladder.jsonl
  deploy    deploy_student.py        only with --deploy: copy into buzzdetect (--force to overwrite)

Step-budget curve (`--wsd-max 56000`: the ceiling and --wsd-halvings (3) halvings of it, 7000-56000; or a list,
`--wsd 7000,14000,28000,70000`, budgets also in passes over the rung: `--wsd 0.5p,1p,2p`): each
student becomes one warmup-stable-decay trunk `<name>_wsd` and one decayed branch per budget B,
`<name>_wsd<B>`, an ordinary run of B total steps (stable to (1 - --wsd-decay) B, then decay). Stages
interleave: trunk to the first branch point, that branch (train, export, eval, ...), trunk on to the next.
`--wsd-eval` limits the costly stages (export onwards) to some budgets; `--wsd-stop <points>` ends the trunk
once a branch raises teacher-hit % (share of the teacher's V buzz detections the student also makes) on the
previous branch by less than that. `ladder_record.py wsd` prints
the curve. Budgets resolve to steps via the plan's frame count for the rung (`--passes` does the same for
the cosine `--steps`).

Every stage is skipped when its artifact exists, so re-running the same command after a crash or a
quit picks up where it stopped (mid-training included). A stage that fails stops the whole run.
Sharing: `_mel/<spec>` (spectrograms), `_shared/durations.csv` and the plan's ranks are teacher-
independent; targets, shards, runs and models are per teacher. See README.md.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--teacher', help='teacher model name (default: DISTILL_TEACHER / paths.local.json / v4-ft-ps-e60-moderate)')
    ap.add_argument('--runs', default='yamnet:a0.50', help='space-separated <frontend>:<arch>[:<init>]')
    ap.add_argument('--rung', default='B', choices=list('ABCD'))
    ap.add_argument('--steps', type=int, default=7000)
    ap.add_argument('--passes', type=float, help='cosine budget in passes over the rung instead of --steps')
    ap.add_argument('--wsd', default='', help='warmup-stable-decay budgets, comma-separated steps or passes (2p)')
    ap.add_argument('--wsd-max', default='', help='WSD ceiling (steps or passes, 2p); budgets are it and its halvings')
    ap.add_argument('--wsd-halvings', type=int, default=3, help='--wsd-max: how many halvings below the ceiling '
                                                                '(3: 56000 -> 7000,14000,28000,56000)')
    ap.add_argument('--wsd-decay', type=float, default=0.15, help='decay share of each WSD budget')
    ap.add_argument('--warmup', type=int, default=300, help='WSD linear warmup steps')
    ap.add_argument('--wsd-eval', default='', help='budgets (as given to --wsd) that get export/eval/probe/speed/record; '
                                                   'default all')
    ap.add_argument('--wsd-stop', type=float, help='end the trunk once a branch raises teacher-hit %% (V buzz detections '
                                                   'the student shares with the teacher) by less than this on the '
                                                   'previous one (1.3 = the rung-A repeat spread)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--prefix', default='fe', help='run-name prefix')
    ap.add_argument('--until', choices=['frontend', 'plan', 'teacher', 'cache', 'cache_fe', 'pack', 'train',
                                        'export', 'eval', 'probe', 'speed', 'record', 'deploy'],
                    help='stop after this stage (for the last run)')
    ap.add_argument('--deploy', action='store_true', help='copy each finished student into buzzdetect')
    ap.add_argument('--force-deploy', action='store_true', help='overwrite an existing deployed model dir')
    ap.add_argument('--dry-run', action='store_true')
    return ap.parse_args()


A = parse()
if A.teacher:
    os.environ['DISTILL_TEACHER'] = A.teacher
sys.path.insert(0, HERE)
import dpaths as D  # noqa: E402  (after DISTILL_TEACHER is set)

TRAIN_PY = sys.executable
ORDER = ['frontend', 'plan', 'teacher', 'cache', 'cache_fe', 'pack', 'train', 'export', 'eval', 'probe', 'speed', 'record', 'deploy']


def say(msg):
    print(f'[chain] {msg} {time.strftime("%T")}', flush=True)


def short_classes(classes):
    """ins_buzz+ambient_rain+human -> buzz-rain-human (the part after the first underscore)."""
    return '-'.join(c.split('_', 1)[-1] for c in classes.split(','))


def run_name(spec):
    fe, arch, init, classes, lam, lr = spec
    return (f'{A.prefix}_{A.rung}_{fe}_{arch}_s{A.seed}' + (f'_{init}' if init else '')
            + (f'_c-{short_classes(classes)}' if classes else '') + (f'_lam{lam}' if lam else '')
            + (f'_lr{lr}' if lr else ''))


def parse_runs():
    """'<frontend>:<arch>[:<init>][:classes=a+b+c][:lam=0][:lr=2e-3]'
    -> (frontend, arch, init, 'a,b,c' or '', lam or '', lr or '')."""
    out = []
    for r in A.runs.split():
        fe, arch, *rest = r.split(':')
        init = next((t for t in rest if '=' not in t), '')
        kv = dict(t.split('=', 1) for t in rest if '=' in t)
        unknown = set(kv) - {'classes', 'lam', 'lr'}
        if unknown:
            sys.exit(f'run {r!r}: unknown option(s) {sorted(unknown)} (classes=a+b+c, lam=<float>, lr=<float>)')
        out.append((fe, arch, init, kv.get('classes', '').replace('+', ','), kv.get('lam', ''), kv.get('lr', '')))
    return out


def rung_frames():
    """Frames in the rung by the plan (nominal: the pack may lack a few slices; training prints the real passes)."""
    import csv
    if not os.path.exists(D.PLAN):
        sys.exit('budgets in passes need the plan: run main.py --until plan first')
    rungs = 'ABCD'[:'ABCD'.index(A.rung) + 1]
    return sum(int(r['n_frames']) for r in csv.DictReader(open(D.PLAN)) if r['first_rung'] in rungs)


def to_steps(tok):
    """'7000' -> 7000; '2p' / '2.5p' -> that many passes over the rung at --batch."""
    tok = str(tok).strip()
    if tok.endswith('p'):
        return max(1, round(float(tok[:-1]) * rung_frames() / A.batch))
    return int(tok)


def budgets():
    """WSD budgets in steps: --wsd's list, or --wsd-max and --wsd-halvings halvings of it."""
    if A.wsd_max:
        top = to_steps(A.wsd_max)
        return [round(top / 2 ** i) for i in range(A.wsd_halvings + 1)]
    return [to_steps(t) for t in A.wsd.split(',')]


def wsd_points():
    """[(budget B, branch step s)], ascending: the trunk is kept at s, the branch decays from s to B."""
    pts = sorted({(b, round(b * (1 - A.wsd_decay))) for b in budgets()})
    ss = [s for _, s in pts]
    if not 0 < A.wsd_decay < 1 or ss[0] <= 0 or len(set(ss)) != len(ss) or any(s >= b for b, s in pts):
        sys.exit(f'--wsd {A.wsd or A.wsd_max} / --wsd-decay {A.wsd_decay}: branch points {pts} must be distinct and inside each budget')
    return pts


def branch_hit_pct(name):
    """Teacher-hit % of a finished run: of the teacher's buzz detections on V, the share the student also
    makes (100 - lost%; curve.json's final entry), else None."""
    d = os.path.join(D.RUNS, name)
    if not os.path.exists(os.path.join(d, 'TRAIN_DONE')):
        return None
    v = [x for x in json.load(open(os.path.join(d, 'curve.json')))['val'] if x.get('final')][-1]
    return 100 - 100 * v['buzz_lost'] / max(1, v['buzz_teacher'])


def plateaued(earlier):
    """--wsd-stop: some branch among `earlier` (ascending budgets) raised teacher-hit % on its predecessor by less
    than the tolerance, so the trunk is not taken further."""
    if A.wsd_stop is None:
        return False
    hp = [branch_hit_pct(n) for n in earlier]
    return any(p is not None and c is not None and c - p < A.wsd_stop for p, c in zip(hp, hp[1:]))


class Stage:
    """One subprocess with a done-check; `artifact` is a path or a callable returning bool. `skip` (callable)
    is checked right before running: True leaves the stage out (a WSD plateau)."""

    def __init__(self, key, label, cmd, done, env=None, wall=None, skip=None):
        self.key, self.label, self.cmd, self.done, self.env, self.wall, self.skip = key, label, cmd, done, env, wall, skip

    def is_done(self):
        return self.done() if callable(self.done) else (self.done is not None and os.path.exists(self.done))


def stages_for_teacher(runs):
    fes = sorted({r[0] for r in runs if r[0] != 'yamnet'})
    st = [
        Stage('frontend', 'frontend_only.onnx', [TRAIN_PY, 'bench_arch.py', 'export', '--candidates', 'frontend_only'],
              D.FRONTEND_ONNX),
        Stage('plan', 'plan', [D.ONNX_PY, 'plan.py'], D.PLAN),
        Stage('teacher', 'teacher_onnx', [D.ONNX_PY, 'teacher_onnx.py'], teacher_current),
        Stage('cache', 'cache V', [D.ONNX_PY, 'cache.py', '--rung', 'V'], None),
        Stage('cache', f'cache {A.rung}', [D.ONNX_PY, 'cache.py', '--rung', A.rung], None),
    ]
    if fes:
        for rung in ('V', A.rung):
            st.append(Stage('cache_fe', f'cache_fe {rung} {",".join(fes)}',
                            [D.ONNX_PY, 'cache_fe.py', '--rung', rung, '--frontends', ','.join(fes)], None))
    if A.rung == 'D':
        st.append(Stage('pack', 'pack shards D', [TRAIN_PY, '-u', 'shards.py', 'pack', '--rung', 'D'], None))
    return st


def teacher_current():
    """teacher.json exists, matches the teacher ONNX on disk (or an identical re-export) and records code_dim."""
    try:
        import store
        meta = json.load(open(D.TEACHER_JSON))
        return meta.get('onnx_sha256') == store.teacher_sha() and 'code_dim' in meta and os.path.exists(D.TEACHER_EXT)
    except (OSError, ValueError, KeyError):
        return False


def in_ladder(name):
    try:
        return any(json.loads(l).get('name') == name for l in open(D.LADDER))
    except OSError:
        return False


def train_cmd(spec, name, steps, extra=()):
    fe, arch, init, classes, lam, lr = spec
    loader = 'stream' if A.rung == 'D' else 'mem'
    return [TRAIN_PY, '-u', 'distill_train.py', '--rung', A.rung, '--steps', str(steps), '--batch', str(A.batch),
            '--seed', str(A.seed), '--name', name, '--arch', arch, '--frontend', fe, '--loader', loader,
            '--eval-every', str(A.eval_every)] + (['--init', init] if init else []) \
        + (['--classes', classes] if classes else []) + (['--lam', lam] if lam else []) \
        + (['--lr', lr] if lr else []) + [str(x) for x in extra]


def judge_stages(spec, name, steps, skip=None):
    """Everything after training for one finished run: export, eval, probe, speed, record (+ deploy)."""
    fe, arch, init, classes, lam, lr = spec
    m = os.path.join(D.MODELS, name)
    ev = os.path.join(D.EVAL, name)
    wall = os.path.join(D.RUNS, name, 'wall.txt')
    loader = 'stream' if A.rung == 'D' else 'mem'
    rec = [TRAIN_PY, 'ladder_record.py', 'record', '--rung', A.rung, '--seed', str(A.seed), '--steps', str(steps),
           '--name', name, '--wall', 'WALL', '--arch', arch, '--frontend', fe, '--init', init, '--loader', loader,
           '--classes', classes, '--lam', lam or '0.1']
    st = [
        Stage('export', f'{name} export', [TRAIN_PY, '-u', 'export_student.py', 'export', '--run', name, '--name', name],
              os.path.join(m, 'model.onnx'), None, wall, skip),
        Stage('eval', f'{name} eval', [TRAIN_PY, '-u', 'eval_folds.py', 'run', '--check-labels',
                                        '--onnx', os.path.join(m, 'model.onnx'), '--out', ev],
              os.path.join(ev, 'folds_sx.csv'), None, wall, skip),
        Stage('probe', f'{name} probe', [TRAIN_PY, '-u', 'eval_folds.py', 'probe',
                                          '--onnx', os.path.join(m, 'model.onnx'), '--out', ev],
              os.path.join(ev, 'probe', 'probe.json'), None, wall, skip),
        Stage('speed', f'{name} speed', [D.ENGINE_PY, 'export_student.py', 'time', '--name', name, '--repeats', '15'],
              os.path.join(m, 'speed_200.json'), None, wall, skip),
        Stage('record', f'{name} record', rec, lambda: in_ladder(name), None, wall, skip),
    ]
    if A.deploy:
        st.append(Stage('deploy', f'{name} deploy', [TRAIN_PY, 'deploy_student.py', name] +
                        (['--force'] if A.force_deploy else []),
                        (lambda n=name: os.path.exists(os.path.join(D.ENGINE_MODELS, n, 'README.md')))
                        if not A.force_deploy else None, None, None, skip))
    return st


def wsd_stages(spec, name):
    """Trunk `<name>_wsd` in segments, one decayed branch `<name>_wsd<B>` (judged if in --wsd-eval) after each."""
    trunk = f'{name}_wsd'
    pts = wsd_points()
    judged = {to_steps(t) for t in A.wsd_eval.split(',') if t} if A.wsd_eval else {b for b, _ in pts}
    s_max = pts[-1][1]
    trunk_args = ['--schedule', 'wsd', '--warmup', A.warmup, '--branch-at', ','.join(str(s) for _, s in pts)]
    trunk_dir = os.path.join(D.RUNS, trunk)
    st, done = [], []
    for b, s in pts:
        skip = (lambda earlier=tuple(done): plateaued(earlier))
        bname = f'{name}_wsd{b}'
        st.append(Stage('train', f'{trunk} trunk to {s}',
                        train_cmd(spec, trunk, s_max, trunk_args + (['--stop-at', s] if s != s_max else [])),
                        os.path.join(trunk_dir, 'branches', f's{s}.npz'), None, os.path.join(trunk_dir, 'wall.txt'), skip))
        st.append(Stage('train', f'{bname} train (decay {s}-{b})',
                        train_cmd(spec, bname, b, ['--schedule', 'wsd', '--warmup', A.warmup, '--decay-from', f'{trunk}:{s}']),
                        os.path.join(D.RUNS, bname, 'TRAIN_DONE'), None, os.path.join(D.RUNS, bname, 'wall.txt'), skip))
        if b in judged:
            st += judge_stages(spec, bname, b, skip)
        done.append(bname)
    st.append(Stage('record', f'{trunk} curve', [TRAIN_PY, 'ladder_record.py', 'wsd', '--name', trunk], lambda: False))
    return trunk, trunk_dir, st


def stages_for_run(spec):
    name = run_name(spec)
    if A.wsd or A.wsd_max:
        return wsd_stages(spec, name)
    d = os.path.join(D.RUNS, name)
    st = [Stage('train', f'{name} train', train_cmd(spec, name, A.steps), os.path.join(d, 'TRAIN_DONE'), None,
                os.path.join(d, 'wall.txt'))]
    st += judge_stages(spec, name, A.steps)
    st.append(Stage('record', 'frontier table', [TRAIN_PY, 'ladder_record.py', 'frontier'], lambda: False))
    return name, d, st


def add_wall(path, t0):
    prev = int(open(path).read()) if os.path.exists(path) else 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').write(str(prev + int(time.time() - t0)))


def execute(stage):
    """Run one stage. Returns False when it was skipped as done."""
    if stage.is_done():
        say(f'{stage.label}: done (skip)')
        return False
    say(f'{stage.label}: start')
    env = dict(os.environ, PYTHONUNBUFFERED='1', DISTILL_TEACHER=D.TEACHER, **(stage.env or {}))
    cmd = [str(c) for c in stage.cmd]
    if 'WALL' in cmd:
        cmd[cmd.index('WALL')] = open(stage.wall).read() if stage.wall and os.path.exists(stage.wall) else '0'
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HERE, env=env)
    if r.returncode != 0:
        say(f'{stage.label}: FAILED exit {r.returncode}')
        sys.exit(r.returncode)
    if stage.wall:
        add_wall(stage.wall, t0)
    say(f'{stage.label}: done')
    return True


def preflight(all_stages):
    need = {D.ONNX_PY: 'onnx venv (.local/venv-onnx, or distill_onnx_python in paths.local.json)',
            D.ENGINE_PY: 'the buzzdetect engine venv (buzzdetect_dest in paths.local.json)'}
    for k, why in need.items():
        if any(k in s.cmd for s in all_stages if not s.is_done()) and not (k and os.path.exists(k)):
            sys.exit(f'missing interpreter {k}: {why}')
    D.need_cache()
    if not os.path.isfile(os.path.join(D.TEACHER_MODEL_DIR, 'config_model.json')):
        sys.exit(f'teacher {D.TEACHER!r} is not a model dir under models/')


def main():
    if A.wsd and A.wsd_max:
        sys.exit('--wsd lists the budgets, --wsd-max generates them: give one')
    if A.passes is not None:
        if A.wsd or A.wsd_max:
            sys.exit('--passes sets the cosine budget; give WSD budgets in passes as --wsd 1p,2p,...')
        A.steps = to_steps(f'{A.passes}p')
        say(f'--passes {A.passes} = {A.steps} steps at batch {A.batch} (plan frame count, rung {A.rung})')
    runs = parse_runs()
    tstages = stages_for_teacher(runs)
    plans = [stages_for_run(r) for r in runs]
    all_stages = tstages + [s for _, _, ss in plans for s in ss]
    if A.dry_run:
        print(f'teacher {D.TEACHER}; cache {D.CACHE}; local {D.LOCAL}')
        for s in all_stages:
            state = 'done' if s.is_done() else ('plateau' if s.skip and s.skip() else
                                                 'resumable' if s.done is None else 'pending')
            print(f'  [{state:9s}] {s.label}')
        return
    preflight(all_stages)
    os.makedirs(D.LOCAL, exist_ok=True)
    stop = len(all_stages) - 1
    if A.until:
        stop = max(i for i, s in enumerate(all_stages) if s.key == A.until) if any(
            s.key == A.until for s in all_stages) else stop
    for s in all_stages[:stop + 1]:
        if s.skip and not s.is_done() and s.skip():
            say(f'{s.label}: skipped (WSD plateau, --wsd-stop {A.wsd_stop})')
            continue
        if s.wall:
            os.makedirs(os.path.dirname(s.wall), exist_ok=True)
        execute(s)
    say(f'stopped after {A.until}' if A.until else 'all done')


if __name__ == '__main__':
    main()
