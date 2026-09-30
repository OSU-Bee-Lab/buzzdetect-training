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

Per student (`--runs`: <frontend>:<arch>[:<init>][:classes=a+b+c][:lam=<x>] ...;
name fe_<rung>_<frontend>_<arch>_s<seed>[_<init>][_c-<classes>][_lam<x>]; `classes=` distils only those
teacher classes (the head has only those outputs; must include ins_buzz), `lam=0` drops the code-regression loss)
  train     distill_train.py         resumes from its last checkpoint; TRAIN_DONE marks the end
  export    export_student.py export ONNX with the teacher's centers folded in, parity-checked
  eval      eval_folds.py run        headline sensitivity_exclquiet @ fpr 0.005 on the rotating folds
  probe     eval_folds.py probe      1_95 (jet fold, a training fold) scored apart from the headline: threshold, sens, jet share of FPs
  speed     export_student.py time   x YAMNet on the GPU, 20 s and 200 s chunks
  record    ladder_record.py record  one row in 05_distill/data/<teacher>/ladder.jsonl
  deploy    deploy_student.py        only with --deploy: copy into buzzdetect (--force to overwrite)

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
    fe, arch, init, classes, lam = spec
    return (f'{A.prefix}_{A.rung}_{fe}_{arch}_s{A.seed}' + (f'_{init}' if init else '')
            + (f'_c-{short_classes(classes)}' if classes else '') + (f'_lam{lam}' if lam else ''))


def parse_runs():
    """'<frontend>:<arch>[:<init>][:classes=a+b+c][:lam=0]' -> (frontend, arch, init, 'a,b,c' or '', lam or '')."""
    out = []
    for r in A.runs.split():
        fe, arch, *rest = r.split(':')
        init = next((t for t in rest if '=' not in t), '')
        kv = dict(t.split('=', 1) for t in rest if '=' in t)
        unknown = set(kv) - {'classes', 'lam'}
        if unknown:
            sys.exit(f'run {r!r}: unknown option(s) {sorted(unknown)} (classes=a+b+c, lam=<float>)')
        out.append((fe, arch, init, kv.get('classes', '').replace('+', ','), kv.get('lam', '')))
    return out


class Stage:
    """One subprocess with a done-check; `artifact` is a path or a callable returning bool."""

    def __init__(self, key, label, cmd, done, env=None, wall=None):
        self.key, self.label, self.cmd, self.done, self.env, self.wall = key, label, cmd, done, env, wall

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


def stages_for_run(spec):
    fe, arch, init, classes, lam = spec
    name = run_name(spec)
    d = os.path.join(D.RUNS, name)
    m = os.path.join(D.MODELS, name)
    ev = os.path.join(D.EVAL, name)
    loader = 'stream' if A.rung == 'D' else 'mem'
    train = [TRAIN_PY, '-u', 'distill_train.py', '--rung', A.rung, '--steps', str(A.steps), '--batch', str(A.batch),
             '--seed', str(A.seed), '--name', name, '--arch', arch, '--frontend', fe, '--loader', loader,
             '--eval-every', str(A.eval_every)] + (['--init', init] if init else []) \
        + (['--classes', classes] if classes else []) + (['--lam', lam] if lam else [])
    env = None
    wall = os.path.join(d, 'wall.txt')
    rec = [TRAIN_PY, 'ladder_record.py', 'record', '--rung', A.rung, '--seed', str(A.seed), '--steps', str(A.steps),
           '--name', name, '--wall', 'WALL', '--arch', arch, '--frontend', fe, '--init', init, '--loader', loader,
           '--classes', classes, '--lam', lam or '0.1']
    st = [
        Stage('train', f'{name} train', train, os.path.join(d, 'TRAIN_DONE'), env, wall),
        Stage('export', f'{name} export', [TRAIN_PY, '-u', 'export_student.py', 'export', '--run', name, '--name', name],
              os.path.join(m, 'model.onnx'), None, wall),
        Stage('eval', f'{name} eval', [TRAIN_PY, '-u', 'eval_folds.py', 'run', '--check-labels',
                                        '--onnx', os.path.join(m, 'model.onnx'), '--out', ev],
              os.path.join(ev, 'folds_sx.csv'), None, wall),
        Stage('probe', f'{name} probe', [TRAIN_PY, '-u', 'eval_folds.py', 'probe',
                                          '--onnx', os.path.join(m, 'model.onnx'), '--out', ev],
              os.path.join(ev, 'probe', 'probe.json'), None, wall),
        Stage('speed', f'{name} speed', [D.ENGINE_PY, 'export_student.py', 'time', '--name', name, '--repeats', '15'],
              os.path.join(m, 'speed_200.json'), None, wall),
        Stage('record', f'{name} record', rec, lambda: in_ladder(name), None, wall),
        Stage('record', 'frontier table', [TRAIN_PY, 'ladder_record.py', 'frontier'], lambda: False),
    ]
    if A.deploy:
        st.append(Stage('deploy', f'{name} deploy', [TRAIN_PY, 'deploy_student.py', name] +
                        (['--force'] if A.force_deploy else []),
                        (lambda n=name: os.path.exists(os.path.join(D.ENGINE_MODELS, n, 'README.md')))
                        if not A.force_deploy else None))
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
    runs = parse_runs()
    tstages = stages_for_teacher(runs)
    plans = [stages_for_run(r) for r in runs]
    all_stages = tstages + [s for _, _, ss in plans for s in ss]
    if A.dry_run:
        print(f'teacher {D.TEACHER}; cache {D.CACHE}; local {D.LOCAL}')
        for s in all_stages:
            state = 'done' if s.is_done() else ('resumable' if s.done is None else 'pending')
            print(f'  [{state:9s}] {s.label}')
        return
    preflight(all_stages)
    os.makedirs(D.LOCAL, exist_ok=True)
    stop = len(all_stages) - 1
    if A.until:
        stop = max(i for i, s in enumerate(all_stages) if s.key == A.until) if any(
            s.key == A.until for s in all_stages) else stop
    for s in all_stages[:stop + 1]:
        if s.wall:
            os.makedirs(os.path.dirname(s.wall), exist_ok=True)
        execute(s)
    say(f'stopped after {A.until}' if A.until else 'all done')


if __name__ == '__main__':
    main()
