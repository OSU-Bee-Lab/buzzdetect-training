"""Re-export trained students with another graph (same weights), then judge them like any student.

    python 05_distill/reexport.py --fe-impl conv --suffix _cfe --runs <run> [<run> ...]

For each run already in the log: export `<run><suffix>` (export_student.py --fe-impl), score it on the
rotating folds, time it, and record a row that takes its V readouts from the source run (`ladder_record.py
record --run`). Every stage is skipped when its artifact exists, so a rerun resumes. Stage lines are
`[chain] ...`, as main.py's.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dpaths as D  # noqa: E402


def stage(label, cmd, done):
    if done():
        print(f'[chain] {label}: done already', flush=True)
        return
    print(f'[chain] {label}: start', flush=True)
    r = subprocess.run(cmd, cwd=HERE, env=dict(os.environ, PYTHONUNBUFFERED='1', DISTILL_TEACHER=D.TEACHER))
    if r.returncode or not done():
        sys.exit(f'[chain] {label}: FAILED exit {r.returncode}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--fe-impl', default='conv')
    ap.add_argument('--suffix', default='_cfe')
    a = ap.parse_args()
    rows = {r['name']: r for r in D.read_log()}
    for run in a.runs:
        src, name = rows[run], run + a.suffix
        m, ev = os.path.join(D.MODELS, name), os.path.join(D.EVAL, name)
        stage(f'{name} export', [sys.executable, '-u', 'export_student.py', 'export', '--run', run, '--name', name,
                                 '--fe-impl', a.fe_impl], lambda: os.path.exists(os.path.join(m, 'model.onnx')))
        stage(f'{name} eval', [sys.executable, '-u', 'eval_folds.py', 'run', '--check-labels',
                               '--onnx', os.path.join(m, 'model.onnx'), '--out', ev],
              lambda: os.path.exists(os.path.join(ev, 'folds_sx.csv')))
        stage(f'{name} speed', [D.ENGINE_PY, 'export_student.py', 'time', '--name', name, '--repeats', '15'],
              lambda: os.path.exists(os.path.join(m, 'speed_200.json')))
        stage(f'{name} record',
              [sys.executable, 'ladder_record.py', 'record', '--run', run, '--fe-impl', a.fe_impl, '--name', name,
               '--rung', src['rung'], '--seed', str(src['seed']), '--steps', str(src['steps']),
               '--wall', str(src.get('wall_s', 0)), '--arch', src['arch'], '--frontend', src.get('frontend', 'yamnet'),
               '--init', src.get('init') or '', '--loader', src.get('loader', 'mem'),
               '--classes', src.get('classes') or '', '--lam', str(src.get('lam', 0.1))],
              lambda: any(r.get('name') == name for r in D.read_log()))


if __name__ == '__main__':
    main()
