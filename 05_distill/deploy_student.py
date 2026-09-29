"""Carry an exported student to buzzdetect: one command, everything that matters.

    conda run -n buzzdetect-train python 05_distill/deploy_student.py fe_B_fast32_a0.50_s1
    ... deploy_student.py <name> [--as lite-fast32] [--dest DIR] [--force] [--dry-run]

Reads `.local/distill/models/<name>/` (export_student.py's model.onnx + config_model.json), the run's
eval (`.local/distill/eval/<name>/folds_sx.csv`) and record (`ladder.jsonl` row, `runs/<name>/curve.json`),
and writes `<dest>/<as or name>/`:

    model.onnx, config_model.json    the deployable pair
    folds_sx.csv                     the per-fold sensitivity table the headline came from
    README.md                        provenance card: teacher, architecture, front end, init, metrics, speed, caveats

`--dest` defaults to `buzzdetect_dest` in paths.local.json (the same place 04_deploy/export_onnx.py writes).
Refuses to touch an existing model dir unless `--force`; checks before copying that the ONNX loads, takes a
waveform and returns the config's 15 classes, and after copying that each file's sha256 matches the source.
This copies files only: it never commits or edits anything inside buzzdetect.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config  # noqa: E402

MAIN = ROOT.split('/.claude/worktrees/')[0]
LOCAL = os.path.join(MAIN, '.local', 'distill')
COPIED = ['model.onnx', 'config_model.json']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ladder_row(name):
    path = os.path.join(LOCAL, 'ladder.jsonl')
    rows = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
    rows = [r for r in rows if r.get('name') == name]
    return rows[-1] if rows else None


def smoke(onnx_path, cfg):
    """The engine's contract: input 'waveform', one row of len(classes) logits per 0.96 s patch."""
    import onnxruntime as ort
    s = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    assert s.get_inputs()[0].name == 'waveform', f'input is {s.get_inputs()[0].name!r}, engine expects waveform'
    n = max(cfg['samples_min'], 2 * cfg['samples_hop'])
    out = s.run(None, {'waveform': np.zeros(n, np.float32)})[0]
    assert out.ndim == 2 and out.shape[1] == len(cfg['classes']), \
        f'output {out.shape}, expected (frames, {len(cfg["classes"])})'
    assert np.isfinite(out).all(), 'non-finite logits on silence'
    return out.shape


def readme(name, as_name, cfg, curve, row):
    a = curve['args']
    md = cfg['metadata']
    fmt = lambda v: 'n/a' if v is None else f'{v:.3f}'
    lines = [f'# {as_name}', '',
             f'Distilled single-pass student of `{md["teacher"]}` (trained on the `{md["set"]}` set), exported {md["trained_date"]}.',
             'Source run: `%s`. **Not** a model trained on labels: it learns the teacher\'s logits and embeddings from '
             'a cache and never sees annotations or the eval deployments.' % name, '',
             '## What it is', '',
             f'- Architecture: MobileNetV1-style, YAMNet\'s layout at alpha {a["alpha"]} (`{a["arch"]}`), '
             f'filters {md["filters"]}, one pass.',
             f'- Spectrogram front end: `{md["frontend"]}`'
             + (' (YAMNet\'s own)' if md['frontend'] == 'yamnet' else ' (custom, see 05_distill/frontends.py)') + '.',
             f'- Init `{a["init"]}`; data rung {a["rung"]}; seed {a["seed"]}; {a["steps"]} steps, batch {a["batch"]}, lr {a["lr"]}.',
             f'- Head bias has the teacher\'s activation centers folded in: detect at logit > 0.', '']
    if row:
        lines += ['## Metrics', '',
                  f'Headline `sensitivity_exclquiet` at fpr 0.005, mean over the 5 rotating folds: **{row["headline"]:.3f}** '
                  f'(including quiet buzzes {row["headline_inclusive"]:.3f}). Per fold {row["per_fold"]}.',
                  'Reference points on the same folds: baseline `cv-baseline-v4-moderate` 0.414; teacher honest rotation 0.574; '
                  'teacher ONNX through the same harness 0.692 (trained on those folds, inflated).', '',
                  'Per loudness tier: ' + ', '.join(f'{k} {fmt(v)}' for k, v in row['tiers'].items()) + '.',
                  f'Buzz detections vs the teacher on the held-out V pool: lost {row["lost_pct"]:.1f}%, gained {row["gained_pct"]:.1f}%.', '']
        if row.get('x_yamnet200'):
            lines += [f'Speed (GPU, GTX 1650, buzzdetect engine session): {row["x_yamnet200"]:.2f}x YAMNet at 200 s chunks, '
                      f'{row["x_yamnet20"]:.2f}x at 20 s.', '']
    lines += ['## Caveats', '',
              '- `center_stats` in `config_model.json` are the **teacher\'s** thresholds and precision targets, not recalibrated for this student.',
              '- The headline is the student\'s own ONNX through 05_distill/eval_folds.py, not the full pipeline; one seed, so read gaps below ~0.03 as ties.',
              '- Students inherit fold knowledge through the teacher: a high headline is not "better than the teacher".',
              '- Speed is timed on one GPU (GTX 1650) through the engine session; CPU timings on the i7-2600 (no AVX2) are not representative.', '',
              'Provenance: `.local/distill/runs/%s/curve.json`, `ladder.jsonl` in the training project; folds_sx.csv beside this file.' % name]
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('name')
    ap.add_argument('--as', dest='as_name', help='directory name in buzzdetect (default: the run name)')
    ap.add_argument('--dest', default=config.local('buzzdetect_dest'))
    ap.add_argument('--force', action='store_true', help='overwrite an existing model dir\'s files')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if not a.dest:
        sys.exit('no --dest and no buzzdetect_dest in paths.local.json')

    src = os.path.join(LOCAL, 'models', a.name)
    for f in COPIED:
        if not os.path.exists(os.path.join(src, f)):
            sys.exit(f'{src}/{f} missing: run export_student.py export --run {a.name} --name {a.name} first')
    folds = os.path.join(LOCAL, 'eval', a.name, 'folds_sx.csv')
    curve_path = os.path.join(LOCAL, 'runs', a.name, 'curve.json')
    for p in (folds, curve_path):
        if not os.path.exists(p):
            sys.exit(f'{p} missing (eval_folds.py run / distill_train.py not finished for {a.name}?)')
    cfg = json.load(open(os.path.join(src, 'config_model.json')))
    curve = json.load(open(curve_path))
    row = ladder_row(a.name)
    if row is None:
        print(f'note: no ladder.jsonl row for {a.name}; README will omit metrics (run ladder_record.py record)')

    out = os.path.join(a.dest, a.as_name or a.name)
    if os.path.exists(out) and not a.force:
        if a.dry_run:
            print(f'dry run: {out} exists, a real run would refuse without --force')
            return
        sys.exit(f'{out} exists; --force overwrites its model.onnx, config_model.json, folds_sx.csv and README.md')

    shape = smoke(os.path.join(src, 'model.onnx'), cfg)
    print(f'smoke ok: silence -> logits {shape}')
    if a.dry_run:
        print(f'dry run: would write {out}')
        return

    os.makedirs(out, exist_ok=True)
    for f, s in [('model.onnx', os.path.join(src, 'model.onnx')),
                 ('config_model.json', os.path.join(src, 'config_model.json')),
                 ('folds_sx.csv', folds)]:
        shutil.copy2(s, os.path.join(out, f))
        assert sha(s) == sha(os.path.join(out, f)), f'checksum mismatch on {f}'
    with open(os.path.join(out, 'README.md'), 'w') as f:
        f.write(readme(a.name, a.as_name or a.name, cfg, curve, row))
    print(f'wrote {out}: ' + ', '.join(sorted(os.listdir(out))))


if __name__ == '__main__':
    main()
