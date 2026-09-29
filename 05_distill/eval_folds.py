"""Score a waveform->logits ONNX on the rotating eval folds, as the pipeline does.

The ladder is judged on this number: `sensitivity_exclquiet` at fpr 0.005, the
plain mean over the 5 rotating folds, each fold's threshold set on its own
negatives (03_train/sx.py). This script reproduces what
`03_train/train.py::_score_fold` feeds sx, with an ONNX in place of the probe:

  * the fold's samples are the pickles under
    02_set/sets/moderate/audio/sr16000_fl0.96/raw/<fold>/ (the framed-audio
    cache; each pickle is one annotation snip, its elements the 0.96 s frames at
    framehop 0.2, its filename the raw labels joined by '+'). This is the exact
    audio the pipeline embedded, frame i == embedding row i.
  * labels: dataset.build_fold_dataset's rules (translation `general`: 'exclude'
    drops the snip, no translated class drops it, target = 'ins_buzz' among the
    translated labels), loudness tier = train_utils.buzz_tier. Re-implemented
    here, verified against a pipeline run's predictions.csv (`--check-labels`).
  * each frame goes through the ONNX ALONE (15360 samples -> 1 row), as the
    embedder saw it during extraction. Not a continuous stream: frames overlap
    (hop 0.192 s) and a stream would change the last two mel rows.
  * predictions.csv (activation_ins_buzz, correct, loudness, sample) per fold,
    then sx.summarize_folds. Same code path as folds_sx.csv.

Two phases, since ORT-GPU lives in buzzdetect's engine venv and sx needs TF:

  infer   engine venv:  <engine>/.venv/bin/python3 05_distill/eval_folds.py infer --onnx M --out D
  score   train env:    conda run -n buzzdetect-train python 05_distill/eval_folds.py score --out D
  run     train env, does both (infer through a subprocess of the engine venv)

`--folds` overrides the rotating folds (dev: one fold, or `--max-frames`).
Also writes D/logits.npy (all 15 logits per frame) and D/frames.csv (fold,
sample, frame idx) for flip/parity readouts.
"""
import argparse
import glob
import json
import os
import pickle
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import dpaths as D  # noqa: E402

MAIN = D.MAIN
SP = D.spec()
SET = SP.set                    # the teacher's set: its rotating folds are the eval deployments
TRANSLATION = SP.translation
AUDIO = SP.audio_dir
BUZZ = SP.buzz
ENGINE_PY = D.ENGINE_PY
BASELINE = os.path.join(MAIN, 'models', D.BASELINE)
TEACHER_ONNX = D.TEACHER_ONNX

# train_utils.TIER_MARKERS / TIERS, copied so infer runs without TensorFlow.
# `score --check-labels` proves the copy agrees with the pipeline's own labels.
TIER_MARKERS = (('_faint', 'faint'), ('_quiet', 'quiet'), ('_background', 'background'),
                ('_normal', 'normal'), ('_loud', 'loud'))
TIERS = ('faint', 'quiet', 'background', 'untagged', 'normal', 'loud')
RANK = {t: i for i, t in enumerate(TIERS)}


def label_tier(label):
    low = str(label).lower()
    hits = [t for m, t in TIER_MARKERS if m in low]
    return max(hits, key=RANK.__getitem__) if hits else 'untagged'


def buzz_tier(raw, translated):
    tiers = [label_tier(r) for r, t in zip(raw, translated) if t == 'ins_buzz']
    return max(tiers, key=RANK.__getitem__) if tiers else ''


def rotate_folds():
    df = pd.read_csv(os.path.join(MAIN, '02_set', 'sets', SET, 'folds.csv'), dtype=str)
    return sorted(df.loc[df['role'] == 'rotate', 'fold'].unique())


def fold_samples(fold):
    """(path, raw labels, translated labels, is_buzz, tier) per kept sample, in the
    pipeline's order (glob.glob over the fold dir, as dataset.build_fold_dataset)."""
    tr = pd.read_csv(os.path.join(MAIN, '02_set', 'sets', SET, 'translations', TRANSLATION + '.csv'))
    tdict = dict(zip(tr['from'], tr['to']))
    classes = [c for c in tr['to'].unique().tolist()
               if isinstance(c, str) and c.strip().lower() not in ('', 'ignore', 'exclude')]
    out = []
    for p in glob.glob(os.path.join(AUDIO, fold, '**', '*.pickle'), recursive=True):
        raw = re.split(r'\+', os.path.splitext(os.path.basename(p))[0])
        trn = [tdict.get(l, l) for l in raw]
        if any(isinstance(l, str) and l.strip().lower() == 'exclude' for l in trn):
            continue
        if not any(c in trn for c in classes):
            continue
        out.append((p, raw, trn, 'ins_buzz' in trn, buzz_tier(raw, trn)))
    return out


def read_frames(path):
    els = []
    with open(path, 'rb') as f:
        while True:
            try:
                els.append(np.asarray(pickle.load(f), dtype=np.float32))
            except EOFError:
                break
    return els


def make_session(onnx_path, cpu):
    import onnxruntime as ort
    prov = ['CPUExecutionProvider'] if cpu else ['CUDAExecutionProvider', 'CPUExecutionProvider']
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(onnx_path, so, providers=prov)


def infer(a):
    folds = a.folds or rotate_folds()
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(a.onnx)), 'config_model.json')
    min_samples, buzz = 0, BUZZ
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        min_samples = int(cfg.get('samples_min', 0))
        if 'ins_buzz' in cfg.get('classes', []):      # a class-subset student's buzz output is not column 8
            buzz = cfg['classes'].index('ins_buzz')
    sess = make_session(a.onnx, a.cpu)
    key = sess.get_inputs()[0].name
    print(f'[infer] {a.onnx} on {sess.get_providers()[0]}, {len(folds)} folds', flush=True)
    all_logits, index = [], []
    t0 = time.time()
    zero = np.zeros(15360, np.float32)

    def run_group(frs):
        """Logits for frames scored as if alone. Packed: frame, silent frame, frame, ...
        The silent frame is what an isolated frame's zero padding is, and the graph's hop is
        welded to 15360 so frames land on hop boundaries; row 2i is frame i.
        (`--isolated` runs one call per frame; the two are compared in the README table.)"""
        if a.isolated:
            outs = []
            for fr in frs:
                x = fr if len(fr) >= min_samples else np.pad(fr, (0, min_samples - len(fr)))
                y = sess.run(None, {key: x})[0]
                if len(y) == 0:
                    y = sess.run(None, {key: np.pad(fr, (0, max(0, 30719 - len(fr))))})[0]
                outs.append(y[0])
            return np.stack(outs)
        parts = []
        for i, fr in enumerate(frs):
            parts += [fr, zero] if i < len(frs) - 1 else [fr]
        x = np.concatenate(parts)
        if len(x) < min_samples:
            x = np.pad(x, (0, min_samples - len(x)))
        y = sess.run(None, {key: x})[0]
        assert len(y) >= 2 * len(frs) - 1, (len(y), len(frs))
        return y[0:2 * len(frs) - 1:2]

    for fold in folds:
        samples = fold_samples(fold)
        rows, group, n0 = [], [], len(all_logits)
        n = 0

        def flush():
            if group:
                all_logits.extend(run_group(group).astype(np.float32))
                group.clear()

        for sid, (path, raw, trn, buzz, tier) in enumerate(samples):
            for fi, fr in enumerate(read_frames(path)):
                if a.max_frames and n >= a.max_frames:
                    break
                group.append(fr)
                rows.append((sid, buzz, tier))
                index.append((fold, sid, fi))
                n += 1
                if len(group) == a.pack:
                    flush()
        flush()
        df = pd.DataFrame(rows, columns=['sample', 'correct', 'loudness'])
        df['activation_ins_buzz'] = [l[buzz] for l in all_logits[n0:]]
        d = os.path.join(a.out, 'folds', fold)
        os.makedirs(d, exist_ok=True)
        df[['activation_ins_buzz', 'correct', 'loudness', 'sample']].to_csv(
            os.path.join(d, 'predictions.csv'), index=False)
        print(f'[infer] {fold}: {n} frames in {len(samples)} samples, {time.time() - t0:.0f} s', flush=True)
    np.save(os.path.join(a.out, 'logits.npy'), np.stack(all_logits))
    pd.DataFrame(index, columns=['fold', 'sample', 'frame']).to_csv(
        os.path.join(a.out, 'frames.csv'), index=False)
    json.dump({'onnx': os.path.abspath(a.onnx), 'mode': 'isolated' if a.isolated else f'packed x{a.pack}', 'set': SET,
               'translation': TRANSLATION}, open(os.path.join(a.out, 'eval.json'), 'w'))


def score(a):
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, '03_train'))
    import tensorflow  # noqa: F401  (train_utils imports it; must precede pandas users)
    if not os.path.isdir(BASELINE):
        a.check_labels = False      # the label cross-check needs the baseline CV run's predictions
        print(f'[labels] no {BASELINE}: skipping --check-labels')
    from sx import summarize_folds, format_sx_report, read_fold_predictions, SENS_EXCL

    dfp = read_fold_predictions(os.path.join(a.out, 'folds'))
    if a.check_labels:
        for fold, df in dfp.groupby('fold'):
            ref = pd.read_csv(os.path.join(BASELINE, 'folds', fold, 'predictions.csv'))
            ok = (len(ref) == len(df)
                  and (ref['correct'].astype(bool).to_numpy() == df['correct'].astype(bool).to_numpy()).all()
                  and (ref['loudness'].fillna('').to_numpy() == df['loudness'].fillna('').to_numpy()).all())
            print(f'[labels] {fold}: {len(df)} frames vs pipeline {len(ref)}: '
                  f'{"IDENTICAL correct+loudness" if ok else "MISMATCH"}')
    table = summarize_folds(dfp)
    table.to_csv(os.path.join(a.out, 'folds_sx.csv'), index=False)
    print(format_sx_report(os.path.basename(os.path.normpath(a.out)), table))
    tot = table[table['fold'] == 'total'].iloc[0]
    print(f'HEADLINE {SENS_EXCL} {tot[SENS_EXCL]:.3f}  (inclusive {tot["sensitivity"]:.3f}) '
          f'-> {os.path.join(a.out, "folds_sx.csv")}')


def compare(a):
    """Diff two folds_sx.csv (e.g. this run vs the teacher's shipped one)."""
    x = pd.read_csv(a.out if a.out.endswith('.csv') else os.path.join(a.out, 'folds_sx.csv'))
    y = pd.read_csv(a.ref)
    cols = ['threshold', 'sensitivity', 'sensitivity_exclquiet', 'precision']
    m = x.merge(y, on='fold', suffixes=('', '_ref'))
    for c in cols:
        m[c + '_d'] = m[c] - m[c + '_ref']
    print(m[['fold'] + [f for c in cols for f in (c, c + '_ref', c + '_d')]].round(3).to_string(index=False))


def run(a):
    os.makedirs(a.out, exist_ok=True)
    cmd = [ENGINE_PY, os.path.abspath(__file__), 'infer', '--onnx', a.onnx, '--out', a.out]
    if a.cpu:
        cmd.append('--cpu')
    if a.folds:
        cmd += ['--folds', *a.folds]
    if a.max_frames:
        cmd += ['--max-frames', str(a.max_frames)]
    cmd += ['--pack', str(a.pack)] + (['--isolated'] if a.isolated else [])
    subprocess.run(cmd, check=True)
    score(a)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('phase', choices=['infer', 'score', 'run', 'compare'])
    ap.add_argument('--onnx')
    ap.add_argument('--out', required=True)
    ap.add_argument('--folds', nargs='*')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--pack', type=int, default=64, help='frames per ORT call (silent-frame interleaved)')
    ap.add_argument('--isolated', action='store_true', help='one call per frame (slow, reference)')
    ap.add_argument('--max-frames', type=int, default=0, help='per fold, dev only')
    ap.add_argument('--check-labels', action='store_true',
                    help='score: compare correct/loudness with cv-baseline-v4-moderate predictions')
    ap.add_argument('--ref', help='compare: reference folds_sx.csv')
    a = ap.parse_args()
    if a.phase in ('infer', 'run') and not a.onnx:
        ap.error('--onnx required')
    {'infer': infer, 'score': score, 'run': run, 'compare': compare}[a.phase](a)
