"""Student checkpoint -> deployable waveform->predictions ONNX (+ parity and speed).

    # export (+ parity check on the fixture), train env
    conda run -n buzzdetect-train python 05_distill/export_student.py export \
        --run a05_A --name distill-a05-A
    # speed, same harness as bench_arch.py time, engine venv (CUDA ORT)
    /home/luke/projects/buzzdetect/engine/.venv/bin/python3 05_distill/export_student.py time --name distill-a05-A

`--run` is a distill_train run (`05_distill/data/runs/<run>/student_mel.keras`);
`--init-only` instead exports the untrained YAMNet channel-selected init (for
plumbing and speed checks with real weights). The mel training graph's weights
are copied by layer name into the waveform graph (YAMNet front end inside), the
teacher's `activation_centers` are subtracted in the head's bias (so the output
is on the deployed logit scale: detection is logit > 0, classes without a
center get 0), and the graph goes through the deploy passes (BN fold, Conv+Relu
fuse) and the same io rename as 04_deploy / bench_arch. Parity: ONNX vs the
Keras waveform student on the fixture, max |diff| must be < 1e-4.

Writes `05_distill/data/models/<name>/{model.onnx,config_model.json}`, a
buzzdetect-format model dir (never into buzzdetect's engine/models).
"""
import argparse
import json
import os
import sys

import tensorflow as tf  # noqa: F401  (must come first, see CLAUDE.md)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, '04_deploy'))
import student as st  # noqa: E402
import dpaths as D  # noqa: E402
sys.path.insert(0, ROOT)
from utils import git_branch  # noqa: E402

MAIN = D.MAIN
LOCAL = D.LOCAL                    # 05_distill/data/<teacher>
TEACHER_CFG = os.path.join(D.TEACHER_MODEL_DIR, 'config_model.json')
ENGINE_CFG = os.path.join(D.TEACHER_ENGINE_DIR, 'config_model.json')     # template for the student's engine config
FIXTURE = D.FIXTURE
TOL = 1e-4


def rename_io(model):
    g = model.graph
    old_in, old_out = g.input[0].name, g.output[0].name
    for n in g.node:
        n.input[:] = ['waveform' if i == old_in else i for i in n.input]
        n.output[:] = ['predictions' if o == old_out else o for o in n.output]
    g.input[0].name, g.output[0].name = 'waveform', 'predictions'
    d = g.input[0].type.tensor_type.shape.dim[0]
    if d.HasField('dim_param'):
        d.dim_param = 'samples'
    return model


def centers_vector(classes=None):
    """Teacher activation centers for `classes` (default: all the teacher's), 0 for classes without one."""
    cfg = json.load(open(TEACHER_CFG))
    c = cfg['activation_centers']
    classes = classes or cfg['classes']
    return np.array([c.get(k, 0.0) for k in classes], np.float32), cfg


def filters_of(mel_model):
    out, i = [], 0
    while True:
        try:
            l = mel_model.get_layer(f'layer{i + 1}_' + ('conv' if i == 0 else 'pointwise_conv'))
        except ValueError:
            return out
        out.append(l.filters)
        i += 1


def run_frontend(a):
    """Front end the run trained on (curve.json args), 'yamnet' for --init-only."""
    if a.init_only or not a.run:
        return 'yamnet'
    return json.load(open(os.path.join(LOCAL, 'runs', a.run, 'curve.json')))['args'].get('frontend', 'yamnet')


def run_classes(a):
    """Classes the run's student outputs (curve.json `keep_classes`), None = all of the teacher's."""
    if a.init_only or not a.run:
        return None
    args = json.load(open(os.path.join(LOCAL, 'runs', a.run, 'curve.json')))['args']
    return args.get('keep_classes') if args.get('classes') else None


def load_mel(a):
    if a.init_only:
        import student_init as si
        m = st.build_student(st.widths_for(a.alpha), input_type='mel', name='init')
        si.init_from_yamnet(m)
        return m
    return tf.keras.models.load_model(os.path.join(LOCAL, 'runs', a.run, 'student_mel.keras'),
                                      compile=False)


def do_export(a):
    import onnx
    from onnx_passes import optimize
    mel = load_mel(a)
    filters = filters_of(mel)
    fe_name = run_frontend(a)
    classes = run_classes(a)
    wav = st.build_student(filters, n_out=len(classes) if classes else D.spec().n_classes,
                           name=a.name.replace('-', '_').replace('.', '_'), frontend=fe_name)
    n = st.copy_weights(mel, wav)
    centers, tcfg = centers_vector(classes)
    head = wav.get_layer('logits')
    w, b = head.get_weights()
    head.set_weights([w, b - centers])          # deployed scale: detect at logit > 0
    print(f'copied {n} layers, filters {filters}, params {wav.count_params()}', flush=True)

    d = os.path.join(LOCAL, 'models', a.name)
    os.makedirs(d, exist_ok=True)
    wav(tf.zeros([16000], tf.float32))
    raw = os.path.join(d, 'raw.onnx')
    wav.export(raw, format='onnx', verbose=False, opset_version=17)
    m = onnx.load(raw)
    m, n_folded, n_fused, _ = optimize(m)
    m = rename_io(m)
    onnx.checker.check_model(m)
    onnx.save(m, os.path.join(d, 'model.onnx'))
    os.remove(raw)
    print(f'passes: folded {n_folded}, fused {n_fused}', flush=True)

    cfg = json.load(open(ENGINE_CFG))
    if classes:                                  # a class-subset student: the engine config lists only its outputs
        cfg['classes'] = list(classes)
        cfg['center_stats'] = {k: v for k, v in cfg.get('center_stats', {}).items() if k in classes}
    # one patch needs 15360 samples plus the STFT window's tail (25 ms window: 15600); never below
    # frontends.ENGINE_MIN_SAMPLES, or whole-frame chunks land on the engine's float32 overshoot
    cfg['samples_min'] = 15600 if fe_name == 'yamnet' else st.fes.get(fe_name).min_samples
    cfg['metadata'] = {'embeddername': 'distilled_student', 'set': D.spec().set,
                       'trained_date': __import__('datetime').date.today().isoformat(),
                       'modelname_internal': a.name, 'branch': git_branch(),
                       'teacher': D.TEACHER, 'filters': filters,
                       'source_run': None if a.init_only else a.run, 'frontend': fe_name, 'classes': classes or 'all',
                       'note': 'distilled single-pass student; activation_centers folded into the head bias'}
    json.dump(cfg, open(os.path.join(d, 'config_model.json'), 'w'), indent=2)
    print(f'wrote {d}', flush=True)
    check_framing(os.path.join(d, 'model.onnx'), cfg)

    if not a.no_parity:
        parity(wav, os.path.join(d, 'model.onnx'), a.parity_seconds,
               classes.index('ins_buzz') if classes else D.spec().buzz)


def engine_frames(n, hop, samples_min):
    """buzzdetect's OnnxModel.n_frames, verbatim: float32 multiply by the hop's reciprocal."""
    if n <= 0:
        return 0
    return 1 + int(np.ceil(np.float32(max(0, n - samples_min)) * (np.float32(1.0) / np.float32(hop))))


def check_framing(onnx_path, cfg):
    """Fail unless the graph returns as many frames as the engine will keep, at every length a chunk can have.

    The engine zero-pads each chunk to a fixed session and keeps engine_frames(n) rows. If it expects one more
    than the graph returns for n real samples, the extra row is computed from padding alone: a silence frame
    (a buzz-positive logit) stamped one frame late, which was fast_v1's false positive on every chunk boundary.
    Whole-frame chunks (what the analyzer cuts) and ragged lengths just either side of the first frame and of
    the hop multiples are checked; lengths where float32(k*hop)/hop rounds up past k are skipped because only
    a graph that reproduces that overshoot could match, and they are not whole-frame chunks.
    """
    import onnxruntime as ort
    hop, mn = cfg['samples_hop'], cfg['samples_min']
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    recip = np.float32(1.0) / np.float32(hop)
    lengths = {F * hop for F in (1, 2, 3, 4, 5, 6, 8, 21, 104, 208)}
    lengths |= {mn - 1, mn, mn + 1, mn + 137}
    for k in range(1, 13):
        if int(np.ceil(np.float32(k * hop) * recip)) == k:      # exact there; the overshoot k are skipped
            lengths |= {mn + k * hop - 1, mn + k * hop, mn + k * hop + 1}
    bad = []
    for n in sorted(lengths):
        got = sess.run(None, {name: np.zeros(n, np.float32)})[0].shape[0]
        want = engine_frames(n, hop, mn)
        if got != want:
            bad.append((n, got, want))
    if bad:
        raise SystemExit('framing mismatch (n, graph rows, engine expects): '
                         + '; '.join(map(str, bad[:6])) + (f' (+{len(bad) - 6} more)' if len(bad) > 6 else ''))
    print(f'framing OK: graph rows == engine frame count at {len(lengths)} lengths (samples_min {mn})', flush=True)


def parity(wav, onnx_path, seconds, buzz):
    import librosa
    import onnxruntime as ort
    x, _ = librosa.load(FIXTURE, sr=16000, mono=True)
    x = x[:int(seconds * 16000)].astype(np.float32)
    ref = wav(x, training=False).numpy()
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    out = sess.run(None, {'waveform': x})[0]
    diff = np.abs(ref - out).max()
    print(f'parity on fixture ({len(x) / 16000:.0f} s, {len(ref)} frames): max |onnx - keras| = {diff:.2e} '
          f'({"PASS" if diff < TOL else "FAIL"} < {TOL:g}); buzz detections keras {(ref[:, buzz] > 0).sum()} '
          f'onnx {(out[:, buzz] > 0).sum()}', flush=True)
    if diff >= TOL:
        sys.exit(1)


def do_time(a):
    """bench_arch's harness verbatim (20 s and 200 s chunks), reference in-process."""
    import bench_arch
    bench_arch.OUT = os.path.join(LOCAL, 'models')
    for secs in (20, 200):
        bench_arch.do_time(secs, a.repeats, 2, [a.name])
        r = json.load(open(os.path.join(bench_arch.OUT, 'results.json')))
        row = {'seconds': secs, 'repeats': a.repeats, 'gpu': r[a.name]['GPU']['rate'],
               'yamnet_gpu': r['yamnet_large_general']['GPU']['rate'],
               'teacher_gpu': r[D.TEACHER]['GPU']['rate']}
        row['x_yamnet'] = row['gpu'] / row['yamnet_gpu']
        json.dump(row, open(os.path.join(bench_arch.OUT, a.name, f'speed_{secs}.json'), 'w'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('phase', choices=['export', 'time', 'check'])
    ap.add_argument('--dir', help='check: a model dir (model.onnx + config_model.json), e.g. a deployed one')
    ap.add_argument('--run')
    ap.add_argument('--name')
    ap.add_argument('--init-only', action='store_true')
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--no-parity', action='store_true')
    ap.add_argument('--parity-seconds', type=float, default=120)
    ap.add_argument('--repeats', type=int, default=15)
    a = ap.parse_args()
    if a.phase in ('export', 'time') and not a.name:
        ap.error('--name')
    if a.phase == 'export':
        if not a.run and not a.init_only:
            ap.error('--run or --init-only')
        do_export(a)
    elif a.phase == 'check':
        check_framing(os.path.join(a.dir, 'model.onnx'),
                      json.load(open(os.path.join(a.dir, 'config_model.json'))))
    else:
        do_time(a)
