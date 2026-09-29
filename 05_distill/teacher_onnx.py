"""Teacher graph for distillation targets: the shipped ONNX + a `code` output, plus
the YAMNet mel front end, as fixed-length ORT sessions.

    python 05_distill/teacher_onnx.py [--force]     # build + verify + write teacher.json

Run with a python that has onnx, onnxruntime-gpu, soundfile, numpy
(.local/venv-onnx/bin/python). Idempotent: does nothing when `teacher.json` already
records this teacher ONNX's sha256 and `teacher_ext.onnx` exists (`--force` rebuilds).

The teacher graph ends `... -> Reshape (n,D) -> MatMul -> Add -> Sub(centers)
-> predictions`; the tensor feeding that MatMul is the head input, the `code`
(D is read off the graph: 2048 for the trunk teachers, two 1024-d views
concatenated). `_manifest/teacher_ext.onnx` has outputs (logits, code).
The mel graph is the step-0 `frontend_only` export (bench_arch.py export): waveform ->
(n,96,64) log-mel patches; patch k starts at sample k*15360 (hop welded to 0.96 s) and
needs 240 samples of lookahead beyond its 15360, so a 62-frame slice needs
SLICE_SAMPLES = 953600 (59.6 s) of audio. It does not depend on the teacher.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dpaths as D  # noqa: E402
import store  # noqa: E402

HOP, FRAME_S, SLICE_FRAMES, SLICE_SAMPLES = D.HOP, D.FRAME_S, D.SLICE_FRAMES, D.SLICE_SAMPLES
TEACHER_ONNX, FRONTEND_ONNX, FIXTURE = D.TEACHER_ONNX, D.FRONTEND_ONNX, D.FIXTURE
sha256 = store.file_sha


def find_code_tensor(model):
    """Name of the tensor feeding the head's MatMul (the code)."""
    g = model.graph
    prod = {o: n for n in g.node for o in n.output}
    node = prod[g.output[0].name]
    while node.op_type != 'MatMul':
        if node.input[0] not in prod:
            raise SystemExit('teacher graph has no MatMul head to take a code from; not a probe-headed model')
        node = prod[node.input[0]]
    return node.input[0]


def build_ext(path_in=TEACHER_ONNX, path_out=None):
    import onnx
    from onnx import TensorProto, helper
    m = onnx.load(path_in)
    code = find_code_tensor(m)
    m.graph.output.append(helper.make_tensor_value_info(code, TensorProto.FLOAT, [None, None]))
    if path_out:
        onnx.save(m, path_out)
    return m, code


def session(path_or_bytes, samples, gpu=True):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.add_free_dimension_override_by_name('samples', samples)
    if gpu:
        so.intra_op_num_threads = 2     # CPU-side ops only; more threads spin against the ffmpeg decoders
    prov = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if gpu else ['CPUExecutionProvider']
    s = ort.InferenceSession(path_or_bytes, so, providers=prov)
    if gpu and 'CUDAExecutionProvider' not in s.get_providers():
        print('WARNING: CUDA EP not loaded', file=sys.stderr)
    return s


class Teacher:
    """One waveform (SLICE_SAMPLES float32) -> mel (62,96,64), logits (62,C), code (62,D)."""

    def __init__(self, gpu=True, samples=SLICE_SAMPLES):
        self.ext = session(D.TEACHER_EXT, samples, gpu)
        self.mel = session(FRONTEND_ONNX, samples, gpu)
        self.samples = samples

    def __call__(self, wav):
        x = np.ascontiguousarray(wav, np.float32)
        mel = self.mel.run(None, {'waveform': x})[0]
        logits, code = self.ext.run(None, {'waveform': x})
        return mel[:len(logits)], logits, code   # mel graph may emit one extra trailing patch


def up_to_date():
    if not (os.path.isfile(D.TEACHER_JSON) and os.path.isfile(D.TEACHER_EXT)):
        return False
    meta = json.load(open(D.TEACHER_JSON))     # a teacher.json from before code_dim was recorded is rebuilt
    return meta.get('onnx_sha256') == store.teacher_sha() and 'code_dim' in meta


def check_framing():
    """The slice geometry (62 x 0.96 s at 16 kHz) is fixed: refuse a teacher framed differently."""
    p = os.path.join(D.TEACHER_ENGINE_DIR, 'config_model.json')
    cfg = json.load(open(p))
    if cfg.get('samplerate') != 16000 or abs(cfg.get('framelength_s', 0) - FRAME_S) > 1e-9 \
            or cfg.get('samples_hop') != HOP:
        sys.exit(f'teacher {D.TEACHER}: {p} is not 16 kHz / {FRAME_S} s / hop {HOP}; the slice cache assumes it')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    import onnx
    D.need_cache()
    if not os.path.isfile(TEACHER_ONNX):
        sys.exit(f'teacher ONNX {TEACHER_ONNX} not found: deploy the teacher to buzzdetect first (04_deploy)')
    if not os.path.isfile(FRONTEND_ONNX):
        sys.exit(f'{FRONTEND_ONNX} missing: bench_arch.py export --candidates frontend_only (train env)')
    if up_to_date() and not a.force:
        print(f'teacher {D.TEACHER}: teacher.json current ({store.teacher_sha()[:12]}), nothing to do')
        return
    check_framing()
    os.makedirs(D.MANIFEST, exist_ok=True)
    m, code = build_ext(TEACHER_ONNX, D.TEACHER_EXT)
    print('code tensor:', code)
    r = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', FIXTURE, '-ac', '1', '-ar', '16000',
                        '-f', 'f32le', 'pipe:1'], capture_output=True, check=True)
    x = np.frombuffer(r.stdout, np.float32).copy()
    sr = 16000
    n = len(x)
    print(f'fixture {n / sr:.1f} s')

    orig = session(TEACHER_ONNX, n, gpu=False)
    ext = session(D.TEACHER_EXT, n, gpu=False)
    p0 = orig.run(None, {'waveform': x})[0]
    p1, c1 = ext.run(None, {'waveform': x})
    err = float(np.abs(p0 - p1).max())
    print(f'logits {p0.shape} code {c1.shape} max|orig-ext| = {err:.2e}')
    assert err < 1e-5
    assert p0.shape[1] == D.spec().n_classes, (p0.shape, D.spec().classes)
    code_dim = int(c1.shape[1])
    print(f'code stats: mean {c1.mean():.3f} std {c1.std():.3f} min {c1.min():.2f} max {c1.max():.2f}; '
          f'fp16 max err {np.abs(c1.astype(np.float16).astype(np.float32) - c1).max():.3g}')

    # mel front end on the whole fixture and alignment checks
    mel_full = session(FRONTEND_ONNX, n, gpu=False).run(None, {'waveform': x})[0]
    print('mel (may have +1 trailing patch vs logits)', mel_full.shape, mel_full.dtype,
          f'range {mel_full.min():.2f}..{mel_full.max():.2f}')
    assert mel_full.shape[1:] == (96, 64), mel_full.shape
    k = 5
    cut = np.ascontiguousarray(x[k * HOP:])
    mel_cut = session(FRONTEND_ONNX, len(cut), gpu=False).run(None, {'waveform': cut})[0]
    d_mel = float(np.abs(mel_cut[0] - mel_full[k]).max())
    p_cut = session(TEACHER_ONNX, len(cut), gpu=False).run(None, {'waveform': cut})[0]
    d_log = float(np.abs(p_cut[0] - p0[k]).max())
    print(f'frame {k} vs cut at {k * HOP}: mel diff {d_mel:.2e}, logits diff {d_log:.2e}')
    assert d_mel < 1e-4 and d_log < 1e-3

    # a 62-frame slice with SLICE_SAMPLES gives exactly 62 frames, equal to the full run's
    sl = np.ascontiguousarray(x[:SLICE_SAMPLES])
    mel, lg, cd = Teacher()(sl)
    print('slice shapes', mel.shape, lg.shape, cd.shape,
          f'logit diff vs full run {np.abs(lg - p0[:62]).max():.2e}, mel diff {np.abs(mel - mel_full[:len(mel)]).max():.2e}')
    assert mel.shape == (62, 96, 64) and lg.shape == (62, D.spec().n_classes)

    try:
        commit = subprocess.run(['git', '-C', D.ROOT, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    meta = dict(
        teacher=D.TEACHER, onnx=TEACHER_ONNX, onnx_sha256=store.teacher_sha(),
        ext_onnx='teacher_ext.onnx', ext_sha256=sha256(D.TEACHER_EXT),
        code_tensor=code, code_dim=code_dim, classes=D.spec().classes,
        opset=[(o.domain, o.version) for o in m.opset_import],
        frontend_onnx=FRONTEND_ONNX, frontend_sha256=sha256(FRONTEND_ONNX),
        slice_samples=SLICE_SAMPLES, slice_frames=SLICE_FRAMES, hop=HOP,
        parity_fixture=os.path.basename(FIXTURE), parity_max_abs_err=err,
        no_loss_classes=[D.spec().classes[i] for i in D.spec().dead],
        git_commit=commit,
    )
    json.dump(meta, open(D.TEACHER_JSON, 'w'), indent=2)
    print('wrote teacher.json')


if __name__ == '__main__':
    main()
