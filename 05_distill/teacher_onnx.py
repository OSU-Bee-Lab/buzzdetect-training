"""Teacher graph for distillation targets: shipped ONNX + a `code` output, plus
the YAMNet mel front end, as fixed-length ORT sessions.

    python 05_distill/teacher_onnx.py     # build + verify + write teacher.json

Run with a python that has onnx, onnxruntime-gpu, soundfile, numpy
(/home/luke/projects/buzzdetect-training/.local/venv-onnx/bin/python).

The teacher graph ends `... -> Reshape (n,2048) -> MatMul -> Add -> Sub(centers)
-> predictions`; the Reshape output is the 2048-d code (two 1024-d views
concatenated). `_manifest/teacher_ext.onnx` has outputs (logits, code).
The mel graph is the step-0 `frontend_only` export: waveform -> (n,96,64)
log-mel patches; patch k starts at sample k*15360 (hop welded to 0.96 s) and
needs 240 samples of lookahead beyond its 15360, so a 62-frame slice needs
SLICE_SAMPLES = 953600 (59.6 s) of audio.
"""
import hashlib, json, os, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config  # noqa: E402

TEACHER_ONNX = '/home/luke/projects/buzzdetect/engine/models/v4-ft-ps-e60-moderate/model.onnx'
FRONTEND_ONNX = '/home/luke/projects/buzzdetect-training/.local/distill/arch/frontend_only/model.onnx'
FIXTURE = os.path.join(os.path.dirname(HERE), '04_deploy', 'fixtures', '230808_1208_s89520.flac')
HOP = 15360
FRAME_S = 0.96
SLICE_FRAMES = 62
SLICE_SAMPLES = 953600          # 59.6 s: 62 frames + 240 samples lookahead
CODE_DIM = 2048


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def find_code_tensor(model):
    """Name of the tensor feeding the head's MatMul (the 2048-d code)."""
    g = model.graph
    prod = {o: n for n in g.node for o in n.output}
    mm = [n for n in g.node if n.op_type == 'MatMul' and prod.get(n.output[0]) is not None]
    # the last MatMul before `predictions`
    node = prod[g.output[0].name]
    while node.op_type != 'MatMul':
        node = prod[node.input[0]]
    return node.input[0]


def build_ext(path_in=TEACHER_ONNX, path_out=None):
    import onnx
    from onnx import TensorProto, helper
    m = onnx.load(path_in)
    code = find_code_tensor(m)
    m.graph.output[0].name = m.graph.output[0].name       # keep 'predictions'
    m.graph.output.append(helper.make_tensor_value_info(code, TensorProto.FLOAT, [None, CODE_DIM]))
    if path_out:
        onnx.save(m, path_out)
    return m, code


def session(path_or_bytes, samples, gpu=True):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.add_free_dimension_override_by_name('samples', samples)
    prov = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if gpu else ['CPUExecutionProvider']
    s = ort.InferenceSession(path_or_bytes, so, providers=prov)
    if gpu and 'CUDAExecutionProvider' not in s.get_providers():
        print('WARNING: CUDA EP not loaded', file=sys.stderr)
    return s


class Teacher:
    """One waveform (SLICE_SAMPLES float32) -> mel (62,96,64), logits (62,15), code (62,2048)."""

    def __init__(self, cache=None, gpu=True, samples=SLICE_SAMPLES):
        cache = cache or config.DISTILL_CACHE
        self.ext = session(os.path.join(cache, '_manifest', 'teacher_ext.onnx'), samples, gpu)
        self.mel = session(FRONTEND_ONNX, samples, gpu)
        self.samples = samples

    def __call__(self, wav):
        x = np.ascontiguousarray(wav, np.float32)
        mel = self.mel.run(None, {'waveform': x})[0]
        logits, code = self.ext.run(None, {'waveform': x})
        return mel[:len(logits)], logits, code   # mel graph may emit one extra trailing patch


def main():
    import onnx
    man = os.path.join(config.DISTILL_CACHE, '_manifest')
    os.makedirs(man, exist_ok=True)
    out = os.path.join(man, 'teacher_ext.onnx')
    m, code = build_ext(TEACHER_ONNX, out)
    print('code tensor:', code)
    r = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', FIXTURE, '-ac', '1', '-ar', '16000',
                        '-f', 'f32le', 'pipe:1'], capture_output=True, check=True)
    x = np.frombuffer(r.stdout, np.float32).copy()
    sr = 16000
    n = len(x)
    print(f'fixture {n / sr:.1f} s')

    orig = session(TEACHER_ONNX, n, gpu=False)
    ext = session(out, n, gpu=False)
    p0 = orig.run(None, {'waveform': x})[0]
    p1, c1 = ext.run(None, {'waveform': x})
    err = float(np.abs(p0 - p1).max())
    print(f'logits {p0.shape} code {c1.shape} max|orig-ext| = {err:.2e}')
    assert err < 1e-5
    print(f'code stats: mean {c1.mean():.3f} std {c1.std():.3f} min {c1.min():.2f} max {c1.max():.2f}; fp16 max err {np.abs(c1.astype(np.float16).astype(np.float32) - c1).max():.3g}')

    # mel front end on the whole fixture and alignment checks
    mel_full = session(FRONTEND_ONNX, n, gpu=False).run(None, {'waveform': x})[0]
    print('mel (may have +1 trailing patch vs logits)', mel_full.shape, mel_full.dtype, f'range {mel_full.min():.2f}..{mel_full.max():.2f}')
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
    t = Teacher()
    mel, lg, cd = t(sl)
    print('slice shapes', mel.shape, lg.shape, cd.shape,
          f'logit diff vs full run {np.abs(lg - p0[:62]).max():.2e}, mel diff {np.abs(mel - mel_full[:len(mel)]).max():.2e}')
    assert mel.shape == (62, 96, 64) and lg.shape == (62, 15)

    try:
        commit = subprocess.run(['git', '-C', config.ROOT, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    meta = dict(
        onnx=TEACHER_ONNX, onnx_sha256=sha256(TEACHER_ONNX),
        ext_onnx='teacher_ext.onnx', ext_sha256=sha256(out),
        code_tensor=code, code_dim=CODE_DIM, opset=[(o.domain, o.version) for o in m.opset_import],
        frontend_onnx=FRONTEND_ONNX, frontend_sha256=sha256(FRONTEND_ONNX),
        slice_samples=SLICE_SAMPLES, slice_frames=SLICE_FRAMES, hop=HOP,
        parity_fixture=os.path.basename(FIXTURE), parity_max_abs_err=err,
        no_loss_classes=['aambient_scraping', 'mech_quadcopter'],
        git_commit=commit,
    )
    json.dump(meta, open(os.path.join(man, 'teacher.json'), 'w'), indent=2)
    print('wrote teacher.json')


if __name__ == '__main__':
    main()
