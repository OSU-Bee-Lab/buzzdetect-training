"""Weight-agnostic inference-speed head-to-head: YAMNet vs. student candidates.

Two phases, because export needs TensorFlow (buzzdetect-train env) and timing
needs a CUDA onnxruntime plus the buzzdetect engine's make_session (its venv):

  conda run -n buzzdetect-train python 05_distill/bench_arch.py export
  /home/luke/projects/buzzdetect/engine/.venv/bin/python3 05_distill/bench_arch.py time

`export` builds each candidate with random weights, exports it to ONNX and runs
the deploy passes (04_deploy/onnx_passes.optimize + rename io), writes
<OUT>/<name>/model.onnx and stats.json (params, MACs). `time` follows
buzzdetect/benchmarks/model-speed/bench_models.py: 20 s of noise, 2 warmup + 15
timed session.run() calls, rate = audio s / wall s; it times the reference
engine models in the same process and writes results.json. CPU rates are for
information only: this box's i7-2600 has no AVX2 and is not representative.
"""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import dpaths as D  # noqa: E402

OUT = D.ARCH                                  # random-weight timings: independent of the teacher
ENGINE = D.ENGINE
REFERENCES = ['yamnet_large_general', D.TEACHER]

# name -> (kind, alpha, depth)
CANDIDATES = {
    'a1.00': ('net', 1.0, 14),
    'a0.75': ('net', 0.75, 14),
    'a0.50': ('net', 0.5, 14),
    'a0.375': ('net', 0.375, 14),
    'a0.25': ('net', 0.25, 14),
    'a0.50_d12': ('net', 0.5, 12),
    'frontend_only': ('frontend', 0.5, 14),
}


def parse(name):
    """(kind, alpha, depth, frontend). 'a0.50@two32' = trunk a0.50 on front end two32;
    'fe@two32' = that front end alone; plain names are the YAMNet front end."""
    base, _, fe = name.partition('@')
    if base == 'fe':
        return 'frontend', 0.5, 14, fe
    if base in CANDIDATES:
        kind, alpha, depth = CANDIDATES[base]
    else:                                   # a<alpha>[_d<depth>]
        a, _, d = base[1:].partition('_d')
        kind, alpha, depth = 'net', float(a), int(d or 14)
    return kind, alpha, depth, fe or 'yamnet'


def do_export(names):
    import tensorflow  # noqa: F401  (must precede pandas-importing modules)
    import onnx

    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, '04_deploy'))
    sys.path.insert(0, HERE)
    from onnx_passes import optimize
    import student as st

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

    for name in names:
        kind, alpha, depth, fe = parse(name)
        filters = st.widths_for(alpha, depth)
        model = st.build_student(filters, stop_at_features=(kind == 'frontend'), frontend=fe,
                                 name=name.replace('.', '_').replace('@', '_'))
        model(tensorflow.zeros([16000], tensorflow.float32))
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, 'raw.onnx')
        model.export(tmp, format='onnx', verbose=False, opset_version=17)
        m = onnx.load(tmp)
        m, n_folded, n_fused, _ = optimize(m)
        m = rename_io(m)
        onnx.checker.check_model(m)
        onnx.save(m, os.path.join(d, 'model.onnx'))
        os.remove(tmp)
        stats = {'filters': filters, 'params': int(model.count_params()),
                 'macs': None if kind == 'frontend' else st.macs_per_frame(
                     filters, h=(96 if fe == 'yamnet' else st.fes.get(fe).frames),
                     w=(64 if fe == 'yamnet' else st.fes.get(fe).bands),
                     in_channels=(1 if fe == 'yamnet' else st.fes.get(fe).n_channels)),
                 'frontend': fe,
                 'folded': n_folded, 'fused': n_fused}
        json.dump(stats, open(os.path.join(d, 'stats.json'), 'w'), indent=1)
        print(name, stats, flush=True)


def do_time(seconds, repeats, warmup, names):
    import numpy as np
    os.chdir(ENGINE)
    sys.path.insert(0, ENGINE)
    from src.inference import onnx as bd_onnx

    def run(path, processor, samples_session, audio):
        session = bd_onnx.make_session(path, processor, samples_session)
        key = session.get_inputs()[0].name
        feed = {key: np.zeros(samples_session, dtype=np.float32)}
        feed[key][:len(audio)] = audio
        for _ in range(warmup):
            session.run(None, feed)
        ts = []
        for _ in range(repeats):
            t = time.perf_counter()
            session.run(None, feed)
            ts.append(time.perf_counter() - t)
        rates = [seconds / t for t in ts]
        return statistics.mean(rates), statistics.stdev(rates) / len(rates) ** 0.5, \
            session.get_providers()[0]

    n = int(round(seconds * 16000))
    hop, mn = 15360, 15600
    frames = 1 + int(np.ceil(np.float32(max(0, n - mn)) * (np.float32(1) / np.float32(hop))))
    samples_session = mn + hop * frames
    audio = np.random.default_rng(0).uniform(-0.1, 0.1, n).astype(np.float32)

    items = [(r, os.path.join(ENGINE, 'models', r, 'model.onnx')) for r in REFERENCES]
    items += [(c, os.path.join(OUT, c, 'model.onnx')) for c in names]
    results = {}
    for name, path in items:
        row = {}
        for proc in ('GPU', 'CPU'):
            mean, se, prov = run(os.path.abspath(path), proc, samples_session, audio)
            row[proc] = {'rate': mean, 'se': se, 'provider': prov}
            print(f'{name:24s} {proc} {mean:8.0f} +- {se:.0f} s/s  ({prov})', flush=True)
        sp = os.path.join(OUT, name, 'stats.json')
        if os.path.exists(sp):
            row.update(json.load(open(sp)))
        results[name] = row
    json.dump(results, open(os.path.join(OUT, 'results.json'), 'w'), indent=1)

    g = results['yamnet_large_general']['GPU']['rate']
    t = results[D.TEACHER]['GPU']['rate']
    print(f'\n{seconds:g} s audio/run, {repeats} runs ({warmup} warmup); '
          f'CPU is an i7-2600, informational only')
    print(f'{"model":24s} {"GPU s/s":>8s} {"xYAM":>6s} {"xTeach":>7s} {"CPU s/s":>8s} '
          f'{"params":>10s} {"MMACs":>8s}')
    for name, r in results.items():
        gr = r['GPU']['rate']
        macs = f'{r["macs"] / 1e6:8.1f}' if r.get('macs') else f'{"-":>8s}'
        print(f'{name:24s} {gr:8.0f} {gr / g:6.2f} {gr / t:7.2f} {r["CPU"]["rate"]:8.0f} '
              f'{r.get("params", 0):10d} {macs}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('phase', choices=['export', 'time'])
    ap.add_argument('--seconds', type=float, default=20)
    ap.add_argument('--repeats', type=int, default=15)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--candidates', nargs='*', default=list(CANDIDATES))
    ap.add_argument('--out', help='output dir (default 05_distill/data/arch)')
    a = ap.parse_args()
    if a.out:
        OUT = a.out
    if a.phase == 'export':
        do_export(a.candidates)
    else:
        do_time(a.seconds, a.repeats, a.warmup, a.candidates)
