"""Time micro-graphs as bench_arch does (engine venv). Also with the input already on the device (IOBinding)."""
import glob, os, statistics, sys, time
import numpy as np, onnxruntime as ort
d, seconds = sys.argv[1], float(sys.argv[2])
n = int(round(seconds * 16000)); hop, mn = 15360, 15600
frames = 1 + int(np.ceil(max(0, n - mn) / hop)); x = np.zeros(mn + hop * frames, np.float32)
x[:n] = np.random.default_rng(0).uniform(-0.1, 0.1, n)
so = ort.SessionOptions(); so.log_severity_level = 3


def med(f):
    for _ in range(3): f()
    ts = []
    for _ in range(25):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return 1000 * statistics.median(ts)


print(f'{seconds:g} s of audio ({x.nbytes / 1e6:.1f} MB); ms per call')
print(f'{"graph":28s} {"GPU":>7s} {"GPU, input on device":>21s} {"CPU":>7s}  nodes on CPU EP (GPU session)')
for p in sorted(glob.glob(os.path.join(d, '*.onnx'))):
    so_p = ort.SessionOptions(); so_p.log_severity_level = 3; so_p.enable_profiling = False
    g = ort.InferenceSession(p, so, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    c = ort.InferenceSession(p, so, providers=['CPUExecutionProvider'])
    t_g = med(lambda: g.run(None, {'waveform': x}))
    xg = ort.OrtValue.ortvalue_from_numpy(x, 'cuda', 0)
    b = g.io_binding(); b.bind_ortvalue_input('waveform', xg); b.bind_output('predictions', 'cuda')
    def bound():
        g.run_with_iobinding(b); b.copy_outputs_to_cpu()
    t_b = med(bound)
    t_c = med(lambda: c.run(None, {'waveform': x}))
    print(f'{os.path.basename(p)[:-5]:28s} {t_g:7.2f} {t_b:21.2f} {t_c:7.2f}', flush=True)
