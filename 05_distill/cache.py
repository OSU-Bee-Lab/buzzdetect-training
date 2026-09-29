"""Build the teacher-target cache for a rung (see DESIGN.md).

    python 05_distill/cache.py --rung A [--workers 16] [--limit N]

Needs onnxruntime-gpu + onnx + numpy: use
/home/luke/projects/buzzdetect-training/.local/venv-onnx/bin/python.
Long: run through tools/launch_job.sh. Resumable (existing npz are skipped;
files are written to a temp name and renamed). Prerequisites, in order:
plan.py (writes _manifest/plan.csv), teacher_onnx.py (writes teacher_ext.onnx).

N decode threads run `ffmpeg -ss <start> -t 59.6 -i file -ac 1 -ar 16000 -f f32le`
(seek before input) on slices taken in path order (kind to the spinning disk);
one GPU consumer computes mel + logits + code per slice and writes
<cache>/<relpath without extension>/h<hour:06d>.npz with
  mel   float16 (n,96,64)   YAMNet log-mel patches of the plain frame
  code  float16 (n,2048)    teacher's 2048-d head input
  logits float32 (n,15)     teacher predictions, centers subtracted (detect: > 0)
  start_s scalar            slice start in the source file
n = 62 normally; fewer for short files or a file that decodes short.
Rungs are cumulative: --rung B caches A and B (A's already-written slices skip).
Rung V is the validation pool (separate deployments), not part of A-D.
"""
import argparse, collections, csv, io, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import config  # noqa: E402
import teacher_onnx as T  # noqa: E402

ORDER = ['A', 'B', 'C', 'D']
DECODE_S = 59.6
LOOKAHEAD = 240


def decode(path, start_s):
    """float32 mono 16 kHz samples of [start_s, start_s + 59.6 s), or None on failure."""
    cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-ss', repr(float(start_s)), '-t', repr(DECODE_S),
           '-i', path, '-ac', '1', '-ar', '16000', '-f', 'f32le', 'pipe:1']
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or len(r.stdout) < 4 * 16000:
        return None
    return np.frombuffer(r.stdout, np.float32)[:T.SLICE_SAMPLES]


def out_path(cache, rel, hour):
    return os.path.join(cache, os.path.splitext(rel)[0], f'h{int(hour):06d}.npz')


def write_npz(path, mel, code, logits, start_s):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        np.savez(f, mel=mel.astype(np.float16), code=code.astype(np.float16),
                 logits=logits.astype(np.float32), start_s=np.float64(start_s))
    os.replace(tmp, path)


README = """# distill cache: teacher-v4-ft-ps-e60-moderate

Teacher targets for distilling the lite student (05_distill/DESIGN.md in the
buzzdetect-training repo). One npz per 1-minute slice (62 frames of 0.96 s,
starting at h*3600 s of the source file), path mirroring the audio tree:

    <relpath of source file, no extension>/h<hour:06d>.npz
      mel    float16 (n,96,64)  YAMNet log-mel patches of the plain frame (student input)
      code   float16 (n,2048)   teacher's 2048-d head input (two 1024-d views concatenated)
      logits float32 (n,15)     teacher predictions, activation_centers subtracted (detect: logit > 0)
      start_s float64           slice start in the source file, seconds
    n is 62 except for short files; frame k = samples [k*15360, (k+1)*15360) of the
    16 kHz mono slice. Classes 0 (aambient_scraping) and 14 (mech_quadcopter) sit above 0
    everywhere: ignore them.

_manifest/
    plan.csv          every slice with rung: relpath,hour,start_s,deployment,rank,first_rung,n_frames
                      (rungs A<B<C<D nested by rank with a per-deployment floor; V = validation
                      deployments, never in A-D)
    teacher.json      teacher onnx path + sha256, code tensor, opset, git commit, parity
    teacher_ext.onnx  the teacher with an extra `code` output
    blacklist.txt     excluded deployment dirs (all deployments the teacher trained on)
    durations.csv     ffprobe durations (cache for plan.py)

Made by (repo root; python = .local/venv-onnx/bin/python, needs onnxruntime-gpu):
    python 05_distill/plan.py                 # enumerate, rank, assign rungs
    python 05_distill/teacher_onnx.py         # build/verify teacher_ext.onnx
    python 05_distill/cache.py --rung A       # then B, C, D, V (cumulative, resumable)
Audio decode: ffmpeg -ss <start> -t 59.6 -i <file> -ac 1 -ar 16000 -f f32le (mono downmix, swr resample).
Paths come from paths.local.json keys audio_root and distill_cache.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rung', required=True, choices=ORDER + ['V'])
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int, default=None, help='only the first N pending slices (testing)')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--cache', default=config.DISTILL_CACHE)
    a = ap.parse_args()
    cache, root = a.cache, config.AUDIO_ROOT
    rungs = {'V'} if a.rung == 'V' else set(ORDER[:ORDER.index(a.rung) + 1])
    open(os.path.join(cache, 'README.md'), 'w').write(README)

    plan = [r for r in csv.DictReader(open(os.path.join(cache, '_manifest', 'plan.csv'))) if r['first_rung'] in rungs]
    plan.sort(key=lambda r: (r['relpath'], int(r['hour'])))
    todo = [r for r in plan if not os.path.exists(out_path(cache, r['relpath'], r['hour']))]
    if a.limit:
        todo = todo[:a.limit]
    tot_h = sum(int(r['n_frames']) * T.FRAME_S for r in todo) / 3600
    print(f'rung {a.rung}: {len(plan)} slices planned, {len(plan) - len(todo)} present, {len(todo)} to do ({tot_h:.1f} h)', flush=True)
    if not todo:
        return
    teacher = T.Teacher(cache, gpu=not a.cpu)
    print('sessions:', teacher.ext.get_providers()[0], teacher.mel.get_providers()[0], flush=True)

    def load(r):
        return decode(os.path.join(root, r['relpath']), float(r['start_s']))

    dec = ThreadPoolExecutor(a.workers)
    wr = ThreadPoolExecutor(4)
    ahead = collections.deque()
    wfut = collections.deque()
    it = iter(todo)
    done = failed = short = 0
    frames = 0
    t0 = tl = time.time()
    dl = 0
    counts = collections.Counter()

    def refill():
        while len(ahead) < a.workers * 3:
            r = next(it, None)
            if r is None:
                return
            ahead.append((r, dec.submit(load, r)))

    refill()
    while ahead:
        r, fut = ahead.popleft()
        refill()
        x = fut.result()
        if x is None:
            failed += 1
            print(f'FAILED decode {r["relpath"]} h{r["hour"]}', flush=True)
        else:
            nfr = min(int(r['n_frames']), max(0, (len(x) - LOOKAHEAD) // T.HOP))
            if len(x) < T.SLICE_SAMPLES:
                x = np.concatenate([x, np.zeros(T.SLICE_SAMPLES - len(x), np.float32)])
            if nfr == 0:
                failed += 1
            else:
                if nfr < int(r['n_frames']):
                    short += 1
                mel, lg, cd = teacher(x)
                nfr = min(nfr, len(lg))
                wfut.append(wr.submit(write_npz, out_path(cache, r['relpath'], r['hour']),
                                      mel[:nfr], cd[:nfr], lg[:nfr], float(r['start_s'])))
                frames += nfr
                counts['ins_buzz>0'] += int((lg[:nfr, 8] > 0).sum())
                done += 1
            while len(wfut) > 32:
                wfut.popleft().result()
        now = time.time()
        if now - tl >= 120:
            el = now - t0
            print(f'[{time.strftime("%H:%M:%S")}] {done}/{len(todo)} slices, {done / el:.2f} slices/s, '
                  f'{frames * T.FRAME_S / 3600 / (el / 60):.2f} h audio/min, failed {failed}, short {short}, '
                  f'ins_buzz>0 {counts["ins_buzz>0"] / max(1, frames):.4f}', flush=True)
            tl = now
    for f in wfut:
        f.result()
    el = time.time() - t0
    print(f'SUMMARY rung {a.rung}: wrote {done} slices ({frames} frames, {frames * T.FRAME_S / 3600:.1f} h) in {el / 60:.1f} min = '
          f'{done / el:.2f} slices/s, {frames * T.FRAME_S / 3600 / (el / 60):.2f} h audio/min; failed {failed}, short {short}, '
          f'ins_buzz>0 fraction {counts["ins_buzz>0"] / max(1, frames):.4f}', flush=True)


if __name__ == '__main__':
    main()
