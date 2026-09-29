"""Build the teacher-target cache for a rung (see DESIGN.md).

    python 05_distill/cache.py --rung A [--workers 20] [--limit N]     # teacher: DISTILL_TEACHER / main.py --teacher

Needs onnxruntime-gpu + onnx + numpy: use .local/venv-onnx/bin/python.
Long: run through tools/launch_job.sh. Resumable (existing npz are skipped;
files are written to a temp name and renamed). Prerequisites, in order:
plan.py (writes _manifest/plan.csv), teacher_onnx.py (writes teacher_ext.onnx).
Refuses to run if the teacher's model.onnx changed since teacher.json was written.

N decode threads run `ffmpeg -ss <start> -t 59.6 -i file -ac 1 -ar 16000 -f f32le`
(seek before input) on slices taken in path order (kind to the spinning disk);
one GPU consumer computes mel + logits + code per slice and writes two files:
  <cache>/<relpath without extension>/h<hour:06d>.npz          the teacher's targets
    code  float16 (n,D)     teacher's head input (D read off the graph, teacher.json)
    logits float32 (n,C)    teacher predictions, centers subtracted (detect: > 0)
    start_s scalar          slice start in the source file
  <distill_cache>/_mel/yamnet/<same relpath>/h<hour:06d>.npz   YAMNet log-mel (n,96,64) f16
The mel does not depend on the teacher, so it is written once per slice for all teachers (and
skipped when another teacher already wrote it); caches built before the split embed `mel` in the
targets npz, which readers still accept (store.mel_path).
n = 62 normally; fewer for short files or a file that decodes short.
Rungs are cumulative: --rung B caches A and B (A's already-written slices skip).
Rung V is the validation pool (separate deployments), not part of A-D.
"""
import argparse, collections, csv, io, os, subprocess, sys, threading, time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import dpaths as D  # noqa: E402
import store  # noqa: E402
import teacher_onnx as T  # noqa: E402

ORDER = ['A', 'B', 'C', 'D']
DECODE_S = 59.6
LOOKAHEAD = 240


def ffmpeg_decode(path, start_s):
    """Generic path: ffmpeg -ss before -i. On long mp3 without an index this costs
    ~3 s per 100 000 s of offset (it scans frame headers from the start)."""
    cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-ss', repr(float(start_s)), '-t', repr(DECODE_S),
           '-i', path, '-ac', '1', '-ar', '16000', '-f', 'f32le', 'pipe:1']
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or len(r.stdout) < 4 * 16000:
        return None
    return np.frombuffer(r.stdout, np.float32)[:T.SLICE_SAMPLES]


_MP3_BR = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]   # MPEG1 layer 3, kbps
_MP3_SR = [44100, 48000, 32000]


def _mp3_hdr(b, i):
    """(bitrate_idx, sr_idx) of an MPEG1-layer-3 frame header at b[i], else None."""
    if i + 4 > len(b) or b[i] != 0xFF or (b[i + 1] & 0xFE) != 0xFA:
        return None
    br, sr = b[i + 2] >> 4, (b[i + 2] >> 2) & 3
    if br in (0, 15) or sr == 3:
        return None
    return br, sr


def mp3_probe(path, size, dur):
    """(data_start, bytes_per_s, frame_bytes, sr, hdr) if this is a CBR MPEG1-L3 file whose
    size agrees with bitrate*duration to 1%; else None."""
    with open(path, 'rb') as f:
        h = f.read(4096)
    d = 0
    if h[:3] == b'ID3':
        d = 10 + ((h[6] << 21) | (h[7] << 14) | (h[8] << 7) | h[9])
        with open(path, 'rb') as f:
            f.seek(d); h = f.read(4096)
    for i in range(len(h) - 8):
        hd = _mp3_hdr(h, i)
        if hd:
            br, sr = _MP3_BR[hd[0]] * 1000, _MP3_SR[hd[1]]
            fb = 144 * br / sr
            if _mp3_hdr(h, i + int(fb)) or _mp3_hdr(h, i + int(fb) + 1):
                Bps = br / 8
                if abs(Bps * dur - (size - d - i)) > 0.01 * Bps * dur:
                    return None
                return d + i, Bps, fb, sr, hd
    return None


def mp3_decode(path, start_s, info):
    """Byte-offset seek by the CBR bitrate, then ffmpeg on just that byte range (stdin)."""
    d0, Bps, fb, sr, hd = info
    lead = 8000                                   # bytes of resync/bit-reservoir lead-in
    want = int((DECODE_S + 2.0) * Bps) + lead
    pos = d0 + int(start_s * Bps) - lead
    pos = max(d0, pos)
    with open(path, 'rb') as f:
        f.seek(pos); buf = f.read(want + 1024)
    i0 = None
    for i in range(0, min(len(buf) - 4, 2048)):     # first frame header with matching params, confirmed by the next
        if _mp3_hdr(buf, i) == hd and any(_mp3_hdr(buf, i + int(fb) + k) == hd for k in (0, 1)):
            i0 = i; break
    if i0 is None:
        return None
    n0 = round((pos + i0 - d0) / fb)              # frames before the first decoded frame
    t_dec = n0 * 1152 / sr
    r = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-f', 'mp3', '-i', 'pipe:0', '-ac', '1', '-ar', '16000',
                        '-f', 'f32le', 'pipe:1'], input=buf[i0:], capture_output=True)
    x = np.frombuffer(r.stdout, np.float32)
    skip = int(round((start_s - t_dec) * 16000))
    if r.returncode != 0 or skip < 0 or len(x) < skip + 4 * 16000:
        return None
    return x[skip:skip + T.SLICE_SAMPLES]


def decode(path, start_s, mp3info=None):
    """float32 mono 16 kHz samples of [start_s, start_s + 59.6 s), or None on failure."""
    if mp3info is not None:
        x = mp3_decode(path, start_s, mp3info)
        if x is not None:
            return x
    return ffmpeg_decode(path, start_s)


_MP3 = {}


def load_slice(root, rel, start_s, size_dur):
    """Decode worker (runs in a spawned process: forking a threaded decoder pool out of the
    CUDA-holding parent was the throughput killer)."""
    if rel not in _MP3:
        _MP3[rel] = None
        if rel.lower().endswith('.mp3') and size_dur:
            try:
                _MP3[rel] = mp3_probe(os.path.join(root, rel), *size_dur)
            except Exception:
                pass
    return decode(os.path.join(root, rel), start_s, _MP3[rel])


out_path = D.slice_path      # the layout every cache level shares (cache_fe.py uses it too)


def write_slice(cache, rel, hour, mel, code, logits, start_s):
    """Shared mel first (if no teacher wrote it yet), targets last: the targets file is the done marker."""
    if store.mel_path('yamnet', rel, hour, cache) is None:
        store.write_mel('yamnet', rel, hour, mel)
    store.write_targets(cache, rel, hour, code, logits, start_s)


README = """# distill cache: teacher {teacher}

Teacher targets for distilling lite students (05_distill/DESIGN.md in the
buzzdetect-training repo). One npz per 1-minute slice (62 frames of 0.96 s,
starting at h*3600 s of the source file), path mirroring the audio tree:

    <relpath of source file, no extension>/h<hour:06d>.npz
      code   float16 (n,{code_dim})   teacher's head input
      logits float32 (n,{n_classes})     teacher predictions, activation_centers subtracted (detect: logit > 0)
      start_s float64           slice start in the source file, seconds
    n is 62 except for short files; frame k = samples [k*15360, (k+1)*15360) of the
    16 kHz mono slice. Classes without an activation center sit above 0 everywhere: ignore them.
    (Caches built before the mel split also hold `mel` float16 (n,96,64) in this file.)

The student's input spectrograms are not here: they do not depend on the teacher, so they live one
level up, shared by every teacher: ../_mel/<spec>/<same relpath>/h<hour:06d>.npz (`yamnet` = YAMNet's
log-mel; other front ends from cache_fe.py), each spec directory stamped with a fingerprint.json.
../_shared/durations.csv is the ffprobe durations, also teacher-independent.

_manifest/
    plan.csv          every slice with rung: relpath,hour,start_s,deployment,rank,first_rung,n_frames
                      (rungs A<B<C<D nested by rank with a per-deployment floor; V = validation
                      deployments, never in A-D)
    teacher.json      teacher onnx path + sha256, code tensor and width, opset, git commit, parity
    teacher_ext.onnx  the teacher with an extra `code` output
    blacklist.txt     excluded deployment dirs (all deployments the teacher trained on)

Made by main.py (05_distill/main.py --teacher <name>), or by hand, repo root, python =
.local/venv-onnx/bin/python (needs onnxruntime-gpu):
    python 05_distill/plan.py                 # enumerate, rank, assign rungs (kept once written)
    python 05_distill/teacher_onnx.py         # build/verify teacher_ext.onnx
    python 05_distill/cache.py --rung A       # then B, C, D, V (cumulative, resumable)
Audio decode: ffmpeg -ss <start> -t 59.6 -i <file> -ac 1 -ar 16000 -f f32le (mono downmix, swr resample).
Paths come from paths.local.json keys audio_root and distill_cache; the teacher from DISTILL_TEACHER.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rung', required=True, choices=ORDER + ['V'])
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--limit', type=int, default=None, help='only the first N pending slices (testing)')
    ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args()
    cache, root = D.need_cache(), D.AUDIO_ROOT
    store.require_current_teacher()
    store.stamp_mel('yamnet')
    rungs = {'V'} if a.rung == 'V' else set(ORDER[:ORDER.index(a.rung) + 1])
    sp = D.spec()
    open(os.path.join(cache, 'README.md'), 'w').write(
        README.format(teacher=D.TEACHER, code_dim=sp.code_dim, n_classes=sp.n_classes))

    plan = [r for r in csv.DictReader(open(D.PLAN)) if r['first_rung'] in rungs]
    plan.sort(key=lambda r: (r['relpath'], int(r['hour'])))
    todo = [r for r in plan if not os.path.exists(out_path(cache, r['relpath'], r['hour']))]
    if a.limit:
        todo = todo[:a.limit]
    tot_h = sum(int(r['n_frames']) * T.FRAME_S for r in todo) / 3600
    print(f'rung {a.rung}: {len(plan)} slices planned, {len(plan) - len(todo)} present, {len(todo)} to do ({tot_h:.1f} h)', flush=True)
    if not todo:
        return
    dur = store.load_durations()
    dec = ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context('spawn'))
    list(dec.map(int, ['0'] * a.workers))          # start the decode processes before CUDA exists
    teacher = T.Teacher(gpu=not a.cpu)
    print('sessions:', teacher.ext.get_providers()[0], teacher.mel.get_providers()[0], flush=True)
    wr = ThreadPoolExecutor(4)
    ahead = collections.deque()
    wfut = collections.deque()
    it = iter(todo)
    done = failed = short = 0
    frames = 0
    t0 = tl = time.time()
    dl = 0
    counts = collections.Counter()
    tim = collections.Counter()

    def refill():
        while len(ahead) < a.workers * 3:
            r = next(it, None)
            if r is None:
                return
            ahead.append((r, dec.submit(load_slice, root, r['relpath'], float(r['start_s']), dur.get(r['relpath']))))

    refill()
    while ahead:
        r, fut = ahead.popleft()
        refill()
        tw = time.time()
        x = fut.result()
        tim['wait_decode'] += time.time() - tw
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
                tg = time.time()
                mel, lg, cd = teacher(x)
                tim['gpu'] += time.time() - tg
                nfr = min(nfr, len(lg))
                wfut.append(wr.submit(write_slice, cache, r['relpath'], r['hour'],
                                      mel[:nfr], cd[:nfr], lg[:nfr], float(r['start_s'])))
                frames += nfr
                counts['ins_buzz>0'] += int((lg[:nfr, sp.buzz] > 0).sum())
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
          f'ins_buzz>0 fraction {counts["ins_buzz>0"] / max(1, frames):.4f}; '
          f'consumer waited on decode {tim["wait_decode"]:.0f}s, in teacher {tim["gpu"]:.0f}s', flush=True)


if __name__ == '__main__':
    main()
