"""Front-end inputs for the distillation cache: a new spectrogram per slice, same targets.

    python 05_distill/cache_fe.py --rung B --frontends two32,lo32 [--workers 8] [--limit N]

The teacher's targets (`code`, `logits`) in the main cache depend only on the audio, not
on the student's front end, so a new front end needs only its own `mel` re-computed.
This decodes each slice that already has a main-cache npz (same slice list, same frame
count) and writes, per front end,

    <cache>/_fe/<name>/<relpath without extension>/h<hour:06d>.npz
      mel  float16 (n,96,bands,channels)   frontends.mel_patches of the slice

so one decode feeds every requested front end. Resumable (existing npz are skipped).
Rungs are cumulative like cache.py's; `V` is the validation pool. Needs numpy + ffmpeg
only (no TensorFlow, no GPU). Long: run through tools/launch_job.sh.
"""
import argparse
import collections
import csv
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import config  # noqa: E402
import cache as C  # noqa: E402
import frontends as fes  # noqa: E402

SLICE_SAMPLES = 953600


def fe_root(cache, name):
    return os.path.join(cache, '_fe', name)


def work(root, rel, start_s, size_dur, names, nfr):
    """Decode one slice and compute every requested front end (runs in a spawned process)."""
    x = C.load_slice(root, rel, start_s, size_dur)
    if x is None:
        return None
    if len(x) < SLICE_SAMPLES:
        x = np.concatenate([x, np.zeros(SLICE_SAMPLES - len(x), np.float32)])
    out = {}
    for n in names:
        out[n] = fes.mel_patches(x, fes.get(n))[:nfr].astype(np.float16)
    return out


def write(path, mel):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        np.savez(f, mel=mel)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rung', required=True, choices=C.ORDER + ['V'])
    ap.add_argument('--frontends', required=True, help='comma-separated names from frontends.FRONTENDS')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--cache', default=config.DISTILL_CACHE)
    a = ap.parse_args()
    names = a.frontends.split(',')
    for n in names:
        assert n in fes.FRONTENDS and n != 'yamnet', n
    cache, root = a.cache, config.AUDIO_ROOT
    rungs = {'V'} if a.rung == 'V' else set(C.ORDER[:C.ORDER.index(a.rung) + 1])
    plan = [r for r in csv.DictReader(open(os.path.join(cache, '_manifest', 'plan.csv'))) if r['first_rung'] in rungs]
    plan.sort(key=lambda r: (r['relpath'], int(r['hour'])))
    have = [r for r in plan if os.path.exists(C.out_path(cache, r['relpath'], r['hour']))]
    # a slice is pending if any requested front end lacks it
    todo = [r for r in have if not all(os.path.exists(C.out_path(fe_root(cache, n), r['relpath'], r['hour']))
                                       for n in names)]
    if a.limit:
        todo = todo[:a.limit]
    print(f'rung {a.rung} {names}: {len(plan)} planned, {len(have)} with targets, {len(todo)} to do', flush=True)
    if not todo:
        return
    dur = {r['relpath']: (int(r['size']), float(r['dur']))
           for r in csv.DictReader(open(os.path.join(cache, '_manifest', 'durations.csv'))) if r['dur']}
    ex = ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context('spawn'))
    wr = ThreadPoolExecutor(4)
    ahead, wfut = collections.deque(), collections.deque()
    it = iter(todo)
    done = failed = 0
    t0 = tl = time.time()

    def refill():
        while len(ahead) < a.workers * 3:
            r = next(it, None)
            if r is None:
                return
            with np.load(C.out_path(cache, r['relpath'], r['hour'])) as z:
                nfr = len(z['logits'])
            ahead.append((r, ex.submit(work, root, r['relpath'], float(r['start_s']),
                                       dur.get(r['relpath']), names, nfr)))

    refill()
    while ahead:
        r, fut = ahead.popleft()
        refill()
        res = fut.result()
        if res is None:
            failed += 1
            print(f'FAILED decode {r["relpath"]} h{r["hour"]}', flush=True)
        else:
            for n, mel in res.items():
                wfut.append(wr.submit(write, C.out_path(fe_root(cache, n), r['relpath'], r['hour']), mel))
            done += 1
            while len(wfut) > 64:
                wfut.popleft().result()
        if time.time() - tl >= 120:
            el = time.time() - t0
            print(f'[{time.strftime("%H:%M:%S")}] {done}/{len(todo)} slices, {done / el:.2f} slices/s, '
                  f'failed {failed}', flush=True)
            tl = time.time()
    for f in wfut:
        f.result()
    print(f'SUMMARY rung {a.rung} {names}: wrote {done} slices in {(time.time() - t0) / 60:.1f} min; failed {failed}', flush=True)


if __name__ == '__main__':
    main()
