"""Front-end inputs for the distillation cache: a new spectrogram per slice, same targets.

    python 05_distill/cache_fe.py --rung B --frontends two32,lo32 [--workers 8] [--limit N]

The teacher's targets (`code`, `logits`) depend only on the audio, not on the student's front
end, so a new front end needs only its own `mel` re-computed. This decodes each slice that
already has a targets npz (same slice list, same frame count) and writes, per front end,

    <distill_cache>/_mel/<name>/<relpath without extension>/h<hour:06d>.npz
      mel  float16 (n,96,bands,channels)   frontends.mel_patches of the slice

so one decode feeds every requested front end. The mel depends on the audio and the front end
only, so this directory is shared by every teacher and every student architecture, and carries a
fingerprint.json of the front end's definition: editing a spec in frontends.py without renaming
it stops the next run instead of mixing old and new spectrograms. Slices whose mel already
exists (here, or in a pre-split `<teacher>/_fe/<name>`) are skipped. Resumable. Rungs are
cumulative like cache.py's; `V` is the validation pool. Needs numpy + ffmpeg only (no
TensorFlow, no GPU). Long: run through tools/launch_job.sh.
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
import dpaths as D  # noqa: E402
import store  # noqa: E402
import cache as C  # noqa: E402
import frontends as fes  # noqa: E402

SLICE_SAMPLES = D.SLICE_SAMPLES


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rung', required=True, choices=C.ORDER + ['V'])
    ap.add_argument('--frontends', required=True, help='comma-separated names from frontends.FRONTENDS')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    names = a.frontends.split(',')
    for n in names:
        assert n in fes.FRONTENDS and n != 'yamnet', n
    cache, root = D.need_cache(), D.AUDIO_ROOT
    for n in names:
        store.stamp_mel(n)
    rungs = {'V'} if a.rung == 'V' else set(C.ORDER[:C.ORDER.index(a.rung) + 1])
    plan = [r for r in csv.DictReader(open(D.PLAN)) if r['first_rung'] in rungs]
    plan.sort(key=lambda r: (r['relpath'], int(r['hour'])))
    have = [r for r in plan if store.has_targets(cache, r['relpath'], r['hour'])]
    # a slice is pending if any requested front end lacks it (shared level or a pre-split _fe dir)
    todo = [r for r in have if any(store.mel_path(n, r['relpath'], r['hour'], cache) is None for n in names)]
    if a.limit:
        todo = todo[:a.limit]
    print(f'rung {a.rung} {names}: {len(plan)} planned, {len(have)} with targets, {len(todo)} to do', flush=True)
    if not todo:
        return
    dur = store.load_durations()
    ex =ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context('spawn'))
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
                if store.mel_path(n, r['relpath'], r['hour'], cache) is None:    # one spec may already exist
                    wfut.append(wr.submit(store.write_mel, n, r['relpath'], r['hour'], mel))
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
