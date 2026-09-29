"""Shard-level streaming for rungs too big to pack locally (C-D).

    python 05_distill/shards.py pack --rung D      # packs every group A..D (resumable)

`pack` writes `<cache>/_shards/<group>/s<k>.npz` (uncompressed; mel f16, code f16,
logits f32; ~4092 frames, ~65 MB each) plus `index.json` per group. A group is
the slices whose plan `first_rung` is that letter, taken in a fixed random order
(sha1 rank) and cut every 66 slices, so a shard mixes many deployments and a rung
is the union of groups A..R. Reads are 66 random ~1 MB files per shard from the
spinning drive; training then reads whole shards sequentially.

`StreamPool` is the training-side loader: a frame-level shuffle buffer in RAM
(size from free RAM, capped) filled from random shards, refreshed by one random
shard every `shard/batch` steps (a random set of buffer positions is
overwritten), batches drawn at random from the buffer. Each frame is read from
disk once and drawn ~once on average over its stay in the buffer; with the fixed
7000-step budget a rung-D run touches about a third of D.
"""
import argparse
import hashlib
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import dpaths as D  # noqa: E402
import store  # noqa: E402
GROUPS = 'ABCD'
SLICES_PER_SHARD = 66
slice_path = D.slice_path


def shard_root(cache):
    return os.path.join(cache, '_shards')


def _plan(cache):
    return pd.read_csv(os.path.join(cache, '_manifest', 'plan.csv'))


def pack(cache, rung, workers=8):
    root = shard_root(cache)
    os.makedirs(root, exist_ok=True)
    readme = os.path.join(root, 'README.md')
    if not os.path.exists(readme):
        open(readme, 'w').write(
            '# _shards\n\nTraining shards of the teacher-target cache (05_distill/shards.py), derived data:\n'
            'delete freely, `python 05_distill/shards.py pack --rung D` rebuilds. `<group>/s<k>.npz` holds '
            '66 slices (mel f16, code f16, logits f32) taken in random rank order from the slices whose '
            'plan first_rung is <group>; `<group>/index.json` lists files and frame counts.\n')
    plan = _plan(cache)
    for g in GROUPS[:GROUPS.index(rung) + 1]:
        d = os.path.join(root, g)
        rows = plan[plan['first_rung'] == g].copy()
        rows['h'] = [int(hashlib.sha1(f'{r}:{h}'.encode()).hexdigest()[:13], 16) for r, h in zip(rows.relpath, rows.hour)]
        rows = rows.sort_values('h')
        recs = list(rows.itertuples())
        # what this group's shards are derived from: the teacher build, its code width, the group's slice list
        fp = store.fingerprint({'group': g, 'code_dim': D.spec().code_dim, 'teacher_sha': store.teacher_sha(),
                                'slices': store.fingerprint([[r.relpath, int(r.hour)] for r in recs])})
        if os.path.exists(os.path.join(d, 'index.json')):
            have = json.load(open(os.path.join(d, 'index.json'))).get('fingerprint')
            if have in (None, fp):
                print(f'[shards {g}] index exists, skip', flush=True)
                continue
            print(f'[shards {g}] stale (fingerprint {have} != {fp}): rebuilding', flush=True)
            import shutil
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
        files, frames, t0 = [], [], time.time()
        pool = ThreadPoolExecutor(workers)

        def load(r):
            p = slice_path(cache, r.relpath, r.hour)
            if not os.path.exists(p):
                return None
            mp = store.mel_path('yamnet', r.relpath, r.hour, cache)
            if mp is None:
                return None
            with np.load(p) as z:
                return store.load_mel(mp), z['code'], z['logits']

        for k, s in enumerate(range(0, len(recs), SLICES_PER_SHARD)):
            fn = os.path.join(d, f's{k:05d}.npz')
            got = [x for x in pool.map(load, recs[s:s + SLICES_PER_SHARD]) if x is not None] \
                if not os.path.exists(fn) else None
            if got is not None and got:
                tmp = fn + '.tmp.npz'
                np.savez(tmp, mel=np.concatenate([x[0] for x in got]), code=np.concatenate([x[1] for x in got]),
                         logits=np.concatenate([x[2] for x in got]))
                os.replace(tmp, fn)
            if os.path.exists(fn):
                with np.load(fn) as z:
                    n = len(z['logits'])
                files.append(os.path.basename(fn))
                frames.append(int(n))
            if k % 20 == 0:
                print(f'[shards {g}] shard {k} of {-(-len(recs) // SLICES_PER_SHARD)}, {sum(frames)} frames, '
                      f'{time.time() - t0:.0f} s', flush=True)
        json.dump({'files': files, 'frames': frames, 'fingerprint': fp}, open(os.path.join(d, 'index.json'), 'w'))
        print(f'[shards {g}] {len(files)} shards, {sum(frames)} frames', flush=True)


def free_ram_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable'):
            return int(line.split()[1]) / 1e6
    return 8.0


class StreamPool:
    def __init__(self, cache, rung, buffer_gb=0, seed=0):
        root = shard_root(cache)
        self.files, self.frames = [], []
        for g in GROUPS[:GROUPS.index(rung) + 1]:
            idx = json.load(open(os.path.join(root, g, 'index.json')))
            self.files += [os.path.join(root, g, f) for f in idx['files']]
            self.frames += idx['frames']
        self.total = int(sum(self.frames))
        code_dim, n_cls = D.spec().code_dim, D.spec().n_classes
        per_frame = 96 * 64 * 2 + code_dim * 2 + n_cls * 4
        gb = buffer_gb or min(8.0, 0.4 * free_ram_gb())
        self.n = int(min(self.total, gb * 1e9 / per_frame))
        plan = _plan(cache)
        sel = plan[plan['first_rung'].isin(list(GROUPS[:GROUPS.index(rung) + 1]))]
        self.meta = {'slices': len(sel), 'deployments': int(sel['deployment'].nunique())}
        self.mel = np.empty((self.n, 96, 64), np.float16)
        self.code = np.empty((self.n, code_dim), np.float16)
        self.logits = np.empty((self.n, n_cls), np.float32)
        self._seed = seed
        self._q = None
        print(f'[stream] {len(self.files)} shards, {self.total} frames; shuffle buffer {self.n} frames '
              f'= {self.n * per_frame / 1e9:.1f} GB (free RAM {free_ram_gb():.1f} GB)', flush=True)
        self._start_reader(seed)
        self._fill()

    def _start_reader(self, seed):
        self._q = queue.Queue(3)

        def work():
            rng = np.random.default_rng(seed + 12345)
            while True:
                for i in rng.permutation(len(self.files)):
                    with np.load(self.files[i]) as z:
                        self._q.put((z['mel'], z['code'], z['logits']))

        threading.Thread(target=work, daemon=True).start()

    def _fill(self):
        pos, t0 = 0, time.time()
        while pos < self.n:
            m, c, l = self._q.get()
            k = min(len(l), self.n - pos)
            self.mel[pos:pos + k], self.code[pos:pos + k], self.logits[pos:pos + k] = m[:k], c[:k], l[:k]
            pos += k
        # the fill is in shard order (grouped by shard); a full-buffer random permutation is unnecessary
        # because batches sample buffer positions at random.
        print(f'[stream] buffer filled in {time.time() - t0:.0f} s', flush=True)

    def take(self, idx):
        idx = np.sort(idx)
        return (np.asarray(self.mel[idx], np.float32), np.asarray(self.code[idx], np.float32),
                np.asarray(self.logits[idx], np.float32))

    def code_stats(self, n=20000, seed=0):
        idx = np.sort(np.random.default_rng(seed).choice(self.n, min(n, self.n), replace=False))
        c = np.asarray(self.code[idx], np.float32)
        return c.mean(0), c.std(0) + 1e-3

    def batches(self, batch, steps, seed=0, depth=6):
        q = queue.Queue(depth)

        def work():
            rng = np.random.default_rng(seed)
            every = max(1, 4092 // batch)
            for s in range(steps):
                if s and s % every == 0:
                    m, c, l = self._q.get()
                    pos = rng.choice(self.n, len(l), replace=False) if len(l) <= self.n else np.arange(self.n)
                    self.mel[pos], self.code[pos], self.logits[pos] = m[:len(pos)], c[:len(pos)], l[:len(pos)]
                q.put(self.take(rng.choice(self.n, batch, replace=False)))
            q.put(None)

        threading.Thread(target=work, daemon=True).start()
        while True:
            b = q.get()
            if b is None:
                return
            yield b


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['pack'])
    ap.add_argument('--rung', required=True, choices=list(GROUPS))
    a = ap.parse_args()
    pack(D.need_cache(), a.rung)
