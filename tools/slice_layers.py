"""Slice a subset of columns out of an already-extracted embedder's cache and
write it as a new embedder name -- pure numpy, no re-extraction.

Used to build yamnet_pitchshift_aves_l345 out of yamnet_pitchshift_aves_all12
(IDEAS item 20a's follow-up): same frame grid, so unlike
tools/join_broadcast.py this needs no timestamp matching, just a column slice
per pickle.
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


def read_pickle_exhaustive(path):
    out = []
    with open(path, 'rb') as f:
        while True:
            try:
                out.append(pickle.load(f))
            except EOFError:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='medium')
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cols', required=True,
                     help='comma-separated ranges into the source vector, e.g. "0:2048,3584:6144"')
    args = ap.parse_args()

    cols = []
    for part in args.cols.split(','):
        lo, hi = part.split(':')
        cols.extend(range(int(lo), int(hi)))
    cols = np.array(cols)

    dir_source = cfg.dir_embeddings_raw(args.set, args.source)
    dir_out = cfg.dir_embeddings_raw(args.set, args.out)

    n_leaves = 0
    n_frames = 0
    for dirpath, _, filenames in os.walk(dir_source):
        pickles = [f for f in filenames if f.endswith('.pickle')]
        if not pickles and 'frametimes.csv' not in filenames:
            continue
        rel = os.path.relpath(dirpath, dir_source)
        dir_leaf_out = os.path.join(dir_out, rel)
        os.makedirs(dir_leaf_out, exist_ok=True)
        for name in filenames:
            src = os.path.join(dirpath, name)
            dst = os.path.join(dir_leaf_out, name)
            if name == 'frametimes.csv':
                import shutil
                shutil.copyfile(src, dst)
                continue
            if not name.endswith('.pickle'):
                continue
            vecs = read_pickle_exhaustive(src)
            with open(dst, 'wb') as f:
                for v in vecs:
                    pickle.dump(v[cols].astype(np.float32), f)
            n_frames += len(vecs)
        n_leaves += 1
        if n_leaves % 10 == 0:
            print(f'{n_leaves} leaf dirs, {n_frames} frames so far', flush=True)

    print(f'done: {n_leaves} leaf dirs, {n_frames} frames, {len(cols)}-d output')


if __name__ == '__main__':
    main()
