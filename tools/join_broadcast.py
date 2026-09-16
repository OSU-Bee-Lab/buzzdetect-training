"""IDEAS item 2a: broadcast-join a coarse embedder's cache onto a fine one's
frame grid, without re-extracting either.

Both embedders were extracted from the same annotations, so they share the
exact same fold/ident directory tree under raw/; only the frame grid inside
each leaf directory differs (chunk boundaries depend on framelength_s and
overlap_event_s, so the fine embedder's frames are not a clean j -> j//k
subsample of the coarse one's). frametimes.csv (label, row, start) in each
leaf directory is the only durable map from an embedding row to a source-file
timestamp, so it -- not the pickle filenames, which differ because coarser
frames collapse more/different overlapping annotations into one label -- is
the join key.

For every fine-grid frame, finds the coarse frame whose interval covers the
fine frame's centre (falling back to nearest-centre if none does, e.g. at a
snip edge) and concatenates [fine_vec, coarse_vec]. Writes a brand new
embedder cache (same fold/ident tree, same pickle names and row order as the
fine embedder, so it's a drop-in --embedder name for 03_train) under
02_set/sets/<set>/embeddings/<out_name>/raw/.
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

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


def load_leaf(dir_leaf):
    """label -> list of vectors (row order), plus frametimes df."""
    frametimes = pd.read_csv(os.path.join(dir_leaf, 'frametimes.csv'))
    vecs_by_label = {}
    for p in glob.glob(os.path.join(dir_leaf, '*.pickle')):
        label = os.path.basename(p)[:-len('.pickle')]
        vecs_by_label[label] = read_pickle_exhaustive(p)
    return frametimes, vecs_by_label


def nearest_coarse(fine_start, fine_len, coarse_starts, coarse_len):
    """Index into coarse_starts (sorted ascending) whose frame covers, or is
    nearest to, the fine frame's centre."""
    centre = fine_start + fine_len / 2.0
    coarse_centres = coarse_starts + coarse_len / 2.0
    idx = np.searchsorted(coarse_starts, centre)
    candidates = [i for i in (idx - 1, idx) if 0 <= i < len(coarse_starts)]
    best = min(candidates, key=lambda i: abs(coarse_centres[i] - centre))
    return best


def join_ident(dir_fine_leaf, dir_coarse_leaf, dir_out_leaf, fine_len, coarse_len, stats):
    frametimes_fine, vecs_fine = load_leaf(dir_fine_leaf)
    frametimes_coarse, vecs_coarse = load_leaf(dir_coarse_leaf)

    # flatten coarse into one time-sorted array for searchsorted
    coarse_rows = frametimes_coarse.sort_values('start').reset_index(drop=True)
    coarse_starts = coarse_rows['start'].to_numpy()
    coarse_vecs = np.stack([
        vecs_coarse[row.label][row.row] for row in coarse_rows.itertuples()
    ])
    coarse_dim = coarse_vecs.shape[1]
    zero_coarse = np.zeros(coarse_dim, dtype=np.float32)

    # A fine frame can only genuinely fall inside some coarse frame if their
    # centres are within half a coarse frame plus half a fine frame of each
    # other. Perch discards any annotation-cluster snip under 5 s outright
    # (see extract_ident_both's "snip is shorter than one frame" skip), so an
    # isolated short buzz event can have no coarse frame anywhere nearby --
    # nearest_coarse would otherwise silently splice in whatever coarse frame
    # happens to be least-far-away, sometimes hours off (measured: 2.7% of
    # medium's frames, up to 3601 s). Zero-fill the coarse block instead of
    # joining across a gap that isn't a real temporal relationship.
    max_valid_gap = coarse_len / 2.0 + fine_len / 2.0

    os.makedirs(dir_out_leaf, exist_ok=True)
    for label, vecs in vecs_fine.items():
        rows = frametimes_fine[frametimes_fine['label'] == label].sort_values('row')
        assert list(rows['row']) == list(range(len(vecs))), \
            f'{dir_fine_leaf}: {label} frametimes rows not 0..n-1 in order'
        out_path = os.path.join(dir_out_leaf, label + '.pickle')
        with open(out_path, 'wb') as f:
            for row, vec in zip(rows.itertuples(), vecs):
                j = nearest_coarse(row.start, fine_len, coarse_starts, coarse_len)
                gap = abs((coarse_starts[j] + coarse_len / 2.0) - (row.start + fine_len / 2.0))
                stats['n'] += 1
                if gap <= max_valid_gap:
                    stats['gap_sum'] += gap
                    stats['gap_max'] = max(stats['gap_max'], gap)
                    coarse_vec = coarse_vecs[j]
                else:
                    stats['n_zerofilled'] += 1
                    coarse_vec = zero_coarse
                combined = np.concatenate([vec, coarse_vec]).astype(np.float32)
                pickle.dump(combined, f)

    frametimes_fine.to_csv(os.path.join(dir_out_leaf, 'frametimes.csv'), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='medium')
    ap.add_argument('--fine', required=True, help='fine-grid (base) embedder name')
    ap.add_argument('--coarse', required=True, help='coarse embedder name to broadcast')
    ap.add_argument('--out', required=True, help='output embedder name')
    args = ap.parse_args()

    dir_fine_raw = cfg.dir_embeddings_raw(args.set, args.fine)
    dir_coarse_raw = cfg.dir_embeddings_raw(args.set, args.coarse)
    dir_out_raw = cfg.dir_embeddings_raw(args.set, args.out)

    from embedders.embedding import load_embedder
    fine_len = load_embedder(args.fine, framehop_prop=1.0, initialize=False).framelength_s
    coarse_len = load_embedder(args.coarse, framehop_prop=1.0, initialize=False).framelength_s

    leaves = []
    for dirpath, _, filenames in os.walk(dir_fine_raw):
        if 'frametimes.csv' in filenames:
            leaves.append(os.path.relpath(dirpath, dir_fine_raw))

    print(f'{len(leaves)} leaf dirs to join', flush=True)
    stats = {'n': 0, 'gap_sum': 0.0, 'gap_max': 0.0, 'n_zerofilled': 0}
    missing = []
    for i, rel in enumerate(leaves):
        dir_fine_leaf = os.path.join(dir_fine_raw, rel)
        dir_coarse_leaf = os.path.join(dir_coarse_raw, rel)
        if not os.path.exists(os.path.join(dir_coarse_leaf, 'frametimes.csv')):
            missing.append(rel)
            continue
        dir_out_leaf = os.path.join(dir_out_raw, rel)
        join_ident(dir_fine_leaf, dir_coarse_leaf, dir_out_leaf, fine_len, coarse_len, stats)
        print(f'[{i+1}/{len(leaves)}] {rel}', flush=True)

    if missing:
        print(f'WARNING: {len(missing)} leaf dirs had no matching coarse frametimes.csv, skipped:')
        for m in missing:
            print(f'  {m}')

    n_matched = stats['n'] - stats['n_zerofilled']
    print(f'joined {stats["n"]} frames; {stats["n_zerofilled"]} zero-filled '
          f'({100*stats["n_zerofilled"]/max(stats["n"],1):.2f}%); '
          f'mean centre-gap (matched only) {stats["gap_sum"]/max(n_matched,1):.3f}s, '
          f'max {stats["gap_max"]:.3f}s')


if __name__ == '__main__':
    main()
