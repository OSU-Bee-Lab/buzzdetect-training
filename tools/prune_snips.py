"""Delete a set's snips for idents no longer in its annotations.csv.

Stage 2 keeps these on purpose: an effort switched off in build.R and later
switched back on would otherwise have to be re-cut from the source drive. Run
this when the disk space is wanted more than that. Framed audio and embeddings
of dropped idents are already removed by every extraction run.

    python tools/prune_snips.py moderate            # list what would go
    python tools/prune_snips.py moderate --delete   # delete it

No TensorFlow, no pandas.
"""

import argparse
import csv
import os
import shutil
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config as cfg  # noqa: E402


def find_orphans(setname):
    """(path, ident) of every snip directory whose ident is not in annotations.csv.

    Same walk as extract._find_snip_dirs: idents are path-like, so a snip
    directory sits at an arbitrary depth and is recognised by holding snips.
    """
    with open(os.path.join(cfg.dir_set(setname), 'annotations.csv'), newline='') as f:
        known = {row['ident'] for row in csv.DictReader(f)}
    base = cfg.dir_snips(setname)
    orphans = []
    for dirpath, _, filenames in os.walk(base):
        if not any(n.startswith('snip_') and n.endswith('.flac') for n in filenames):
            continue
        ident = os.path.relpath(dirpath, base).replace(os.sep, '/')
        if ident not in known:
            orphans.append((dirpath, ident))
    return sorted(orphans)


def _size(path):
    return sum(os.path.getsize(os.path.join(dp, n)) for dp, _, fn in os.walk(path) for n in fn)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('setname')
    ap.add_argument('--delete', action='store_true', help='delete (default: list only)')
    args = ap.parse_args()

    orphans = find_orphans(args.setname)
    if not orphans:
        print(f'{args.setname}: no orphaned snips')
        return

    by_effort = Counter(ident.split('/')[0] for _, ident in orphans)
    n_bytes = sum(_size(path) for path, _ in orphans)
    print(f'{args.setname}: {len(orphans)} ident(s) with orphaned snips, {n_bytes / 1e9:.2f} GB')
    for effort, n in by_effort.most_common():
        print(f'  {n:6d}  {effort}')

    if not args.delete:
        print('dry run; pass --delete to remove them')
        return

    for path, _ in orphans:
        shutil.rmtree(path)
    print(f'deleted {len(orphans)} snip dir(s)')


if __name__ == '__main__':
    main()
