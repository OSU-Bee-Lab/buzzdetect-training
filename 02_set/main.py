# TensorFlow must be imported before pandas/pyarrow. pandas eagerly imports
# pyarrow, and pyarrow + TF each bundle their own statically-linked abseil; the
# shared lib that loads first wins abseil's weak synchronization symbols
# process-wide. If libarrow wins, TF's in-graph FFT (ducc0) threadpool ends up
# waiting on libarrow's incompatible semaphore impl and deadlocks — YAMNet
# extraction freezes mid-run. Keep this import first, ahead of anything (incl.
# extract.py) that pulls in pandas.
import tensorflow  # noqa: F401  -- imported for load-order side effect only

import argparse
import multiprocessing
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract import extract_set

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', required=True, dest='setname')
    parser.add_argument('--embedder', required=True)
    parser.add_argument('--workers', type=int, required=True,
                        help='processes for the framing+embedding phase (VRAM-bound)')
    parser.add_argument('--snip-workers', type=int, default=4, dest='snip_workers',
                        help='threads for the snip-sync phase (source-drive I/O; no GPU); '
                             '1 = serial')
    # Only consulted when the set has no config_extract.json yet — a set that
    # already has one keeps it, and these are ignored with a warning. Changing
    # a set's extraction params means deleting that file and re-extracting.
    parser.add_argument('--overlap-event-prop', type=float, default=None,
                        dest='overlap_event_prop',
                        help='annotation overlap required to label a frame '
                             '(ignored if the set already has a config_extract.json)')
    parser.add_argument('--framehop-prop', type=float, default=None,
                        dest='framehop_prop',
                        help='frame hop as a proportion of frame length; 1 = no '
                             'overlap (ignored if the set already has a config_extract.json)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    multiprocessing.set_start_method('fork', force=True)

    extract_set(
        setname=args.setname,
        embeddername=args.embedder,
        overlap_event_prop=args.overlap_event_prop,
        framehop_prop=args.framehop_prop,
        n_workers=args.workers,
        snip_workers=args.snip_workers,
        verbose=args.verbose,
    )
