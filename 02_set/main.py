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
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    multiprocessing.set_start_method('fork', force=True)

    extract_set(
        setname=args.setname,
        embeddername=args.embedder,
        n_workers=args.workers,
        verbose=args.verbose,
    )
