import argparse
import multiprocessing
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract import extract_set

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', default='lite', dest='setname')
    parser.add_argument('--embedder', default='yamnet')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    multiprocessing.set_start_method('fork', force=True)

    extract_set(
        setname=args.setname,
        embeddername=args.embedder,
        overlap_event_prop=0.2,
        framehop_prop=0.5,
        n_workers=args.workers,
    )
