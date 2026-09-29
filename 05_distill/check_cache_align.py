"""Check that frontends.mel_patches('yamnet') on a freshly decoded slice reproduces the main
cache's stored YAMNet `mel` (so a new front end's patches line up with the teacher targets).

    .local/venv-onnx/bin/python 05_distill/check_cache_align.py [--n 3]
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import config  # noqa: E402
import cache as C  # noqa: E402
import frontends as fes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=3)
    a = ap.parse_args()
    cache = config.DISTILL_CACHE
    plan = [r for r in csv.DictReader(open(os.path.join(cache, '_manifest', 'plan.csv')))
            if r['first_rung'] == 'A'][:a.n]
    dur = {r['relpath']: (int(r['size']), float(r['dur']))
           for r in csv.DictReader(open(os.path.join(cache, '_manifest', 'durations.csv'))) if r['dur']}
    for r in plan:
        x = C.load_slice(config.AUDIO_ROOT, r['relpath'], float(r['start_s']), dur.get(r['relpath']))
        if len(x) < 953600:
            x = np.concatenate([x, np.zeros(953600 - len(x), np.float32)])
        z = np.load(C.out_path(cache, r['relpath'], r['hour']))
        stored = z['mel'].astype(np.float32)
        mine = fes.mel_patches(x, fes.get('yamnet'))[..., 0]
        n = min(len(stored), len(mine))
        print(f'{r["relpath"]} h{r["hour"]}: stored {stored.shape} mine {mine.shape} '
              f'max|diff| {np.abs(stored[:n] - mine[:n]).max():.3e} (fp16 mel, |mel| mean {np.abs(stored).mean():.2f})')


if __name__ == '__main__':
    main()
