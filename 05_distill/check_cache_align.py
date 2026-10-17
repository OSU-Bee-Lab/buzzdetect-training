"""Check that frontends.mel_patches('yamnet') on a freshly decoded slice reproduces the cache's stored
YAMNet `mel` (so a new front end's patches line up with the teacher targets).

    .local/venv-onnx/bin/python 05_distill/check_cache_align.py [--n 3] [--renamed]

--renamed checks one slice of each file in dpaths.RENAMED instead: a renamed file that is the audio that
was planned reproduces its stored mel.
"""
import argparse
import csv
import sys
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dpaths as D  # noqa: E402
import store  # noqa: E402
import cache as C  # noqa: E402
import frontends as fes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--renamed', action='store_true', help='one slice per renamed file (dpaths.RENAMED)')
    a = ap.parse_args()
    if a.renamed:
        old = {r['old_relpath'] for r in csv.DictReader(open(D.RENAMED))}
        plan = list({r['relpath']: r for r in csv.DictReader(open(D.PLAN))
                     if r['relpath'] in old and store.mel_path('yamnet', r['relpath'], r['hour'])}.values())
        print(f'{len(plan)} of {len(old)} renamed files have a cached slice to check')
    else:
        plan = [r for r in csv.DictReader(open(D.PLAN)) if r['first_rung'] == 'A'][:a.n]
    dur = store.load_durations()
    for r in plan:
        x = C.load_slice(D.AUDIO_ROOT, r['relpath'], float(r['start_s']), dur.get(r['relpath']))
        if len(x) < D.SLICE_SAMPLES:
            x = np.concatenate([x, np.zeros(D.SLICE_SAMPLES - len(x), np.float32)])
        stored = store.load_mel(store.mel_path('yamnet', r['relpath'], r['hour'])).astype(np.float32)
        mine = fes.mel_patches(x, fes.get('yamnet'))[..., 0]
        n = min(len(stored), len(mine))
        print(f'{r["relpath"]} h{r["hour"]}: stored {stored.shape} mine {mine.shape} '
              f'max|diff| {np.abs(stored[:n] - mine[:n]).max():.3e} (fp16 mel, |mel| mean {np.abs(stored).mean():.2f})')


if __name__ == '__main__':
    main()
