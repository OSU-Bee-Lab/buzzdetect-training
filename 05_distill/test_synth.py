"""Synthetic-data test of distill_train, no cache needed.

Real mel patches (framed audio of the eval folds), synthetic teacher: YAMNet's
own GAP code (1024-d, tiled to 2048) and a fixed random linear head on it, with
the two dead classes' logits offset high. Writes train/val shards under
/tmp/dl/synth and prints the command that trains on them:

    conda run -n buzzdetect-train python 05_distill/test_synth.py [--n 6000]
"""
import argparse
import json
import os
import sys

import tensorflow as tf  # noqa: F401
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import student as st  # noqa: E402
import student_init as si  # noqa: E402


def write(d, mel, model_full, head):
    os.makedirs(d, exist_ok=True)
    n = len(mel)
    np.save(os.path.join(d, 'mel.npy'), mel.astype(np.float16))
    gap = np.concatenate([model_full(mel[i:i + 256].astype(np.float32), training=False)[1].numpy()
                          for i in range(0, n, 256)])
    code = np.concatenate([gap, gap], 1).astype(np.float16)
    lg = (gap @ head[0] + head[1]).astype(np.float32)
    lg[:, [0, 14]] = 5.0
    np.save(os.path.join(d, 'code.npy'), code)
    np.save(os.path.join(d, 'logits.npy'), lg)
    json.dump({'frames': n, 'slices': n // 62, 'missing_slices': 0, 'capacity': n,
               'slice_index': [], 'deployments': 3}, open(os.path.join(d, 'meta.json'), 'w'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6000)
    a = ap.parse_args()
    mel = si.mel_of(si.frames_from_folds(a.n + 1500, seed=3))
    full = st.build_student(st.widths_for(1.0), input_type='mel', expose_code=True)
    si.init_from_yamnet(full, si.load_yamnet())
    rng = np.random.default_rng(0)
    head = (rng.normal(0, 0.15, (1024, 15)).astype(np.float32), rng.normal(0, 1, 15).astype(np.float32))
    write('/tmp/dl/synth/train', mel[:a.n], full, head)
    write('/tmp/dl/synth/val', mel[a.n:], full, head)
    print('wrote /tmp/dl/synth/{train,val}; train with:\n  python 05_distill/distill_train.py '
          '--rung A --steps 300 --batch 256 --eval-every 100 --name synth '
          '--shards /tmp/dl/synth/train --val-shards /tmp/dl/synth/val')
