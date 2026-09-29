"""Where in frequency does the teacher's buzz call live? (informs front-end band ranges)

    conda run -n buzzdetect-train python 05_distill/band_profile.py [--shards V] [--n 80000]

On the validation-pool shards (YAMNet's 64-band log-mel + the teacher's cached logits), per
YAMNet band: mean log-mel of frames the teacher calls ins_buzz (logit > 0) vs frames where it
calls no live class, and the standardised difference d. Frames are averaged over their 96
time steps first. The teacher-defined labels are the point: this is what the student has to
reproduce, not ground truth. Also prints the share of the total |d| mass below/above cut-offs.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dpaths as D  # noqa: E402
import frontends as fes  # noqa: E402

SP = D.spec()
BUZZ = SP.buzz
LIVE_OTHER = [i for i in SP.live if i != BUZZ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', default='V')
    ap.add_argument('--n', type=int, default=80000)
    a = ap.parse_args()
    d = os.path.join(D.SHARDS, a.shards)
    import json
    n_all = json.load(open(os.path.join(d, 'meta.json')))['frames']
    mel = np.load(os.path.join(d, 'mel.npy'), mmap_mode='r')
    lg = np.load(os.path.join(d, 'logits.npy'), mmap_mode='r')
    idx = np.sort(np.random.default_rng(0).choice(n_all, min(a.n, n_all), replace=False))
    lg = np.asarray(lg[idx])
    pos = lg[:, BUZZ] > 0
    neg = ~pos & ~(lg[:, LIVE_OTHER] > 0).any(1)
    prof = np.asarray(mel[idx], np.float32).mean(1)          # (N, 64): mean over the 96 time steps
    p, q = prof[pos], prof[neg]
    print(f'{len(idx)} frames sampled: buzz-positive {pos.sum()}, clean-negative {neg.sum()}')
    sd = np.sqrt((p.var(0) + q.var(0)) / 2)
    dd = (p.mean(0) - q.mean(0)) / sd
    hz = fes.band_centers_hz(fes.get('yamnet').channels[0])
    print(f'{"band":>4s} {"Hz":>6s} {"buzz":>7s} {"neg":>7s} {"d":>6s}')
    for i in range(64):
        print(f'{i:4d} {hz[i]:6.0f} {p.mean(0)[i]:7.2f} {q.mean(0)[i]:7.2f} {dd[i]:6.2f}')
    w = np.abs(dd)
    for cut in (500, 1000, 2000, 2500, 4000):
        print(f'share of |d| mass below {cut} Hz: {w[hz < cut].sum() / w.sum():.2f}')


if __name__ == '__main__':
    main()
