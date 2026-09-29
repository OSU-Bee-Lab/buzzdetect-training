"""Front-end parity: numpy `mel_patches` (the cache builder) vs the Keras layer (the deployed graph),
and both vs YAMNet's own WaveformFeatures for the 'yamnet' spec.

    conda run -n buzzdetect-train python 05_distill/test_frontends.py [--seconds 20]

Uses the 04_deploy fixture audio (real recording). Tolerance 1e-3 on log-mel (float32 FFT differences).
"""
import argparse
import os
import sys

import tensorflow as tf  # noqa: F401  (must come first, see CLAUDE.md)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import frontends as fes  # noqa: E402
import student as st  # noqa: E402

FIXTURE = os.path.join(ROOT, '04_deploy', 'fixtures', '230808_1208_s89520.flac')
TOL = 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=20)
    a = ap.parse_args()
    import librosa
    x, _ = librosa.load(FIXTURE, sr=16000, mono=True)
    # whole 0.96 s patches plus the window's tail, so numpy and the padded graph see the same patches
    x = x[:int(a.seconds * 16000)].astype(np.float32)
    bad = 0
    for name, fe in fes.FRONTENDS.items():
        n_np = fes.mel_patches(x, fe)
        m = st.build_student([8], stop_at_features=True, frontend=name)
        n_tf = m(x, training=False).numpy() if name != 'yamnet' else \
            m(x, training=False).numpy()[..., None]
        n = min(len(n_np), len(n_tf))
        if name == 'yamnet':
            n_np = n_np[:n]
        d = float(np.abs(n_np[:n] - n_tf[:n]).max())
        ok = d < TOL
        bad += not ok
        print(f'{name:8s} numpy {n_np.shape} keras {n_tf.shape}  max|diff| over {n} patches = {d:.2e}  '
              f'{"OK" if ok else "FAIL"}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
