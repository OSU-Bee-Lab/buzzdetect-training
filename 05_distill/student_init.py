"""Initialise the student from YAMNet's weights by output-channel selection.

Layer by layer, keep the student's `filters[i]` of YAMNet's channels:

  layer1_conv          keep the top-k of its 32 filters (L1 of the filter,
                       scaled by its BN's 1/sqrt(var + eps)).
  layer{i}_pointwise   keep the top-k output channels by the same score; the
                       kernel's input channels are the previous layer's kept
                       set, so the kept sub-network is a real sub-network of
                       YAMNet, not a re-wired one.
  layer{i}_depthwise   no choice: its channels are the previous layer's kept
                       set (kernel, BN follow).
  every BN             follows its layer's kept channels (YAMNet BN is
                       centre-only: beta, moving mean, moving var).
  logits head          not initialised: YAMNet's 521-class head has no
                       counterpart to the teacher's 15 classes (left at Keras'
                       default). distill_train fits it before step 0.

`init_from_yamnet(student)` works on either graph (mel or waveform) and returns
the kept-channel indices per layer. At width 1.0 the selection is the identity
and the student is YAMNet exactly (checked below).

    conda run -n buzzdetect-train python 05_distill/student_init.py --check

`--check` builds the alpha-0.5 mel student, initialises it, and reports (on real
mel patches from the eval folds' framed audio) how well its GAP code tracks the
matching YAMNet channels, against a randomly initialised student.
"""
import argparse
import glob
import os
import pickle
import sys

import tensorflow as tf  # noqa: F401  (must come first, see CLAUDE.md)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import student as st  # noqa: E402

YAMNET_KERAS = os.path.join(st.main_checkout(), 'embedders', 'yamnet', 'yamnet.keras')
BN_EPS = 1e-4


def load_yamnet():
    import keras  # noqa: F401
    from embedders.yamnet.yamnet import WaveformFeatures  # noqa: F401  (registers the layer)
    return tf.keras.models.load_model(YAMNET_KERAS, compile=False)


def _topk(score, k):
    """Indices of the k largest scores, kept in ascending channel order."""
    return np.sort(np.argsort(-score, kind='stable')[:k])


def _score(kernel, moving_var):
    """L1 of each output filter, after BN's 1/sqrt(var + eps) folded in."""
    l1 = np.abs(kernel).reshape(-1, kernel.shape[-1]).sum(0)
    return l1 / np.sqrt(moving_var + BN_EPS)


def select_channels(yamnet, filters):
    """Kept YAMNet output-channel indices per layer, list of len(filters)."""
    sels = []
    for i, k in enumerate(filters):
        name = 'layer1_conv' if i == 0 else f'layer{i + 1}_pointwise_conv'
        kernel = yamnet.get_layer(name).get_weights()[0]
        bn = yamnet.get_layer(name + '_bn').get_weights()  # beta, mean, var
        assert k <= kernel.shape[-1]
        sels.append(_topk(_score(kernel, bn[2]), k))
    return sels


def init_from_yamnet(student, yamnet=None, filters=None):
    """Set the student's conv-stack weights from YAMNet by channel selection.

    Returns the per-layer kept indices (into YAMNet's channels).
    """
    yamnet = yamnet or load_yamnet()
    if filters is None:
        filters, i = [], 0
        while _has(student, i):
            filters.append(student.get_layer(f'layer{i + 1}_' + ('conv' if i == 0 else 'pointwise_conv')).filters)
            i += 1
    sels = select_channels(yamnet, filters)

    def bn_set(name, sel):
        beta, mean, var = yamnet.get_layer(name).get_weights()
        student.get_layer(name).set_weights([beta[sel], mean[sel], var[sel]])

    for i, sel in enumerate(sels):
        if i == 0:
            kernel = yamnet.get_layer('layer1_conv').get_weights()[0]
            student.get_layer('layer1_conv').set_weights([kernel[..., sel]])
            bn_set('layer1_conv_bn', sel)
            continue
        prev = sels[i - 1]
        n = f'layer{i + 1}'
        dw = yamnet.get_layer(n + '_depthwise_conv').get_weights()[0]
        student.get_layer(n + '_depthwise_conv').set_weights([dw[:, :, prev, :]])
        bn_set(n + '_depthwise_conv_bn', prev)
        pw = yamnet.get_layer(n + '_pointwise_conv').get_weights()[0]
        student.get_layer(n + '_pointwise_conv').set_weights([pw[:, :, prev, :][..., sel]])
        bn_set(n + '_pointwise_conv_bn', sel)
    return sels


def recalibrate_bn(student, mel, batch=256):
    """Re-estimate every BN's moving mean/var on real mel patches, layer by layer.

    Dropping half a layer's input channels shifts every downstream
    pre-activation, so YAMNet's own BN statistics no longer normalise it and the
    stack collapses toward all-zero after ReLU. Statistics are re-measured in
    order (each BN sees the already-recalibrated layers before it); beta is
    kept. No gradient step, just moments.
    """
    bns = [l for l in student.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    for bn in bns:
        probe = tf.keras.Model(student.input, bn.input)
        n, s1, s2 = 0, 0.0, 0.0
        for i in range(0, len(mel), batch):
            x = probe(mel[i:i + batch], training=False).numpy().astype(np.float64)
            x = x.reshape(-1, x.shape[-1])
            n += len(x)
            s1 = s1 + x.sum(0)
            s2 = s2 + (x ** 2).sum(0)
        mean = s1 / n
        var = np.maximum(s2 / n - mean ** 2, 1e-8)
        beta = bn.get_weights()[0]
        bn.set_weights([beta, mean.astype(np.float32), var.astype(np.float32)])


def _has(student, i):
    try:
        student.get_layer(f'layer{i + 1}_' + ('conv' if i == 0 else 'pointwise_conv'))
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------- check

def frames_from_folds(n_frames, seed=0):
    """Real 15360-sample frames from the eval folds' framed-audio cache."""
    root = os.path.join(st.main_checkout(), '02_set', 'sets', 'moderate', 'audio',
                        'sr16000_fl0.96', 'raw')
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(os.path.join(root, '**', '*.pickle'), recursive=True))
    rng.shuffle(files)
    out = []
    for f in files:
        with open(f, 'rb') as h:
            while True:
                try:
                    out.append(np.asarray(pickle.load(h), dtype=np.float32))
                except EOFError:
                    break
        if len(out) >= n_frames:
            break
    return np.stack(out[:n_frames])


def mel_of(frames):
    fe = st.build_student([8], stop_at_features=True)
    return np.concatenate([fe(f, training=False).numpy() for f in frames])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--frames', type=int, default=1500)
    a = ap.parse_args()
    yam = load_yamnet()
    mel = mel_of(frames_from_folds(a.frames))
    print('mel patches', mel.shape, flush=True)

    # YAMNet's own code on the same patches: the width-1.0 student IS YAMNet.
    full = st.build_student(st.widths_for(1.0), input_type='mel', expose_code=True, name='full')
    sels = init_from_yamnet(full, yam)
    _, code_full = full(mel[:64], training=False)
    ref = yam  # waveform -> 1024-d embeddings
    frames = frames_from_folds(8)
    e_ref = np.concatenate([ref(f, training=False).numpy()[:1] for f in frames])
    _, e_full = full(mel_of(frames), training=False)
    print(f'width-1.0 identity check: max |student - yamnet| GAP code = '
          f'{np.abs(e_ref - e_full.numpy()).max():.2e}', flush=True)

    filters = st.widths_for(a.alpha)
    stud = st.build_student(filters, input_type='mel', expose_code=True, name='stud')
    sels = init_from_yamnet(stud, yam)
    c_raw = np.concatenate([stud(mel[i:i + 256], training=False)[1].numpy() for i in range(0, len(mel), 256)])
    calib = mel[::3]  # every third patch; the stats need no more
    recalibrate_bn(stud, calib)
    rand = st.build_student(filters, input_type='mel', expose_code=True, name='rand')

    def codes(model):
        return np.concatenate([model(mel[i:i + 256], training=False)[1].numpy()
                               for i in range(0, len(mel), 256)])

    c_stud, c_rand, c_full = codes(stud), codes(rand), codes(full)
    c_ref = c_full[:, sels[-1]]  # YAMNet's own GAP channels that the student kept

    def pearson(a_, b_):
        a_ = a_ - a_.mean(0)
        b_ = b_ - b_.mean(0)
        den = np.sqrt((a_ ** 2).sum(0) * (b_ ** 2).sum(0)) + 1e-12
        return (a_ * b_).sum(0) / den

    # per-layer diagnostic: student layer-i output vs the matching YAMNet channels
    lay = []
    for i in range(len(filters)):
        nm = f'layer{i + 1}_' + ('relu' if i == 0 else 'pointwise_conv_relu')
        ps = tf.keras.Model(stud.input, stud.get_layer(nm).output)
        pf = tf.keras.Model(full.input, full.get_layer(nm).output)
        x = mel[:256]
        u, v = ps(x, training=False).numpy(), pf(x, training=False).numpy()[..., sels[i]]
        u, v = u.reshape(-1, u.shape[-1]), v.reshape(-1, v.shape[-1])
        lay.append(f'{np.nanmean(np.nan_to_num(pearson(u, v))):.2f}')
    print('  per-layer mean Pearson (post-ReLU, layers 1..14):', ' '.join(lay))
    r_raw = pearson(c_raw, c_ref)
    print(f'  (before BN recalibration: mean Pearson {r_raw.mean():.3f}, '
          f'{(c_raw.std(0) < 1e-6).mean():.0%} of channels constant)')
    r_stud, r_rand = pearson(c_stud, c_ref), pearson(c_rand, c_ref)
    print(f'alpha {a.alpha}: per-channel Pearson of student GAP code vs the matching YAMNet channel')
    print(f'  init     mean {r_stud.mean():.3f}  median {np.median(r_stud):.3f}  '
          f'p10 {np.percentile(r_stud, 10):.3f}')
    print(f'  random   mean {r_rand.mean():.3f}  median {np.median(r_rand):.3f}')

    # Linear readout: how much of the FULL 1024-d YAMNet code a ridge map from
    # the student's code recovers, held-out (70/30 split by frame).
    n = len(c_full)
    tr, te = slice(0, int(n * 0.7)), slice(int(n * 0.7), n)

    def r2(x):
        x1 = np.c_[x, np.ones(len(x))]
        w = np.linalg.solve(x1[tr].T @ x1[tr] + 1e-2 * np.eye(x1.shape[1]), x1[tr].T @ c_full[tr])
        res = ((x1[te] @ w - c_full[te]) ** 2).sum()
        tot = ((c_full[te] - c_full[tr].mean(0)) ** 2).sum()
        return 1 - res / tot

    print(f'  held-out R^2 of ridge(student code) -> full YAMNet code: '
          f'init+recal {r2(c_stud):.3f}  random {r2(c_rand):.3f}')


if __name__ == '__main__':
    main()
