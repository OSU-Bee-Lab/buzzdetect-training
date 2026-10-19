"""Front-end specs for the distillation student: which spectrogram the student sees.

YAMNet's front end is one 25 ms-window, 64-band, 125-7500 Hz log-mel. A `Frontend`
here is one or more such channels, each with its own STFT window, band count and
band range, stacked as the student's input channels: (96 frames, bands, channels).
Every channel has the same number of bands (they share the conv's frequency axis),
the same 10 ms hop and the same 96 frames per 0.96 s patch, so the student's
layers 2-14, the head and the frame layout are unchanged; only layer 1's input
channel count and the front end itself differ. Register new specs in `FRONTENDS`.

Two implementations of the same math, checked against each other by
`test_frontends.py`:
  - `mel_patches(x, fe)`            numpy, whole slice -> (n,96,B,C): the cache builder.
  - `FrontendFeatures(fe)` (Keras)  waveform -> (n,96,B,C): the deployed graph.
Both take a continuous STFT of the whole waveform and cut it into 96-frame patches
(patch k starts at sample k*15360, as the deployed graph does), and both use
`tf.signal.linear_to_mel_weight_matrix`'s definition of the mel matrix.

    python 05_distill/frontends.py            # describe every spec (Hz per band, empty bands)
"""
import os
import sys
from dataclasses import dataclass

import numpy as np

SR = 16000
HOP = 160                     # 10 ms: YAMNet's hop, the default
PATCH_SAMPLES = 15360         # one 0.96 s frame of the deployed graph
LOG_OFFSET = 0.001


@dataclass(frozen=True)
class Channel:
    window: int               # STFT window, samples
    bands: int
    fmin: float
    fmax: float
    fft: int = 0              # 0 -> next power of two >= window

    stride = 1                # class attribute, not a field: a SparseChannel sets it (keeps old specs' stamps)

    @property
    def n_fft(self):
        return self.fft or 2 ** int(np.ceil(np.log2(self.window)))

    def mel(self):
        """This channel's (n_fft/2+1, bands) mel matrix; a SparseChannel zeroes the bins it does not read."""
        return mel_matrix(self.n_fft, self.bands, self.fmin, self.fmax)


@dataclass(frozen=True)
class SparseChannel(Channel):
    """A Channel whose DFT is read on a coarser grid where its mel bands are wide: bin b is kept if it lies at
    least min(stride, floor(band spacing at b / bin width)) bins past the last kept one. The triangles are
    the same, evaluated on the kept bins only, each weighted by the bins it stands for (so a band's sum
    approximates the full grid's); the windowed-DFT MatMul (the front end's FLOP-bound part,
    mmt-overlap) shrinks with the kept-bin count, every other op is unchanged."""
    stride: int = 2

    def mel(self):
        mm = mel_matrix(self.n_fft, self.bands, self.fmin, self.fmax)
        hz = np.linspace(0.0, SR / 2.0, len(mm))
        dmel = (_hz_to_mel(self.fmax) - _hz_to_mel(self.fmin)) / (self.bands + 1)
        spacing = (700.0 + hz) / 1127.0 * dmel                               # band-centre spacing, Hz
        keep, last = np.zeros(len(mm), bool), -len(mm)
        for b in np.flatnonzero(mm.sum(1) > 0):
            if b - last >= np.clip(np.floor(spacing[b] / (SR / self.n_fft)), 1, self.stride):
                keep[b], last = True, b
        kept = np.flatnonzero(keep)
        mm[kept] *= np.diff(np.append(kept, kept[-1] + 1))[:, None]        # a kept bin stands for the gap to the next
        mm[~keep] = 0.0
        return mm


@dataclass(frozen=True)
class Frontend:
    name: str
    channels: tuple
    hop: int = HOP            # STFT hop, samples; must divide 15360 (frames per patch = 15360 / hop)

    def __post_init__(self):
        assert len({c.bands for c in self.channels}) == 1, 'channels share one band count'
        assert PATCH_SAMPLES % self.hop == 0, 'hop must divide 15360'
        assert all(c.window <= c.n_fft for c in self.channels)

    @property
    def frames(self):
        return PATCH_SAMPLES // self.hop

    @property
    def bands(self):
        return self.channels[0].bands

    @property
    def n_channels(self):
        return len(self.channels)

    @property
    def max_window(self):
        return max(c.window for c in self.channels)


def _ch(ms, bands, fmin, fmax, fft=0, stride=1):
    if stride > 1:
        return SparseChannel(int(round(SR * ms / 1000)), bands, fmin, fmax, fft, stride)
    return Channel(int(round(SR * ms / 1000)), bands, fmin, fmax, fft)


# Baseline first. Others are the buzz-tuned candidates; edit after band_profile.py.
FRONTENDS = {f.name: f for f in [
    Frontend('yamnet', (_ch(25, 64, 125, 7500),)),
    # one channel, band range moved down onto the buzz (fundamental + first harmonics),
    # a longer window so the ~200 Hz fundamental resolves from its harmonics
    Frontend('lo64', (_ch(40, 64, 100, 2500, 1024),)),
    Frontend('lo32', (_ch(40, 32, 100, 2500, 1024),)),
    # two channels: the fundamental/harmonic stack, and the high-frequency wing "rasp"
    # (short window: the rasp is noise-like and transient, not tonal)
    Frontend('two32', (_ch(50, 32, 80, 1000, 1024), _ch(16, 32, 1000, 7800))),
    Frontend('two64', (_ch(50, 64, 80, 1000, 2048), _ch(16, 64, 1000, 7800))),
    # speed end of the frontier: short window (fft 256), fewer bands. Bands also scale every trunk layer.
    Frontend('fast32', (_ch(16, 32, 125, 7500),)),
    Frontend('fast16', (_ch(16, 16, 125, 7500),)),
    # lower frame rate: window = hop, so no audio is skipped. 60 frames/patch (16 ms) and 30 (32 ms)
    Frontend('fast32h16', (_ch(16, 32, 125, 7500),), hop=256),
    Frontend('fast32h32', (_ch(32, 32, 125, 7500),), hop=512),
    # band-placement twins of fast32 / fast32h16: same window, hop, bands (so the same speed), only
    # the range moved from YAMNet's 125-7500 Hz down onto the buzz, 100-2500 Hz
    Frontend('fast32lo', (_ch(16, 32, 100, 2500),)),
    Frontend('fast32h16lo', (_ch(16, 32, 100, 2500),), hop=256),
    # two channels at 60 frames: fundamentals get the longer window (fft 512), the rasp the short one
    Frontend('twofast32', (_ch(32, 32, 80, 1500), _ch(16, 32, 1500, 7800)), hop=256),
    # YAMNet's own window, bands and range at fast32h16's frame rate (hop 16 ms, 60 frames): separates
    # frame rate from the 64 -> 32 bands and 25 -> 16 ms window that the fast front ends also changed
    Frontend('yam64h16', (_ch(25, 64, 125, 7500),), hop=256),
    # 64 bands at fast32h32's frame rate (hop 32 ms, 30 frames); the window grows to the hop (32 ms, fft 512)
    # so no audio falls between frames. yam64h16 found the hop nearly free on 64 bands (-0.002 / -0.008)
    Frontend('yam64h32', (_ch(32, 64, 125, 7500),), hop=512),
    # between them: hop 24 ms (40 frames), window = hop (fft 512). yam64h32 cost 0.04-0.05 against the h16 twins;
    # this asks whether 40 frames keeps most of h16's headline at near-h32 speed
    Frontend('yam64h24', (_ch(24, 64, 125, 7500),), hop=384),
    # hop 20 ms (48 frames), YAMNet's 25 ms window kept (overlapping, as h16). h16 -> h24 cost 0.04-0.05 and h24 = h32,
    # so the cliff lies between 60 and 40 frames; this asks where
    Frontend('yam64h20', (_ch(25, 64, 125, 7500),), hop=320),
    # yam64h16 with its 64 bands over 125-2500 Hz: the mmt(z) front end is FLOP-bound in frames x window x DFT bins
    # (mmt-overlap), and 2500 Hz keeps 75 bins of 235; the lo twins of fast32/fast32h16 tied their full-band twins
    Frontend('yam64h16lo', (_ch(25, 64, 125, 2500),), hop=256),
    # yam64h16's full band read on a thinned DFT grid: every 2nd bin (62.5 Hz) wherever the bands are >= 62.5 Hz
    # apart (above ~1 kHz), 132 bins of 235. yam64h16lo showed the >2.5 kHz band carries background-tier buzz
    Frontend('yam64h16sp2', (_ch(25, 64, 125, 7500, stride=2),), hop=256),
    Frontend('lo16', (_ch(40, 16, 100, 2500, 1024),)),
    Frontend('two16', (_ch(50, 16, 80, 1000, 1024), _ch(16, 16, 1000, 7800))),
]}


def get(name):
    return FRONTENDS[name]


# ---------------------------------------------------------------- mel matrix

def _hz_to_mel(f):
    return 1127.0 * np.log1p(np.asarray(f, np.float64) / 700.0)


def mel_matrix(n_fft, bands, fmin, fmax, sr=SR):
    """(n_fft/2+1, bands): `tf.signal.linear_to_mel_weight_matrix`, numpy (DC bin zeroed)."""
    n_bins = n_fft // 2 + 1
    lin = np.linspace(0.0, sr / 2.0, n_bins)[1:]
    bin_mel = _hz_to_mel(lin)[:, None]
    edges = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), bands + 2)
    lo, ce, up = edges[:-2][None], edges[1:-1][None], edges[2:][None]
    w = np.maximum(0.0, np.minimum((bin_mel - lo) / (ce - lo), (up - bin_mel) / (up - ce)))
    return np.pad(w, [[1, 0], [0, 0]]).astype(np.float32)


def band_centers_hz(ch):
    mel = np.linspace(_hz_to_mel(ch.fmin), _hz_to_mel(ch.fmax), ch.bands + 2)[1:-1]
    return 700.0 * np.expm1(mel / 1127.0)


# ---------------------------------------------------------------- numpy

def _log_mel(x, ch, hop):
    """Continuous STFT of the whole waveform -> (frames, bands) log-mel."""
    n = ch.window
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)   # periodic Hann
    mm = ch.mel()
    view = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    out = np.empty((len(view), ch.bands), np.float32)
    step = 2048
    for i in range(0, len(view), step):
        spec = np.abs(np.fft.rfft(view[i:i + step] * win, n=ch.n_fft, axis=-1)).astype(np.float32)
        out[i:i + step] = np.log(spec @ mm + LOG_OFFSET)
    return out


def mel_patches(x, fe):
    """Waveform (float32, 16 kHz) -> (n_patches, frames, bands, channels), float32.

    Patch k covers STFT frames [f*k, f*k+f) (f = 15360/hop, 96 at YAMNet's 10 ms hop), i.e. samples from k*15360 on; only
    complete patches. The number of patches is set by the widest window, as in the
    deployed graph (which pads the waveform for that window).
    """
    x = np.ascontiguousarray(x, np.float32)
    f = fe.frames
    n_p = min((len(x) - c.window) // fe.hop + 1 for c in fe.channels) // f
    chans = [_log_mel(x, c, fe.hop)[:n_p * f].reshape(n_p, f, c.bands) for c in fe.channels]
    return np.stack(chans, axis=-1)


# ---------------------------------------------------------------- keras

def _tf():
    import tensorflow as tf
    return tf


def stft_kernel(ch):
    """(window, 1, 2 * bins) conv kernels and the DFT bins they stand for: the Hann-windowed real DFT of one
    frame as a strided Conv1D. Channels [:bins] are the real parts, [bins:] the imaginary parts, of the bins
    the mel matrix reads (the DC bin and any bin above fmax have zero weight in every band and are left out)."""
    n = ch.window
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)                     # periodic Hann, as tf.signal.stft
    bins = np.flatnonzero(ch.mel().sum(1) > 0)
    ang = 2 * np.pi * np.outer(np.arange(n), bins) / ch.n_fft
    k = np.concatenate([win[:, None] * np.cos(ang), -win[:, None] * np.sin(ang)], axis=1)
    return k[:, None, :].astype(np.float32), bins


def make_features_layer(fe, impl='stft'):
    """Keras layer: waveform (samples,) -> (log_mel_frames_of_first_channel, patches (n,96,B,C)).

    impl 'stft': tf.signal.stft and tf.signal.frame, the graph every student was exported with until
    2026-10-11. 'conv': the same spectrogram from one strided Conv1D per channel (`stft_kernel`) and a
    Reshape for the patch cut; equal up to float rounding (test_frontends.py), a cheaper exported graph.
    'mm': for windows that are a whole number of hops, frames are rows of a Reshape and the windowed DFT is
    one MatMul per hop of the window (`stft_kernel`'s rows, cut by hop); no Gather, no convolution.
    'mmt': 'mm' for any window (its kernel zero-padded to a whole number of hops) with no Pad over the
    waveform: the complete patches come from a Slice of it, the last (zero-padded) one or two from its
    short tail, joined at the patch level. A Pad or Concat over a 200 s waveform costs 11-20 ms on the CUDA
    EP against ~1 ms for the Slice + Reshape + MatMul (micro/, 2026-10-11). Same values.
    'mmtz': 'mmt' without the kernel's zero rows: hop j of the window multiplies only the window's samples in
    it (a column Slice of the last, partial hop), so a frame costs `window` kernel rows, not `m * hop`. The
    MatMuls are FLOP-bound on the GPU (mmt-overlap, 2026-10-18: ~80% of a yam64h20 student's time is the
    front end). Same values."""
    assert impl in ('stft', 'conv', 'mm', 'mmt', 'mmtz')
    assert impl != 'mm' or all(c.window % fe.hop == 0 for c in fe.channels), 'mm needs window = m * hop'
    tf = _tf()
    from tensorflow.keras import layers

    class FrontendFeatures(layers.Layer):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.min_samples = PATCH_SAMPLES - fe.hop + fe.max_window

        def _pad(self, waveform):
            # features_lib.pad_waveform's arithmetic in integers: its float32 seconds
            # (0.96 + 0.016 - 0.016) can land on 15359 in the exported graph, which adds
            # a whole patch when the input is an exact number of patches (hop 256, 120 s: 125 vs 126)
            n = tf.shape(waveform)[0]
            after = tf.maximum(n, self.min_samples) - self.min_samples
            hops = (after + PATCH_SAMPLES - 1) // PATCH_SAMPLES
            pad = tf.maximum(0, self.min_samples - n) + PATCH_SAMPLES * hops - after
            return tf.pad(waveform, [[0, pad]], mode='CONSTANT', constant_values=0.0)

        def _log_mel(self, padded, c):
            mm = c.mel()
            if impl == 'stft':
                mag = tf.abs(tf.signal.stft(padded, frame_length=c.window, frame_step=fe.hop,
                                            fft_length=c.n_fft))
            else:
                kernel, bins = stft_kernel(c)
                if impl == 'conv':
                    y = tf.nn.conv1d(padded[tf.newaxis, :, tf.newaxis], tf.constant(kernel), stride=fe.hop,
                                     padding='VALID')[0]                     # (frames, 2 * bins)
                else:
                    m = c.window // fe.hop                                   # hops per window
                    rows = tf.shape(padded)[0] // fe.hop
                    a = tf.reshape(padded[:rows * fe.hop], [rows, fe.hop])
                    k = kernel[:, 0, :]
                    y = tf.add_n([tf.matmul(a[j:rows - m + 1 + j], tf.constant(k[j * fe.hop:(j + 1) * fe.hop]))
                                  for j in range(m)])                        # frame i = rows i .. i+m-1
                re, im = y[:, :len(bins)], y[:, len(bins):]
                mag, mm = tf.sqrt(re * re + im * im), mm[bins]
            return tf.math.log(tf.matmul(mag, tf.constant(mm)) + LOG_OFFSET)

        def _patches(self, lm):
            if impl == 'stft':
                return tf.signal.frame(lm, fe.frames, fe.frames, axis=0)
            n = tf.shape(lm)[0] // fe.frames                                 # patches do not overlap
            return tf.reshape(lm[:n * fe.frames], [n, fe.frames, fe.bands])

        def _mm_frames(self, x, c, n_frames):
            """log-mel of the first n_frames frames of x (frame i = samples [i*hop, i*hop + window)), by
            Reshape + MatMul; x must hold n_frames + m - 1 whole hops (m = hops per zero-padded window)."""
            kernel, bins = stft_kernel(c)
            m = -(-c.window // fe.hop)
            k = np.zeros((m * fe.hop, kernel.shape[2]), np.float32)
            k[:c.window] = kernel[:, 0, :]
            rows = tf.shape(x)[0] // fe.hop
            a = tf.reshape(x[:rows * fe.hop], [rows, fe.hop])
            if impl == 'mmtz':      # hop j's live samples only: the last hop of the window is partial
                w = [min(fe.hop, c.window - j * fe.hop) for j in range(m)]
                y = tf.add_n([tf.matmul(a[j:j + n_frames, :w[j]] if w[j] < fe.hop else a[j:j + n_frames],
                                        tf.constant(k[j * fe.hop:j * fe.hop + w[j]])) for j in range(m)])
            else:
                y = tf.add_n([tf.matmul(a[j:j + n_frames], tf.constant(k[j * fe.hop:(j + 1) * fe.hop]))
                              for j in range(m)])
            re, im = y[:, :len(bins)], y[:, len(bins):]
            mm = c.mel()[bins]
            lm = tf.math.log(tf.matmul(tf.sqrt(re * re + im * im), tf.constant(mm)) + LOG_OFFSET)
            return tf.reshape(lm, [n_frames // fe.frames, fe.frames, fe.bands])

        def _call_mmt(self, waveform):
            # T patches in all, as _pad defines them; k of them are complete in the waveform itself
            wp = max(-(-c.window // fe.hop) for c in fe.channels) * fe.hop      # widest zero-padded window
            n = tf.shape(waveform)[0]
            after = tf.maximum(n, self.min_samples) - self.min_samples
            total = (after + PATCH_SAMPLES - 1) // PATCH_SAMPLES + 1
            k = tf.clip_by_value((n - (wp - fe.hop)) // PATCH_SAMPLES, 0, total)
            main = waveform[:k * PATCH_SAMPLES + wp - fe.hop]
            len_tail = 2 * PATCH_SAMPLES + wp - fe.hop                          # room for the 0-2 patches left
            tail = waveform[k * PATCH_SAMPLES:]
            tail = tf.reshape(tf.pad(tail, [[0, len_tail - tf.shape(tail)[0]]]), [len_tail])
            per = [tf.concat([self._mm_frames(main, c, k * fe.frames),
                              self._mm_frames(tail, c, 2 * fe.frames)[:total - k]], axis=0) for c in fe.channels]
            return tf.stack(per, axis=-1)

        def call(self, waveform):
            if impl in ('mmt', 'mmtz'):
                with tf.name_scope('frontend'):
                    return self._call_mmt(waveform)
            with tf.name_scope('frontend'):
                padded = self._pad(waveform)
                per = [self._patches(self._log_mel(padded, c)) for c in fe.channels]
                n = tf.reduce_min([tf.shape(p)[0] for p in per])
                per = [p[:n] for p in per]
                return tf.stack(per, axis=-1)

    return FrontendFeatures(name='frontend')


# ---------------------------------------------------------------- describe

def describe():
    for fe in FRONTENDS.values():
        print(f'\n{fe.name}: {fe.n_channels} channel(s) x {fe.bands} bands x {fe.frames} frames (hop {fe.hop / SR * 1000:.0f} ms)')
        for i, c in enumerate(fe.channels):
            mm = c.mel()
            hz = band_centers_hz(c)
            empty = int((mm.sum(0) == 0).sum())
            sparse = int(((mm > 0).sum(0) == 1).sum())
            print(f'  ch{i}: window {c.window / SR * 1000:.0f} ms, fft {c.n_fft} ({SR / c.n_fft:.1f} Hz/bin, '
                  f'mainlobe ~{2 * SR / c.window:.0f} Hz), {c.fmin:.0f}-{c.fmax:.0f} Hz; band centres '
                  f'{hz[0]:.0f} ... {hz[len(hz) // 2]:.0f} ... {hz[-1]:.0f} Hz; band width at low end '
                  f'{hz[1] - hz[0]:.0f} Hz; empty bands {empty}, single-bin bands {sparse}')


if __name__ == '__main__':
    describe()
