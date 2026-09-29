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

    @property
    def n_fft(self):
        return self.fft or 2 ** int(np.ceil(np.log2(self.window)))


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


def _ch(ms, bands, fmin, fmax, fft=0):
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
    mm = mel_matrix(ch.n_fft, ch.bands, ch.fmin, ch.fmax)
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


def make_features_layer(fe):
    """Keras layer: waveform (samples,) -> (log_mel_frames_of_first_channel, patches (n,96,B,C))."""
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

        def call(self, waveform):
            with tf.name_scope('frontend'):
                padded = self._pad(waveform)
                per = []
                for c in fe.channels:
                    stft = tf.abs(tf.signal.stft(padded, frame_length=c.window, frame_step=fe.hop,
                                                 fft_length=c.n_fft))
                    mm = tf.constant(mel_matrix(c.n_fft, c.bands, c.fmin, c.fmax))
                    lm = tf.math.log(tf.matmul(stft, mm) + LOG_OFFSET)
                    per.append(tf.signal.frame(lm, fe.frames, fe.frames, axis=0))
                n = tf.reduce_min([tf.shape(p)[0] for p in per])
                per = [p[:n] for p in per]
                return tf.stack(per, axis=-1)

    return FrontendFeatures(name='frontend')


# ---------------------------------------------------------------- describe

def describe():
    for fe in FRONTENDS.values():
        print(f'\n{fe.name}: {fe.n_channels} channel(s) x {fe.bands} bands x {fe.frames} frames (hop {fe.hop / SR * 1000:.0f} ms)')
        for i, c in enumerate(fe.channels):
            mm = mel_matrix(c.n_fft, c.bands, c.fmin, c.fmax)
            hz = band_centers_hz(c)
            empty = int((mm.sum(0) == 0).sum())
            sparse = int(((mm > 0).sum(0) == 1).sum())
            print(f'  ch{i}: window {c.window / SR * 1000:.0f} ms, fft {c.n_fft} ({SR / c.n_fft:.1f} Hz/bin, '
                  f'mainlobe ~{2 * SR / c.window:.0f} Hz), {c.fmin:.0f}-{c.fmax:.0f} Hz; band centres '
                  f'{hz[0]:.0f} ... {hz[len(hz) // 2]:.0f} ... {hz[-1]:.0f} Hz; band width at low end '
                  f'{hz[1] - hz[0]:.0f} Hz; empty bands {empty}, single-bin bands {sparse}')


if __name__ == '__main__':
    describe()
