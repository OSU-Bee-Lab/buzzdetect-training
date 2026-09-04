"""Per-epoch sens@FPR: logged every epoch, and (RestoreBestSens) used to pick
which epoch's weights get shipped.

sens@FPR is the number the project is judged on (sx.py); val_loss is what
EarlyStopping *stops* on. Those are not the same curve — val_loss is BCE with
label_smoothing over the val bulk (calibration-sensitive, penalises a
fine-tuning trunk's growing overconfidence), sens@FPR is a rank statistic on the
buzz-vs-hard-negative tail with the threshold re-derived each epoch
(calibration-invariant). They diverge late in training the way loss and AUC do,
and on this task sens@FPR keeps climbing past the val_loss minimum.

So the split (exp/restore-on-sens): keep stopping on val_loss + patience — the
6.6x variance reduction from min_delta=0.002 is real and val_loss is the more
robust *stopping* signal — but restore the weights of the epoch that maximised a
*smoothed* sens@FPR curve, not the val_loss argmin. Smoothing (rolling mean)
because the per-fold sens@FPR tail rests on ~35 negative frames and the raw
per-epoch curve is jumpy; restoring its raw argmax would harvest selection noise
on a metric that also feeds the held-out score.
"""

import numpy as np
import tensorflow as tf

from metrics import sens_at_fpr


class SensAtFPR(tf.keras.callbacks.Callback):
    """Log sens@FPR on the validation fold at the end of each epoch.

    Scores the raw validation arrays rather than the tf.data pipeline: val_tf
    is shuffled, so predicting through it and reading targets from it
    separately would pair each activation with the wrong label.

    Writes `val_sens_fpr<target>` into `logs`, which puts it in History (and in
    the per-epoch output when --verbose is set). Keras runs callbacks in list order, so
    this must be listed before EarlyStopping for those keys to exist by the
    time anything downstream reads `logs`.
    """

    def __init__(self, embeddings, correct, buzz_index, fprs, batch_size=65568):
        super().__init__()
        self.embeddings = embeddings
        self.correct = correct
        self.buzz_index = buzz_index
        self.fprs = tuple(fprs)
        self.batch_size = batch_size

    @staticmethod
    def key(fpr):
        return f'val_sens_fpr{fpr:g}'

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        chunks = [
            self.model(self.embeddings[i:i + self.batch_size], training=False)[:, self.buzz_index]
            for i in range(0, len(self.embeddings), self.batch_size)
        ]
        activation = np.concatenate([c.numpy() for c in chunks])
        for fpr, sens in sens_at_fpr(activation, self.correct, self.fprs).items():
            logs[self.key(fpr)] = sens


def _trailing_smoothed(values, window):
    """Rolling mean of the last `window` entries, NaN-ignoring. NaN only where
    every entry in the window is NaN (fold hasn't reached the target FPR yet)."""
    out = np.full(len(values), np.nan)
    arr = np.asarray(values, dtype=float)
    for i in range(len(arr)):
        w = arr[max(0, i - window + 1):i + 1]
        if not np.isnan(w).all():
            out[i] = np.nanmean(w)
    return out


class RestoreBestSens(tf.keras.callbacks.Callback):
    """Restore the weights of the epoch that maximised a smoothed sens@FPR curve.

    EarlyStopping still owns *stopping* (monitor='val_loss', patience) but no
    longer restores — this does, so it must be listed after both SensAtFPR
    (whose `logs` key it reads) and EarlyStopping (so on_train_end runs after
    the stop decision). `smoothed` here is a trailing rolling mean, recomputed
    each epoch over the sens history so far, so the epoch it picks is one whose
    *neighbourhood* was good, not a lucky single-epoch spike.

    Fallback: if the fold never reaches the target FPR at any epoch (all-NaN
    sens), restores the val_loss argmin instead, matching the old behaviour for
    that fold. `best_epoch` is 0-indexed like EarlyStopping's.
    """

    def __init__(self, sens_key, window=5, fallback_monitor='val_loss'):
        super().__init__()
        self.sens_key = sens_key
        self.window = window
        self.fallback_monitor = fallback_monitor
        self.best_epoch = 0
        self.best = np.nan
        self.restored_on = None

    def on_train_begin(self, logs=None):
        self._sens = []
        self._fallback = []
        self._sens_weights = None
        self._sens_best = -np.inf
        self._sens_best_epoch = 0
        self._fb_weights = None
        self._fb_best = np.inf
        self._fb_best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._sens.append(logs.get(self.sens_key, np.nan))
        self._fallback.append(logs.get(self.fallback_monitor, np.inf))

        smoothed = _trailing_smoothed(self._sens, self.window)[-1]
        if not np.isnan(smoothed) and smoothed > self._sens_best:
            self._sens_best = smoothed
            self._sens_best_epoch = epoch
            self._sens_weights = self.model.get_weights()

        if self._fallback[-1] < self._fb_best:
            self._fb_best = self._fallback[-1]
            self._fb_best_epoch = epoch
            self._fb_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self._sens_weights is not None:
            self.model.set_weights(self._sens_weights)
            self.best_epoch = self._sens_best_epoch
            self.best = float(self._sens_best)
            self.restored_on = f'smoothed {self.sens_key} (window {self.window})'
        elif self._fb_weights is not None:
            self.model.set_weights(self._fb_weights)
            self.best_epoch = self._fb_best_epoch
            self.best = np.nan
            self.restored_on = f'{self.fallback_monitor} (fold never reached target FPR)'
        print(f'  RestoreBestSens: epoch {self.best_epoch + 1} — {self.restored_on}',
              flush=True)
