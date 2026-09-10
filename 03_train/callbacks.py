"""Per-epoch diagnostics that don't steer training.

sens@FPR is the number the project is actually judged on (sx.py), but val_loss
is what EarlyStopping watches. Those are not the same curve, and until now
nothing recorded the gap: a run could be stopped at val_loss's minimum while
sens@FPR was still climbing, or long after it peaked, and the artifacts would
not show it.

SensAtFPR logs sens@FPR on the held-out fold at the end of every epoch, into
the same `logs` dict Keras writes to History. It is reporting only — nothing
here stops, restores, or reweights anything. Whether sens@FPR is a *usable*
stopping signal is the question these curves exist to answer; exp/tail-loss is
the standing reminder that a monitor can be much noisier than the thing it
proxies for.
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

        # Two buzz-only candidate selection statistics, reporting only (nothing
        # here stops or restores on them). See IDEAS.md "A buzz-only, low-variance
        # selection statistic": val_loss averages over all 15 neurons and can
        # stop while ins_buzz is still learning; sens@fpr0.005 is buzz-only but a
        # tail statistic on ~24-35 negative frames. These give the offline
        # across-epoch SNR comparison something to weigh against both.
        y = np.asarray(self.correct, dtype=bool)
        z = np.asarray(activation, dtype=np.float64)
        n_pos = int(y.sum())
        n_neg = y.size - n_pos
        if n_pos and n_neg:
            order = np.argsort(z, kind='mergesort')
            ranks = np.empty(z.size, dtype=np.float64)
            ranks[order] = np.arange(1, z.size + 1)
            # average-rank correction for ties
            zs = z[order]
            i = 0
            while i < zs.size:
                j = i + 1
                while j < zs.size and zs[j] == zs[i]:
                    j += 1
                if j - i > 1:
                    ranks[order[i:j]] = (i + 1 + j) / 2.0
                i = j
            auc = (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        else:
            auc = float('nan')
        logs['val_auc_buzz'] = auc
        # numerically stable buzz-only sigmoid cross-entropy
        ce = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
        logs['val_ce_buzz'] = float(ce.mean())


class RestoreTrueBest(tf.keras.callbacks.EarlyStopping):
    """EarlyStopping that restores the true val_loss argmin, not the last epoch
    that beat the previous best by `min_delta`.

    Keras uses one comparison for two jobs: `current < best - min_delta` both
    resets the patience counter *and* decides which weights get snapshotted. So
    with min_delta=0.002 the restored epoch is the last one that improved by a
    *material* margin -- every smaller improvement after it is discarded, and
    the model that ships is measurably worse than the best one the run saw.

    `restore-on-sens` measured the gap on the 11-fold CV: restoring the real
    argmin instead was +0.008 sens@fpr0.005, 9 folds up, 0 down. That is the
    portable half of that experiment; the other half (restoring the sens@FPR
    argmax rather than a loss argmin) is a backbone-fine-tuning effect that sits
    inside the noise floor on a frozen probe, so it is deliberately not done
    here.

    min_delta keeps its stopping job untouched: patience still counts epochs
    since the last material improvement, so runs stop exactly when they did
    before. Only the snapshot changes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._true_best = None
        self._true_best_weights = None
        self._true_best_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is not None:
            # self.monitor_op is np.less / np.greater WITHOUT min_delta baked in
            # (Keras folds min_delta into self.min_delta and applies it
            # separately), so this is the plain argmin/argmax.
            if self._true_best is None or self.monitor_op(current, self._true_best):
                self._true_best = current
                self._true_best_epoch = epoch
                if self.restore_best_weights:
                    self._true_best_weights = self.model.get_weights()
        super().on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        # Let EarlyStopping do its own restore/logging first, then overwrite
        # with the true argmin so `best_epoch` downstream reflects what shipped.
        super().on_train_end(logs)
        if self.restore_best_weights and self._true_best_weights is not None:
            self.model.set_weights(self._true_best_weights)
            self.best_epoch = self._true_best_epoch
            self.best = self._true_best
