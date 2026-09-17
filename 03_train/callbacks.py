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
