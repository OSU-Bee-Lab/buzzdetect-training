"""Smoke-test a compile()'d loss/model against the exact save/load contract
training and inference actually use, on dummy data, in seconds.

Written after a real debugging detour in exp/class-weight-fix: a throwaway
script for this hit two self-inflicted bugs — `tf.keras.saving.load_model`
doesn't exist (it's `keras.saving.load_model`, see write_model_py.py), and a
relative sys.path import silently picked up main's train_utils.py instead of
a worktree's, since both trees have identically-named modules. This gets both
right once so a future experiment doesn't re-debug them.

Usage from a worktree, testing a custom loss before spending a full CV on it:

    import sys
    sys.path.insert(0, '/abs/path/to/worktree')
    sys.path.insert(0, '/abs/path/to/worktree/03_train')
    from train_utils import weighted_bce_loss
    sys.path.insert(0, '/abs/path/to/worktree/tools')
    from smoke_model import smoke_test_loss

    smoke_test_loss(weighted_bce_loss([2.0, 1.0, 0.5], label_smoothing=0.2), n_classes=3)

Run directly, it self-tests the harness with the production default loss
(BinaryCrossentropy, matching 03_train/train.py before any experiment change):

    conda run -n buzzdetect-train python tools/smoke_model.py

No config/repo dependency: this only needs TensorFlow/Keras and dummy
tensors, so it works the same from main or any worktree.
"""

import tensorflow  # noqa: F401  -- load-order side effect (see 03_train/main.py)

import os
import tempfile

import keras
import numpy as np


def smoke_test_loss(loss, n_embeddings=8, n_classes=3, n_samples=20, dropout=0.0,
                     optimizer=None, verbose=False):
    """Build the same architecture 03_train/train.py does (Input -> Dense(
    n_classes), logits out — plus a Dropout when `dropout` is nonzero, matching
    train.py's --dropout default of 0.0), compile with `loss`, fit one step on random
    multi-hot dummy data, then run the exact round trip production uses:
    `model.save(..., include_optimizer=True)` followed by
    `keras.saving.load_model(..., compile=False)` (write_model_py.py's load
    path) and a predict call.

    Raises on any failure. Returns nothing — this is a pass/fail gate, not a
    correctness check of the loss's math (test that separately with known
    inputs/outputs before calling this).
    """
    layers = [keras.layers.Input(shape=(n_embeddings,), dtype='float32')]
    if dropout:
        layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(n_classes))
    model = keras.Sequential(layers)
    model.compile(
        loss=loss,
        optimizer=optimizer or keras.optimizers.Adam(learning_rate=0.002),
        metrics=['accuracy'],
    )

    x = np.random.randn(n_samples, n_embeddings).astype('float32')
    y = (np.random.rand(n_samples, n_classes) > 0.5).astype('float32')
    model.fit(x, y, epochs=1, verbose=2 if verbose else 0)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'model.keras')
        model.save(path, include_optimizer=True)
        reloaded = keras.saving.load_model(path, compile=False)
        pred = reloaded(x[:2])

    if pred.shape != (2, n_classes):
        raise AssertionError(f'reloaded model predict shape {pred.shape} != (2, {n_classes})')
    if not np.all(np.isfinite(pred.numpy())):
        raise AssertionError('reloaded model produced non-finite predictions')

    print(f'smoke_test_loss OK: compile -> fit -> save(include_optimizer=True) '
          f'-> load(compile=False) -> predict, n_classes={n_classes}')


if __name__ == '__main__':
    smoke_test_loss(
        keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.2),
        verbose=True,
    )
