# TensorFlow imported first — see 03_train/main.py for rationale.
import json
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf

import config as cfg

from dataset import build_fold_dataset, load_augmented
from train_utils import build_weights, build_classes, can_write_model, Sample
from embedders.embedding import load_embedder
from plot_history import plot_history
from write_model_py import write_model_py


def _load_data(setname, embeddername, name_translation, aug_dirnames):
    translation = pd.read_csv(os.path.join(cfg.DIR_TRANSLATIONS, name_translation + '.csv'))
    classes = build_classes(translation)

    data_train: list[Sample] = build_fold_dataset(
        cfg.dir_embeddings_fold(setname, embeddername, 'train'), translation,
    )
    if aug_dirnames:
        data_train += load_augmented(setname, embeddername, aug_dirnames, translation)

    data_val: list[Sample] = build_fold_dataset(
        cfg.dir_embeddings_fold(setname, embeddername, 'validate'), translation,
    )

    weights = build_weights(data_train, classes)
    weight_dict = {i: w for i, w in enumerate(weights['weight'])}

    size_batch = 65568
    size_shuffle = 10 * size_batch

    def to_tf(data):
        embeddings, targets = [], []
        for s in data:
            embeddings.extend(s.embeddings)
            targets.extend([s.target_array] * s.frames)
        idx = np.random.permutation(len(embeddings))
        emb_np = np.array(embeddings, dtype=np.float32)[idx]
        tgt_np = np.array(targets, dtype=np.float32)[idx]
        return (
            tf.data.Dataset.from_tensor_slices((emb_np, tgt_np))
            .cache().shuffle(size_shuffle).batch(size_batch).prefetch(tf.data.AUTOTUNE)
        )

    return (
        to_tf(data_train), to_tf(data_val),
        classes, weight_dict, weights, translation,
        size_batch, size_shuffle,
    )


def _train_one(modelname, embeddername, setname, name_translation,
               data_train_tf, data_val_tf, classes, weight_dict, weights, translation,
               size_batch, size_shuffle, epochs_max, aug_dirnames, verbose):
    if not can_write_model(modelname):
        print(f'[{modelname}] model folder exists; skipping')
        return False

    print(f'[{modelname}] training...')
    dir_model = os.path.join(cfg.DIR_MODELS, modelname)
    os.makedirs(dir_model, exist_ok=True)

    embedder = load_embedder(embeddername, framehop_prop=1, initialize=False)
    model = tf.keras.Sequential(name=modelname)
    model.add(tf.keras.layers.Input(shape=(embedder.n_embeddings,), dtype=tf.float32, name='input'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(len(classes), kernel_regularizer=tf.keras.regularizers.l2(1e-4)))

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=50, min_delta=0.002, restore_best_weights=True,
    )
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.2),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        metrics=['accuracy'],
    )

    history = model.fit(
        data_train_tf,
        epochs=epochs_max,
        validation_data=data_val_tf,
        callbacks=callback,
        class_weight=weight_dict,
        verbose=1 if verbose else 0,
    )

    n_epochs = len(history.history['val_loss'])
    best_val_loss = min(history.history['val_loss'])
    print(f'[{modelname}] done — {n_epochs} epochs, best epoch {callback.best_epoch + 1}, val_loss {best_val_loss:.4f}')

    model.save(os.path.join(dir_model, 'model.keras'), include_optimizer=True)

    with open(os.path.join(dir_model, 'history.pickle'), 'wb') as f:
        pickle.dump(history, f)

    dir_set = cfg.dir_set(setname)
    shutil.copy(os.path.join(dir_set, 'annotations.csv'), dir_model)
    shutil.copy(os.path.join(dir_set, 'folds.csv'), dir_model)
    weights.to_csv(os.path.join(dir_model, 'weights.csv'), index=False)
    translation.to_csv(os.path.join(dir_model, 'translation.csv'), index=False)

    config_model = {
        'embeddername': embeddername,
        'set': setname,
        'translation': name_translation,
        'classes': classes,
        'size_shuffle': size_shuffle,
        'size_batch': size_batch,
        'digits_results': 8,
        'aug_dirnames': aug_dirnames or [],
    }
    with open(os.path.join(dir_model, 'config_model.json'), 'x') as f:
        f.write(json.dumps(config_model))

    plot_history(history, modelname, callback.best_epoch, os.path.join(dir_model, 'loss_curves.svg'))
    write_model_py(dir_model, modelname, embeddername, config_model['digits_results'])
    return True


def train_set(name, runs, embeddername, setname, name_translation,
              epochs_max=400, aug_dirnames=None, verbose=False):
    data_train_tf, data_val_tf, classes, weight_dict, weights, translation, size_batch, size_shuffle = (
        _load_data(setname, embeddername, name_translation, aug_dirnames)
    )
    for i in range(1, runs + 1):
        _train_one(
            f'{name}_v{i}', embeddername, setname, name_translation,
            data_train_tf, data_val_tf, classes, weight_dict, weights, translation,
            size_batch, size_shuffle, epochs_max, aug_dirnames, verbose,
        )
