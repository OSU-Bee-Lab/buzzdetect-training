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
from embedders.embedding import load_embedder, BaseEmbedder
from plot_history import plot_history
from write_model_py import write_model_py

# YAMNet layer names that make up the fine-tunable tail (layers 13 + 14)
_YAMNET_TAIL_LAYERS = [
    'layer13_depthwise_conv', 'layer13_depthwise_conv_bn', 'layer13_depthwise_conv_relu',
    'layer13_pointwise_conv', 'layer13_pointwise_conv_bn', 'layer13_pointwise_conv_relu',
    'layer14_depthwise_conv', 'layer14_depthwise_conv_bn', 'layer14_depthwise_conv_relu',
    'layer14_pointwise_conv', 'layer14_pointwise_conv_bn', 'layer14_pointwise_conv_relu',
]


def _build_finetune_model(embedding_shape, classes, yamnet_dir):
    """Build fine-tune model: frozen YAMNet tail + classification head.

    Returns (model, yamnet_full) so the caller can unfreeze the tail later.
    The tail is frozen on construction; phase 2 unfreezes it.
    """
    from embedders.yamnet.yamnet import WaveformFeatures  # register custom layer
    _ = WaveformFeatures.dtype

    yamnet_full = tf.keras.models.load_model(
        os.path.join(yamnet_dir, 'yamnet.keras'), compile=False
    )
    for layer in yamnet_full.layers:
        layer.trainable = False

    inp = tf.keras.Input(shape=embedding_shape, dtype=tf.float32, name='input')
    x = inp
    for lname in _YAMNET_TAIL_LAYERS:
        x = yamnet_full.get_layer(lname)(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(len(classes))(x)

    return tf.keras.Model(inputs=inp, outputs=out), yamnet_full


def _unfreeze_tail(yamnet_full):
    """Unfreeze conv layers in the tail; keep BN frozen to preserve running stats."""
    for lname in _YAMNET_TAIL_LAYERS:
        layer = yamnet_full.get_layer(lname)
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)


def train_model(modelname, embeddername, setname, name_translation, epochs_max=300, aug_dirnames=None, verbose=False):
    dir_model = os.path.join(cfg.DIR_MODELS, modelname)
    if not can_write_model(modelname):
        print('a model folder with this name already exists; delete or rename the existing model folder and re-run')
        return False
    if verbose:
        print(f"training model {modelname} with embedder {embeddername} from set {setname}")
    os.makedirs(dir_model, exist_ok=True)

    # ---- load data ----
    if verbose:
        print('TRAINING: loading data')
    translation = pd.read_csv(os.path.join(cfg.DIR_TRANSLATIONS, name_translation + '.csv'))

    data_train: list[Sample] = build_fold_dataset(
        cfg.dir_embeddings_fold(setname, embeddername, 'train'),
        translation,
        labels_keep_raw=None,
        exclusive=False
    )

    if aug_dirnames:
        data_train += load_augmented(setname, embeddername, aug_dirnames, translation)

    data_val: list[Sample] = build_fold_dataset(
        cfg.dir_embeddings_fold(setname, embeddername, 'validate'),
        translation,
        labels_keep_raw=None,
        exclusive=False
    )

    # ---- weighting ----
    classes = build_classes(translation)
    weights = build_weights(data_train, classes)
    weight_dict = {target: weight for target, weight in enumerate(weights['weight'])}

    embedder: BaseEmbedder = load_embedder(embeddername, framehop_prop=1, initialize=False)
    embedding_shape = getattr(embedder, 'embedding_shape', None)
    is_finetune = embedding_shape is not None

    # Smaller batches for 3D embeddings to avoid OOM
    size_batch = 4096 if is_finetune else 65568
    size_shuffle = 10 * size_batch

    def data_to_tfset(set_in: list[Sample]):
        embeddings = []
        targets = []
        for s in set_in:
            embeddings.extend(s.embeddings)
            targets.extend([s.target_array] * s.frames)

        idx = np.random.permutation(len(embeddings))
        embeddings_np = np.array(embeddings, dtype=np.float32)[idx]
        targets_np = np.array(targets, dtype=np.float32)[idx]

        dataset_tf = tf.data.Dataset.from_tensor_slices((embeddings_np, targets_np))
        dataset_tf = dataset_tf.cache().shuffle(size_shuffle).batch(size_batch).prefetch(tf.data.AUTOTUNE)
        return dataset_tf

    if verbose:
        print('TRAINING: translating data to tensors')
    data_train_tf = data_to_tfset(data_train)
    data_val_tf = data_to_tfset(data_val)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.2)

    if is_finetune:
        yamnet_dir = os.path.join(cfg.DIR_EMBEDDERS, 'yamnet')
        model, yamnet_full = _build_finetune_model(embedding_shape, classes, yamnet_dir)
        model._name = modelname

        # Phase 1: warm up head with frozen YAMNet tail
        model.compile(loss=loss_fn,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
                      metrics=['accuracy'])
        cb_phase1 = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, min_delta=0.002, restore_best_weights=True)
        model.fit(data_train_tf,
                  epochs=min(epochs_max, 50),
                  validation_data=data_val_tf,
                  callbacks=[cb_phase1],
                  class_weight=weight_dict,
                  verbose=1 if verbose else 0)

        # Phase 2: unfreeze YAMNet tail (BN frozen), fine-tune with low LR
        _unfreeze_tail(yamnet_full)
        model.compile(loss=loss_fn,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                      metrics=['accuracy'])
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=50, min_delta=0.002, restore_best_weights=True)
        history = model.fit(data_train_tf,
                            epochs=epochs_max,
                            validation_data=data_val_tf,
                            callbacks=[callback],
                            class_weight=weight_dict,
                            verbose=1 if verbose else 0)

    else:
        model = tf.keras.Sequential(name=modelname)
        model.add(tf.keras.layers.Input(shape=(embedder.n_embeddings,), dtype=tf.float32, name='input'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(len(classes)))

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=50,
                                                    min_delta=0.002,
                                                    restore_best_weights=True)
        model.compile(loss=loss_fn,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001 * 2),
                      metrics=['accuracy'])
        history = model.fit(data_train_tf,
                            epochs=epochs_max,
                            validation_data=data_val_tf,
                            callbacks=callback,
                            class_weight=weight_dict,
                            verbose=1 if verbose else 0)

    model.save(os.path.join(dir_model, 'model.keras'), include_optimizer=True)

    with open(os.path.join(dir_model, 'history.pickle'), 'wb') as file:
        pickle.dump(history, file)

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
        'aug_dirnames': aug_dirnames or []
    }

    with open(os.path.join(dir_model, 'config_model.json'), 'x') as f:
        f.write(json.dumps(config_model))

    plot_history(history, modelname, callback.best_epoch, os.path.join(dir_model, 'loss_curves.svg'))
    write_model_py(dir_model, modelname, embeddername, config_model['digits_results'])

    return True
