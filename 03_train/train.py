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


def train_model(modelname, embeddername, setname, name_translation, epochs_max=300, aug_dirnames=None):
    dir_model = os.path.join(cfg.DIR_MODELS, modelname)
    if not can_write_model(modelname):
        print('a model folder with this name already exists; delete or rename the existing model folder and re-run')
        return False
    print(f"training model {modelname} with embedder {embeddername} from set {setname}")
    os.makedirs(dir_model, exist_ok=True)

    # ---- load data ----
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

    labels_buzz = translation['from'][translation['to']=='ins_buzz'].to_list()
    data_val: list[Sample] = build_fold_dataset(
        cfg.dir_embeddings_fold(setname, embeddername, 'validate'),
        translation,
        labels_keep_raw=labels_buzz,  # TODO: previously, validating only on buzzes improved performance. Does it still?
        exclusive=False
    )

    # ---- weighting ----
    classes = build_classes(translation)
    weights = build_weights(data_train, classes)
    weight_dict = {target: weight for target, weight in enumerate(weights['weight'])}  # for training

    # Turning datasets into tensorflow datasets
    #
    size_batch = 65568
    size_shuffle = 10*size_batch

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

    # These steps take a while, but I think that's unavoidable due to dataset size
    print('TRAINING: translating data to tensors')
    data_train_tf = data_to_tfset(data_train)
    data_val_tf = data_to_tfset(data_val)

    # Model creation
    #
    embedder: BaseEmbedder = load_embedder(embeddername, framehop_prop=1, initialize=False)

    model = tf.keras.Sequential(name=modelname)
    model.add(tf.keras.layers.Input(shape=(embedder.n_embeddings,), dtype=tf.float32, name='input'))
    model.add(tf.keras.layers.Dense(len(classes)))

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=30,
                                                min_delta=0.01,
                                                restore_best_weights=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001*2)  # 0.001 is default

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=optimizer,
                  metrics=['accuracy'])

    history = model.fit(data_train_tf,
                        epochs=epochs_max,
                        validation_data=data_val_tf,
                        callbacks=callback,
                        class_weight=weight_dict)

    model.save(os.path.join(dir_model, 'model.keras'), include_optimizer=True)


    with open(os.path.join(dir_model, 'history.pickle'), 'wb') as file:
        pickle.dump(history, file)
    # write metadata (after successful training, just in case)
    # copy set info, because sets may change over time (e.g., standard will be updated as new annotations come in)
    dir_set = cfg.dir_set(setname)
    # shutil.copy(os.path.join(dir_set, 'config_set.json'), dir_model)
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
        'digits_results': 8  # set high, tune later
    }

    with open(os.path.join(dir_model, 'config_model.json'), 'x') as f:
        f.write(json.dumps(config_model))

    plot_history(history, modelname, callback.best_epoch, os.path.join(dir_model, 'loss_curves.svg'))
    write_model_py(dir_model, modelname, embeddername, config_model['digits_results'])

    return True
