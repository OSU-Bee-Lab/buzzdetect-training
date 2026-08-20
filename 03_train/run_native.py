# import tensorflow first — later imports include pandas (see other 03_train
# entry points for the full rationale: TF's protobuf init must beat pandas').
import tensorflow as tf

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from dataset import build_fold_dataset, read_fold_roles, folds_by_role, ROLE_ROTATE
from metrics import metrics_by_group, metrics_at_fpr
from sx import summarize_sx, format_sx_report, FPR_TARGETS, FNAME_SX_SUMMARY

from embedders.yamnet.params import Params
from embedders.yamnet.yamnet import yamnet_frames_model

SETNAME = 'medium'
EMBEDDERNAME = 'yamnet'
TRANSLATION = 'general'
NAME = 'yamnet_native_buzz'
TARGET_CLASS_NAME = 'Buzz'  # AudioSet class map row 125

FNAME_PREDICTIONS = 'predictions.csv'
FNAME_FOLD_SUMMARY = 'summary.json'
FNAME_FOLDS_SUMMARY = 'folds_summary.csv'
FNAME_POOLED_METRICS = 'folds_pooled_metrics.csv'
SUBDIR_FOLDS = 'folds'


def build_native_model():
    """A one-neuron 'model' that is just YAMNet's own classification head,
    restricted to the AudioSet class matching insect buzzing. No training —
    the weights are YAMNet's, straight off the audioset-pretrained checkpoint.
    """
    dir_embedder = 'embedders/yamnet'
    class_map = pd.read_csv(os.path.join(dir_embedder, 'yamnet_class_map.csv'))
    row = class_map[class_map['display_name'] == TARGET_CLASS_NAME]
    assert len(row) == 1, f'expected exactly one class named {TARGET_CLASS_NAME!r}, found {len(row)}'
    class_index = int(row['index'].iloc[0])

    full_model = yamnet_frames_model(Params())
    full_model.load_weights(os.path.join(dir_embedder, 'weights_keras2', 'yamnet.h5'))

    dense = None
    for layer in full_model.layers:
        if isinstance(layer, tf.keras.layers.Dense) and layer.units == Params().num_classes:
            dense = layer
            break
    assert dense is not None, 'could not find the classification Dense layer'

    w, b = dense.get_weights()  # w: (1024, 521), b: (521,)
    w_buzz = w[:, class_index:class_index + 1]  # (1024, 1)
    b_buzz = b[class_index:class_index + 1]  # (1,)

    model = tf.keras.Sequential(name='yamnet_native_buzz')
    model.add(tf.keras.layers.Input(shape=(1024,), dtype=tf.float32, name='input'))
    probe = tf.keras.layers.Dense(1, name='ins_buzz')
    model.add(probe)
    probe.set_weights([w_buzz, b_buzz])
    probe.trainable = False

    print(f'[{NAME}] using AudioSet class {class_index} ({TARGET_CLASS_NAME!r}) as the raw logit')
    return model


def score_fold(model, translation, fold):
    samples = build_fold_dataset(cfg.dir_embeddings_fold(SETNAME, EMBEDDERNAME, fold), translation)
    if not samples:
        return None, None

    embeddings = np.concatenate([np.array(s.embeddings, dtype=np.float32) for s in samples])
    correct = np.concatenate([np.full(s.frames, bool(s.target_array[0])) for s in samples])
    activation = model(embeddings, training=False)[:, 0].numpy()

    predictions = pd.DataFrame({'activation_ins_buzz': activation, 'correct': correct})
    return metrics_by_group(predictions), predictions


def main():
    translation = pd.read_csv(os.path.join(cfg.DIR_TRANSLATIONS, TRANSLATION + '.csv'))
    roles = read_fold_roles(SETNAME, EMBEDDERNAME)
    folds_rotate = folds_by_role(roles, ROLE_ROTATE)

    model = build_native_model()

    dir_model = os.path.join(cfg.DIR_MODELS, NAME)
    dir_folds = os.path.join(dir_model, SUBDIR_FOLDS)

    summary_rows, predictions_pooled = [], []
    for i, fold in enumerate(folds_rotate, 1):
        dir_fold = os.path.join(dir_folds, str(fold))
        os.makedirs(dir_fold, exist_ok=True)

        metrics_df, predictions = score_fold(model, translation, fold)
        if metrics_df is None:
            print(f'[{i}/{len(folds_rotate)}] {fold}: no usable frames; skipping')
            continue

        metrics_df.to_csv(os.path.join(dir_fold, cfg.FNAME_METRICS), index=False)
        sx_df = metrics_at_fpr(metrics_df, FPR_TARGETS)
        sx_df.to_csv(os.path.join(dir_fold, cfg.FNAME_SX), index=False)
        predictions.to_csv(os.path.join(dir_fold, FNAME_PREDICTIONS), index=False)

        sens = sx_df.set_index('fpr')['sensitivity']
        scores = {f'sens_fpr{f:g}': sens[f] for f in sens.index}
        summary = {'frames_val': len(predictions), **scores}
        with open(os.path.join(dir_fold, FNAME_FOLD_SUMMARY), 'w') as f:
            json.dump(summary, f)

        summary_rows.append({'fold': fold, **summary})
        predictions['fold'] = fold
        predictions_pooled.append(predictions)

        sens_str = ', '.join(f'sens@fpr{f:.1%}={sens[f]:.3f}' if pd.notna(sens[f]) else f'sens@fpr{f:.1%}=n/a'
                             for f in sens.index)
        print(f'[{i}/{len(folds_rotate)}] {fold}: {len(predictions)} frames, {sens_str}', flush=True)

    if not summary_rows:
        print(f'[{NAME}] no fold produced usable frames')
        return

    os.makedirs(dir_model, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(os.path.join(dir_model, FNAME_FOLDS_SUMMARY), index=False)

    pooled = pd.concat(predictions_pooled, ignore_index=True)
    pooled_metrics = metrics_by_group(pooled.drop(columns='fold'))
    pooled_metrics.to_csv(os.path.join(dir_model, FNAME_POOLED_METRICS), index=False)

    sx = summarize_sx(pooled)
    sx.to_csv(os.path.join(dir_model, FNAME_SX_SUMMARY), index=False)
    print(format_sx_report(NAME, sx))


if __name__ == '__main__':
    main()
