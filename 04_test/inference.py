import glob
import json
import os
import pickle
import re

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

import config as cfg
from embedders.embedding import load_embedder


def _path_to_ident(path_audio):
    ident = os.path.splitext(path_audio)[0]
    ident = re.sub(re.escape(cfg.TEST_DIR_AUDIO), '', ident)
    ident = re.sub('^/', '', ident)
    return ident


def ensure_test_embeddings(embeddername):
    """Embed all test audio for an embedder (skips files already cached).

    Returns ({ident: path_cache}, framelength_s).
    """
    embedder = load_embedder(embeddername, framehop_prop=1, initialize=True)
    framelength_s = embedder.framelength_s

    paths_audio = glob.glob(os.path.join(cfg.TEST_DIR_AUDIO, '**', '*.mp3'), recursive=True)
    idents = {}

    for i, path_audio in enumerate(paths_audio):
        ident = _path_to_ident(path_audio)
        path_cache = os.path.join(cfg.TEST_DIR_EMBEDDINGS, embeddername, ident + '.pickle')
        idents[ident] = path_cache

        if os.path.exists(path_cache):
            continue

        print(f'  [{i+1}/{len(paths_audio)}] embedding {os.path.basename(path_audio)}')
        with sf.SoundFile(path_audio) as track:
            samples = track.read(dtype=embedder.dtype_in)
            samples = librosa.resample(samples, orig_sr=track.samplerate, target_sr=embedder.samplerate)

        embeddings = embedder.embed(samples)
        os.makedirs(os.path.dirname(path_cache), exist_ok=True)
        with open(path_cache, 'wb') as f:
            pickle.dump(embeddings, f)

    return idents, framelength_s


def run_inference_for_model(modelname, embeddings_by_ident, framelength_s):
    """Run classifier predictions for one model against pre-loaded embedding paths.

    Skips if results directory already exists.
    """
    import keras

    dir_model = os.path.join(cfg.DIR_MODELS, modelname)
    dir_results = os.path.join(dir_model, cfg.SUBDIR_TESTS, 'results')

    if os.path.exists(dir_results):
        print(f'  [{modelname}] results exist, skipping')
        return

    with open(os.path.join(dir_model, 'config_model.json')) as f:
        config = json.load(f)
    buzz_index = config['classes'].index('ins_buzz')
    digits = config.get('digits_results', 8)

    print(f'  [{modelname}] running inference...')
    classifier = keras.saving.load_model(os.path.join(dir_model, 'model.keras'), compile=False)

    for ident, path_cache in embeddings_by_ident.items():
        embeddings = pickle.load(open(path_cache, 'rb'))
        results = classifier(embeddings)[:, buzz_index].numpy()

        df = pd.DataFrame({
            'start': np.round(np.arange(len(results)) * framelength_s, 4),
            'activation_ins_buzz': np.round(results, digits),
        })

        path_out = os.path.join(dir_results, ident + '_buzzdetect.csv')
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        df.to_csv(path_out, index=False)


# Legacy single-model entry point (used by older worktrees / standalone calls).
def run_inference(modelname):
    with open(os.path.join(cfg.DIR_MODELS, modelname, 'config_model.json')) as f:
        config = json.load(f)
    embeddername = config['embeddername']

    embeddings_by_ident, framelength_s = ensure_test_embeddings(embeddername)
    run_inference_for_model(modelname, embeddings_by_ident, framelength_s)
    return framelength_s
