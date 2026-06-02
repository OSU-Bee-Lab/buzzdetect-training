import glob
import os
import pickle
import re

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

import config as cfg
from models.models import load_model, BaseModel


def run_inference(modelname):
    print(f'Testing model {modelname}')
    dir_model = os.path.join(cfg.DIR_MODELS, modelname)
    dir_test = os.path.join(dir_model, cfg.SUBDIR_TESTS, 'results')

    model: BaseModel = load_model(modelname, framehop_prop=1, initialize=False)
    framelength_s = model.embedder.framelength_s

    if os.path.exists(dir_test):
        print(f'Test directory {dir_test} already exists, skipping')
        return framelength_s

    buzz_index = model.config['classes'].index('ins_buzz')
    model.initialize()

    paths_audio = glob.glob(os.path.join(cfg.TEST_DIR_AUDIO, '**', '*.mp3'), recursive=True)

    def read_audio(path_audio, ident):
        print(f'  embedding {path_audio}')
        track = sf.SoundFile(path_audio)
        samples = track.read(dtype=model.dtype_in)
        samples = librosa.resample(samples, orig_sr=track.samplerate, target_sr=model.embedder.samplerate)

        embeddings = model.embedder.embed(samples)
        path_cache = os.path.join(cfg.TEST_DIR_EMBEDDINGS, model.embeddername, ident + '.pickle')
        os.makedirs(os.path.dirname(path_cache), exist_ok=True)
        with open(path_cache, 'wb') as file:
            pickle.dump(embeddings, file)

        return embeddings

    def analyze_embeddings(embeddings):
        results = model.predict_embeddings(embeddings)[:,buzz_index].numpy()

        df = pd.DataFrame()
        df['start'] = np.arange(0, len(results), 1) * model.embedder.framelength_s
        df['start'] = np.round(df['start'], 4)
        df['activation_ins_buzz'] = np.round(results, model.digits_results)

        return df

    def analyze(path_audio):
        ident = os.path.splitext(path_audio)[0]
        ident = re.sub(cfg.TEST_DIR_AUDIO, '', ident)
        ident = re.sub('^/', '', ident)

        path_embeddings = os.path.join(cfg.TEST_DIR_EMBEDDINGS, model.embeddername, ident + '.pickle')
        if os.path.exists(path_embeddings):
            embeddings = pickle.load(open(path_embeddings, 'rb'))
        else:
            embeddings = read_audio(path_audio, ident)
        df = analyze_embeddings(embeddings)

        path_out = os.path.join(dir_test, ident + '_buzzdetect.csv')
        os.makedirs(os.path.dirname(path_out), exist_ok=True)

        df.to_csv(path_out, index=False)

    for i, path in enumerate(paths_audio):
        print(f'[{i+1}/{len(paths_audio)}] {os.path.basename(path)}')
        analyze(path)

    print(f'Test complete for model {modelname}')
    return framelength_s
