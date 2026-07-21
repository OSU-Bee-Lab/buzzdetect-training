import tensorflow  # noqa: F401  -- load-order side effect (see 04_test/main.py)

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keras
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

import config as cfg
from embedders.embedding import load_embedder

AUDIO_FILE = os.path.join(cfg.TEST_DIR, 'night-positives', '260507_0000.mp3')
OUT_SUBPATH = os.path.join('night-positives', '260507_0000_buzzdetect.csv')
PART_SUBPATH = os.path.join('night-positives', '260507_0000_buzzpart.csv')
PLOT_SUBPATH = os.path.join('night-positives', 'night-positives.png')

DEFAULT_CHUNK_S = 200

BIN_SECONDS = 20 * 60  # 20-minute bins


def _format_hour(h):
    if h == 0 or h == 24:
        return '12 am'
    if h == 12:
        return '12 pm'
    if h < 12:
        return f'{h} am'
    return f'{h - 12} pm'


def plot_detections(df, path_out, modelname):
    total_seconds = df['start'].max() + BIN_SECONDS
    bins = np.arange(0, total_seconds + BIN_SECONDS, BIN_SECONDS)
    labels = bins[:-1]
    df['bin'] = pd.cut(df['start'], bins=bins, labels=labels, right=False)
    binned = df.groupby('bin', observed=True)['detections_ins_buzz'].sum().reset_index()
    binned['bin'] = binned['bin'].astype(float)

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f0f0f0')

    ax.plot(binned['bin'], binned['detections_ins_buzz'], color='black', linewidth=1.2)

    ax.set_xlim(0, 24 * 3600)
    tick_seconds = [h * 3600 for h in range(0, 25, 6)]
    ax.set_xticks(tick_seconds)
    ax.set_xticklabels([_format_hour(h) for h in range(0, 25, 6)])

    ax.set_ylabel('detections', labelpad=8)
    ax.set_title(modelname, fontsize=10)
    ax.tick_params(axis='both', length=0)
    ax.grid(True, color='white', linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(path_out, dpi=150)
    plt.close(fig)


def run_analysis(modelname, chunk_seconds=DEFAULT_CHUNK_S, verbose=False):
    dir_model = os.path.join(cfg.DIR_MODELS, modelname)

    with open(os.path.join(dir_model, 'config_model.json')) as f:
        config = json.load(f)

    embeddername = config['embeddername']
    buzz_index = config['classes'].index('ins_buzz')
    digits = config.get('digits_results', 8)

    embedder = load_embedder(embeddername, framehop_prop=1, initialize=True)
    framelength_s = embedder.framelength_s

    chunk_frames = round(chunk_seconds / framelength_s)
    chunk_duration_s = chunk_frames * framelength_s

    path_sx = os.path.join(dir_model, cfg.SUBDIR_TESTS, cfg.FNAME_SX)
    sx = pd.read_csv(path_sx)
    threshold = float(sx.loc[sx['precision'] == 0.95, 'threshold'].iloc[0])

    classifier = keras.saving.load_model(
        os.path.join(dir_model, 'model.keras'), compile=False
    )

    path_out = os.path.join(dir_model, cfg.SUBDIR_TESTS, OUT_SUBPATH)
    path_part = os.path.join(dir_model, cfg.SUBDIR_TESTS, PART_SUBPATH)
    os.makedirs(os.path.dirname(path_out), exist_ok=True)

    if os.path.exists(path_part):
        os.remove(path_part)

    frame_offset = 0
    header_written = False

    with sf.SoundFile(AUDIO_FILE) as track:
        source_sr = track.samplerate
        chunk_samples_source = round(chunk_duration_s * source_sr)

        while True:
            samples = track.read(frames=chunk_samples_source, dtype=embedder.dtype_in)
            if len(samples) == 0:
                break

            samples = librosa.resample(samples, orig_sr=source_sr, target_sr=embedder.samplerate)
            embeddings = embedder.embed(samples)
            activations = classifier(embeddings)[:, buzz_index].numpy()

            df_chunk = pd.DataFrame({
                'start': np.round((frame_offset + np.arange(len(activations))) * framelength_s, 4),
                'activation_ins_buzz': np.round(activations, digits),
                'detections_ins_buzz': (activations >= threshold).astype(int),
            })

            df_chunk.to_csv(path_part, mode='a', header=not header_written, index=False)
            header_written = True
            frame_offset += len(activations)
            if verbose:
                print(f'  [{modelname}] {frame_offset * framelength_s:.0f}s processed')

    os.rename(path_part, path_out)

    df = pd.read_csv(path_out)
    print(f'  [{modelname}] threshold={threshold:.6f}  detections={df["detections_ins_buzz"].sum()}  -> {path_out}')


def run_plot(modelname):
    dir_model = os.path.join(cfg.DIR_MODELS, modelname)
    path_out = os.path.join(dir_model, cfg.SUBDIR_TESTS, OUT_SUBPATH)
    df = pd.read_csv(path_out)
    path_plot = os.path.join(dir_model, cfg.SUBDIR_TESTS, PLOT_SUBPATH)
    os.makedirs(os.path.dirname(path_plot), exist_ok=True)
    plot_detections(df, path_plot, modelname)
    print(f'  [{modelname}] plot -> {path_plot}')


def run(modelname, chunk_seconds=DEFAULT_CHUNK_S, verbose=False):
    run_analysis(modelname, chunk_seconds, verbose)
    run_plot(modelname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--chunk', type=float, default=DEFAULT_CHUNK_S,
                        help='chunk duration in seconds (default: %(default)s)')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--plot-only', action='store_true', default=False,
                        help='skip analysis; rebuild plot from existing CSV')
    args = parser.parse_args()

    if args.plot_only:
        run_plot(args.model)
    else:
        run(args.model, chunk_seconds=args.chunk, verbose=args.verbose)
