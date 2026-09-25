import glob
import os
import pickle
import re
import sys
from itertools import product, cycle

# Make the project root and this directory importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import config as cfg
from augment_specs import NoiseSpec, VolumeSpec, CombineSpec, spec_dirname
from embedders.embedding import load_embedder
from utils import read_pickle_exhaustive


def _labels_from_path(path):
    base = os.path.splitext(os.path.basename(path))[0]
    return re.split(r'\+', base)


def _path_matches_labels(path, labels):
    if labels is None:
        return True
    return any(l in _labels_from_path(path) for l in labels)


def _apply_noise(frames, prop):
    return [f + prop * np.random.uniform(-1, 1, len(f)) for f in frames]


def _apply_volume(frames, prop):
    return [f * prop for f in frames]


def _save_audio(frames, path_out):
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    with open(path_out, 'wb') as f:
        for fr in frames:
            pickle.dump(fr, f)


def _embed_and_save(frames, path_embed_out, embedder):
    framelength_samples = int(embedder.framelength_s * embedder.samplerate)
    chunk_samples = cfg.CHUNK_FRAMES * framelength_samples
    frames_flat = np.concatenate(frames)
    os.makedirs(os.path.dirname(path_embed_out), exist_ok=True)
    with open(path_embed_out, 'wb') as f:
        for start in range(0, len(frames_flat), chunk_samples):
            for e in embedder.embed(frames_flat[start:start + chunk_samples]):
                pickle.dump(e, f)


def _augment_noisevol_spec(setname, embeddername, spec, embedder, fold, overwrite):
    audio_key = embedder.audio_cache_key()
    dir_audio_fold = os.path.join(cfg.dir_audio(setname), audio_key, fold)

    if not os.path.isdir(dir_audio_fold):
        raise FileNotFoundError(f'no audio found at {dir_audio_fold}')

    aug_dir = spec_dirname(spec)
    dir_audio_aug = os.path.join(cfg.dir_audio(setname), audio_key, aug_dir, fold)
    dir_embed_aug = os.path.join(cfg.dir_embeddings_augment(setname, embeddername, aug_dir), fold)

    paths_in = glob.glob(os.path.join(dir_audio_fold, '**', '*.pickle'), recursive=True)
    paths_in = [p for p in paths_in if _path_matches_labels(p, spec.labels)]

    if not paths_in:
        print(f'AUGMENT: no matching audio files for {spec}')
        return

    skipped = 0
    for path_in in paths_in:
        rel = os.path.relpath(path_in, dir_audio_fold)
        path_audio_out = os.path.join(dir_audio_aug, rel)
        path_embed_out = os.path.join(dir_embed_aug, rel)

        if not overwrite and os.path.exists(path_embed_out):
            skipped += 1
            continue

        if os.path.exists(path_audio_out):
            # augmented audio already exists (from a prior run with a different embedder) — reuse it
            frames_aug = read_pickle_exhaustive(path_audio_out)
        else:
            frames = read_pickle_exhaustive(path_in)
            frames_aug = _apply_noise(frames, spec.prop) if isinstance(spec, NoiseSpec) else _apply_volume(frames, spec.prop)
            _save_audio(frames_aug, path_audio_out)

        _embed_and_save(frames_aug, path_embed_out, embedder)
        print(f'AUGMENT: {rel}')

    if skipped:
        print(f'AUGMENT: skipped {skipped} already-done files')


def _combine_frames(frames_source, frames_augment, prop, limit):
    def mix(fs, fa):
        return fs * prop + fa * (1 - prop)

    n_aug = len(frames_augment)
    if n_aug <= limit:
        return [mix(s, a) for s, a in product(frames_source, frames_augment)]

    shuffled = frames_augment.copy()
    np.random.shuffle(shuffled)
    cycler = cycle(shuffled)
    return [mix(s, next(cycler)) for s in frames_source for _ in range(limit)]


def _augment_combine_spec(setname, embeddername, spec, embedder, fold, overwrite):
    audio_key = embedder.audio_cache_key()
    dir_audio_fold = os.path.join(cfg.dir_audio(setname), audio_key, fold)

    path_embed_out = os.path.join(
        cfg.dir_embeddings_augment(setname, embeddername, 'augment_combine'),
        fold,
        f'{spec.class_source}+{spec.class_augment}.pickle'
    )

    if not overwrite and os.path.exists(path_embed_out):
        print(f'AUGMENT: skipping {spec.class_source}+{spec.class_augment}, already done')
        return

    def collect_frames(label):
        paths = [p for p in glob.glob(os.path.join(dir_audio_fold, '**', '*.pickle'), recursive=True)
                 if label in _labels_from_path(p)]
        frames = []
        for p in paths:
            frames.extend(read_pickle_exhaustive(p))
        return frames

    frames_source = collect_frames(spec.class_source)
    frames_aug = collect_frames(spec.class_augment)

    if not frames_source:
        print(f'AUGMENT: no source frames for {spec.class_source}')
        return
    if not frames_aug:
        print(f'AUGMENT: no augment frames for {spec.class_augment}')
        return

    print(f'AUGMENT: combining {spec.class_source}+{spec.class_augment}')
    frames_combined = _combine_frames(frames_source, frames_aug, spec.prop, spec.limit)
    _embed_and_save(frames_combined, path_embed_out, embedder)
    print(f'AUGMENT: {spec.class_source}+{spec.class_augment}: {len(frames_combined)} frames combined')


def augment_set(setname, embeddername, specs, fold='train', overwrite=False):
    embedder = load_embedder(embeddername, framehop_prop=1, initialize=True)

    for spec in specs:
        print(f'AUGMENT: {spec}')
        if isinstance(spec, (NoiseSpec, VolumeSpec)):
            _augment_noisevol_spec(setname, embeddername, spec, embedder, fold, overwrite)
        elif isinstance(spec, CombineSpec):
            _augment_combine_spec(setname, embeddername, spec, embedder, fold, overwrite)
        else:
            raise ValueError(f'unknown spec type {type(spec)}')

    print('AUGMENT: complete')


DEFAULT_SPECS = [
    NoiseSpec(prop=0.05),
    NoiseSpec(prop=0.075),
    NoiseSpec(prop=0.2),
    VolumeSpec(prop=2.5),
    VolumeSpec(prop=0.75),
]

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--set', required=True, dest='setname')
    parser.add_argument('--embedder', required=True)
    parser.add_argument('--fold', default='train')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    augment_set(
        setname=args.setname,
        embeddername=args.embedder,
        specs=DEFAULT_SPECS,
        fold=args.fold,
        overwrite=args.overwrite,
    )
