import glob
import os
import pickle
import re
import sys
import time
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


def _augment_noisevol_spec(setname, embeddername, spec, embedder, fold, overwrite, verbose=False):
    t0 = time.time()
    audio_key = embedder.audio_cache_key()
    dir_audio_fold = os.path.join(cfg.dir_audio(setname), audio_key, 'raw', fold)

    if not os.path.isdir(dir_audio_fold):
        raise FileNotFoundError(f'no audio found at {dir_audio_fold}')

    aug_dir = spec_dirname(spec)
    dir_audio_aug = os.path.join(cfg.dir_audio(setname), audio_key, aug_dir, fold)
    dir_embed_aug = os.path.join(cfg.dir_embeddings_augment(setname, embeddername, aug_dir), fold)

    paths_in = glob.glob(os.path.join(dir_audio_fold, '**', '*.pickle'), recursive=True)
    paths_in = [p for p in paths_in if _path_matches_labels(p, spec.labels)]

    if not paths_in:
        print(f'{time.time()-t0:.1f}s - AUGMENT: no matching audio files for {spec}')
        return

    skipped = 0
    processed = 0
    for path_in in paths_in:
        rel = os.path.relpath(path_in, dir_audio_fold)
        path_audio_out = os.path.join(dir_audio_aug, rel)
        path_embed_out = os.path.join(dir_embed_aug, rel)

        if not overwrite and os.path.exists(path_embed_out):
            skipped += 1
            if verbose:
                print(f'{time.time()-t0:.1f}s - AUGMENT: skipping {rel} (cached)')
            continue

        if os.path.exists(path_audio_out):
            # augmented audio already exists (from a prior run with a different embedder) — reuse it
            frames_aug = read_pickle_exhaustive(path_audio_out)
        else:
            frames = read_pickle_exhaustive(path_in)
            frames_aug = _apply_noise(frames, spec.prop) if isinstance(spec, NoiseSpec) else _apply_volume(frames, spec.prop)
            _save_audio(frames_aug, path_audio_out)

        _embed_and_save(frames_aug, path_embed_out, embedder)
        processed += 1

    print(f'{time.time()-t0:.1f}s - AUGMENT: {spec} — {processed} files processed, {skipped} cached')


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


def _augment_combine_spec(setname, embeddername, spec, embedder, fold, overwrite, verbose=False):
    t0 = time.time()
    audio_key = embedder.audio_cache_key()
    dir_audio_fold = os.path.join(cfg.dir_audio(setname), audio_key, 'raw', fold)

    # Label the combined frames with class_source only. The mixed-in augment
    # audio is acoustic texture, not a target — and a category like "ambient"
    # has no translation row, so "source+augment" would surface as untranslated.
    path_embed_out = os.path.join(
        cfg.dir_embeddings_augment(setname, embeddername, spec_dirname(spec)),
        fold,
        f'{spec.class_source}.pickle'
    )

    if not overwrite and os.path.exists(path_embed_out):
        if verbose:
            print(f'{time.time()-t0:.1f}s - AUGMENT: skipping {spec.class_source}+{spec.class_augment}, already done')
        return

    def collect_frames(label):
        # Prefix match: `ins_buzz` collects every `ins_buzz_*`, `ambient` every
        # `ambient_*`. Exact-only matching missed almost all buzz, since raw
        # buzz labels are `ins_buzz_medium/low/high/...`, not bare `ins_buzz`.
        def has(labels):
            return any(l == label or l.startswith(label + '_') for l in labels)
        paths = [p for p in glob.glob(os.path.join(dir_audio_fold, '**', '*.pickle'), recursive=True)
                 if has(_labels_from_path(p))]
        frames = []
        for p in paths:
            frames.extend(read_pickle_exhaustive(p))
        return frames

    frames_source = collect_frames(spec.class_source)
    frames_aug = collect_frames(spec.class_augment)

    if not frames_source or not frames_aug:
        print(f'{time.time()-t0:.1f}s - AUGMENT: skip {fold} — '
              f'{len(frames_source)} {spec.class_source} / {len(frames_aug)} {spec.class_augment} frame(s)')
        return

    print(f'{time.time()-t0:.1f}s - AUGMENT: combining {spec.class_source}+{spec.class_augment}')
    frames_combined = _combine_frames(frames_source, frames_aug, spec.prop, spec.limit)
    _embed_and_save(frames_combined, path_embed_out, embedder)
    print(f'{time.time()-t0:.1f}s - AUGMENT: {spec.class_source}+{spec.class_augment}: {len(frames_combined)} frames combined')


def augment_set(setname, embeddername, specs, folds=('train',), overwrite=False, verbose=False):
    """Build augmented embeddings for each spec, once per fold in `folds`.

    Under the CV layout every non-excluded fold can appear in some rotation's
    training pool, so the caller usually passes them all (see `--all-folds`).
    The pre-CV single 'train' fold no longer exists.
    """
    if isinstance(folds, str):
        folds = [folds]
    t0 = time.time()
    embedder = load_embedder(embeddername, framehop_prop=1, initialize=True)

    for i, fold in enumerate(folds, 1):
        for spec in specs:
            print(f'{time.time()-t0:.1f}s - AUGMENT [{i}/{len(folds)}] {fold}: {spec}')
            if isinstance(spec, (NoiseSpec, VolumeSpec)):
                _augment_noisevol_spec(setname, embeddername, spec, embedder, fold, overwrite, verbose=verbose)
            elif isinstance(spec, CombineSpec):
                _augment_combine_spec(setname, embeddername, spec, embedder, fold, overwrite, verbose=verbose)
            else:
                raise ValueError(f'unknown spec type {type(spec)}')

    print(f'{time.time()-t0:.1f}s - AUGMENT: complete')


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
    parser.add_argument('--fold', default='train', help='single fold id; ignored if --all-folds')
    parser.add_argument('--all-folds', action='store_true',
                        help='augment every non-excluded fold in the set (CV layout)')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--noise', nargs='+', type=float, metavar='PROP',
                        help='noise augmentation prop values (default set used if neither --noise nor --volume given)')
    parser.add_argument('--volume', nargs='+', type=float, metavar='PROP',
                        help='volume augmentation prop values')
    parser.add_argument('--combine', nargs=3, action='append', metavar=('SRC', 'AUG', 'PROP'),
                        help='combine augmentation: mix SRC-class frames with AUG-class frames '
                             'at proportion PROP (repeatable). Labels output with SRC only.')
    args = parser.parse_args()

    if args.noise is not None or args.volume is not None or args.combine is not None:
        specs = (
            [NoiseSpec(prop=p) for p in (args.noise or [])] +
            [VolumeSpec(prop=p) for p in (args.volume or [])] +
            [CombineSpec(class_source=s, class_augment=a, prop=float(p), limit=2) for s, a, p in (args.combine or [])]
        )
    else:
        specs = DEFAULT_SPECS

    if args.all_folds:
        # Read folds.csv directly (stdlib csv, not pandas) to stay clear of the
        # tensorflow-before-pandas import-order invariant.
        import csv
        with open(os.path.join(cfg.dir_set(args.setname), 'folds.csv'), newline='') as fh:
            rows = list(csv.DictReader(fh))
        folds = sorted({r['fold'] for r in rows if (r.get('role') or '').strip() != 'exclude'})
    else:
        folds = [args.fold]

    augment_set(
        setname=args.setname,
        embeddername=args.embedder,
        specs=specs,
        folds=folds,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
