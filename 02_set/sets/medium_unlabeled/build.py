"""Build `medium_unlabeled`: windows of *unannotated* audio from medium's
train-role folds, for pseudo-labelling (exp/pseudo-label, IDEAS item 28).

Rotating folds are left out entirely, so no audio of a scored deployment is
ever read. Each window sits at least GUARD_S from every medium annotation of
its ident (snips are padded by config.SNIP_BUFFER_S, so nothing here overlaps a
labelled snip). Every row carries the placeholder label `unlabeled`; the
embeddings are extracted for real by 02_set/main.py --set medium_unlabeled.

    python 02_set/sets/medium_unlabeled/build.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [ROOT, os.path.join(ROOT, '02_set')]
from audio_drivers import open_audio  # noqa: E402
from extract import get_ident_audio_path  # noqa: E402

MEDIUM = os.path.join(HERE, '..', 'medium')
WINDOW_S = 60.0
WINDOWS_PER_FOLD = 8
GUARD_S = 120.0
SEED = 28

annotations = pd.read_csv(os.path.join(MEDIUM, 'annotations.csv'))
folds = pd.read_csv(os.path.join(MEDIUM, 'folds.csv'))
folds = folds[folds['role'] == 'train']

rng = np.random.default_rng(SEED)
rows, kept = [], []
for fold, idents in folds.groupby('fold', sort=True):
    files = []
    for _, r in idents.iterrows():
        path = get_ident_audio_path(r['ident'])
        if not path:
            continue
        with open_audio(path) as track:
            files.append((r['source'], r['ident'], track.frames / track.samplerate))
    if not files:
        print(f'{fold}: no source audio, skipped')
        continue
    taken = {ident: [(a.start, a.end) for a in annotations[annotations['ident'] == ident].itertuples()]
             for _, ident, _ in files}
    n = 0
    for _ in range(2000):
        if n == WINDOWS_PER_FOLD:
            break
        source, ident, duration = files[rng.integers(len(files))]
        if duration < WINDOW_S + 2:
            continue
        start = float(np.floor(rng.uniform(1, duration - WINDOW_S - 1)))
        end = start + WINDOW_S
        if any(start < e + GUARD_S and end > s - GUARD_S for s, e in taken[ident]):
            continue
        taken[ident].append((start, end))
        rows.append({'source': source, 'ident': ident, 'start': start, 'end': end,
                     'label': 'unlabeled', 'duration': WINDOW_S, 'item': len(rows) + 1})
        n += 1
    kept += [{'source': s, 'ident': i, 'fold': fold, 'role': 'train'}
             for s, i, _ in files if any(r['ident'] == i for r in rows)]
    print(f'{fold}: {n} windows over {len(files)} file(s)')

pd.DataFrame(rows).to_csv(os.path.join(HERE, 'annotations.csv'), index=False)
pd.DataFrame(kept).drop_duplicates().to_csv(os.path.join(HERE, 'folds.csv'), index=False)
with open(os.path.join(MEDIUM, 'config_extract.json')) as f:
    config = json.load(f)
with open(os.path.join(HERE, 'config_extract.json'), 'w') as f:
    json.dump({**config, 'setname': 'medium_unlabeled'}, f, indent=2)
print(f'{len(rows)} windows, {len(rows) * WINDOW_S:.0f} s, {len(set(k["fold"] for k in kept))} folds')
