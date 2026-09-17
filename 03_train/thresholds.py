"""Suggested per-class thresholds for a shipped model, from its CV rotations.

sx.py reads one class (ins_buzz) at one target, because that is what an
experiment is judged on. A shipped model needs something broader: a starting
threshold for every class it outputs, and some sense of how far to trust it.
This module computes that from the rotations' held-out predictions and writes
it into the model's config_model.json, where buzzdetect's export carries it to
the engine, and into a sparse README.md for a person to fill in.

The threshold follows sx.py's convention exactly, so the ins_buzz number agrees
with folds_sx.csv's `total` row: each fold sets its own threshold at the target
FPR on its own held-out audio, and the suggestion is the plain mean over the
folds that could reach it. That makes it a deployment-typical threshold, not a
pooled one -- see sx.py's docstring for why the pooled read was dropped.

What goes in config_model.json:

    thresholds        {class: float}. Numeric and nothing else, so any reader
                      can use it without knowing this module exists. A class
                      reached by fewer than MIN_FOLDS folds is left out.
    threshold_stats   {class: {...}}. How much the number rests on: the FPR
                      target, the number of folds it was set on, the lowest
                      and highest sensitivity and precision observed across
                      those folds, and the buzz events (annotation samples)
                      and frames behind them.

The low/high range is over folds, i.e. over deployments: it says where a new
deployment's own operating point is likely to land, not how precisely the
mean is pinned down.

The source is each rotation's `predictions_classes.csv`: every class's logit
and 0/1 target per held-out frame, plus the `sample` id. Models trained before
that file existed fall back to `surprisal/`, which carries the same frames as
sigmoid probabilities (converted back to logits here) but no sample id, so
events there are counted as runs of positive frames -- an approximation, and
the stats say so (`events_source`).

No TensorFlow: nothing here loads a model.
"""

import glob
import json
import math
import os
import re

import numpy as np
import pandas as pd

from metrics import metrics_by_group, metrics_at_fpr

FNAME_PREDICTIONS_CLASSES = 'predictions_classes.csv'
FNAME_CONFIG = 'config_model.json'
FNAME_README = 'README.md'
SUBDIR_FOLDS = 'folds'
SUBDIR_SURPRISAL = 'surprisal'

# The suggestion's operating point. sx.FPR_TARGETS is the same number; kept
# separate so the headline metric and the shipped suggestion can diverge.
FPR_SUGGESTED = 0.005
# Fewer folds than this and there is no spread to report, so no suggestion.
MIN_FOLDS = 2
DIGITS = 3

ACT_PREFIX = 'activation_'
TARGET_PREFIX = 'target_'


def class_predictions_frame(logits, targets, sample_id, classes):
    """The per-frame table written beside predictions.csv: one logit and one
    0/1 target per class, then the sample id."""
    data = {}
    for k, c in enumerate(classes):
        data[ACT_PREFIX + c] = logits[:, k]
    for k, c in enumerate(classes):
        data[TARGET_PREFIX + c] = targets[:, k].astype(np.int8)
    data['sample'] = sample_id
    return pd.DataFrame(data)


def _classes_in(df):
    return [c[len(TARGET_PREFIX):] for c in df.columns if c.startswith(TARGET_PREFIX)]


def read_fold_class_predictions(dir_model):
    """{fold: DataFrame} from every rotation's predictions_classes.csv, or {}."""
    dir_folds = os.path.join(dir_model, SUBDIR_FOLDS)
    out = {}
    pattern = os.path.join(dir_folds, '**', FNAME_PREDICTIONS_CLASSES)
    for path in sorted(glob.glob(pattern, recursive=True)):
        out[os.path.relpath(os.path.dirname(path), dir_folds)] = pd.read_csv(path)
    return out


def read_surprisal_as_class_predictions(dir_model, classes):
    """{fold: DataFrame} rebuilt from surprisal/ for models that predate
    predictions_classes.csv. Rotating folds only: folds.csv says which ident
    belongs to which fold, and holdout folds were scored by the shipped model.

    There is no sample id in surprisal, so `sample` is a run id over frames
    with an identical label set in start order -- adjacent snips with the same
    labels merge, which undercounts events."""
    path_folds = os.path.join(dir_model, 'folds.csv')
    dir_surprisal = os.path.join(dir_model, SUBDIR_SURPRISAL)
    if not (os.path.exists(path_folds) and os.path.isdir(dir_surprisal)):
        return {}
    folds = pd.read_csv(path_folds)
    folds = folds[folds['role'] == 'rotate']

    frames = {}
    run_offset = 0
    for _, row in folds.iterrows():
        path = os.path.join(dir_surprisal, row['ident'] + '_surprisal.csv')
        if not os.path.exists(path):
            continue
        s = pd.read_csv(path).sort_values('start', kind='mergesort')
        labels = s['label'].fillna('').astype(str)
        p = s[[ACT_PREFIX + c for c in classes]].to_numpy(dtype=np.float64)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        df = pd.DataFrame(np.log(p / (1 - p)), columns=[ACT_PREFIX + c for c in classes])
        sets = labels.str.split(';').apply(set)
        for c in classes:
            df[TARGET_PREFIX + c] = sets.apply(lambda ls, c=c: int(c in ls)).to_numpy()
        runs = (labels != labels.shift()).cumsum().to_numpy()
        df['sample'] = runs + run_offset
        run_offset += int(runs.max()) + 1 if len(runs) else 0
        frames.setdefault(row['fold'], []).append(df)
    return {f: pd.concat(dfs, ignore_index=True) for f, dfs in frames.items()}


def _sweep(df, cls):
    """metrics_by_group's full sweep for one class, one-vs-rest. metrics.py is
    frozen and names its column after ins_buzz, so the class is renamed in."""
    one = pd.DataFrame({
        'activation_ins_buzz': df[ACT_PREFIX + cls].to_numpy(),
        'correct': df[TARGET_PREFIX + cls].astype(bool).to_numpy(),
    })
    return metrics_by_group(one)


def _fold_point(df, cls):
    """One fold's threshold, sensitivity and precision at FPR_SUGGESTED, or
    None if the fold can't reach it -- including, as in sx.py's _fold_sens, a
    target that corresponds to less than one negative frame, where
    metrics_at_fpr would otherwise interpolate inside one frame."""
    correct = df[TARGET_PREFIX + cls].astype(bool)
    n_pos = int(correct.sum())
    n_neg = int((~correct).sum())
    if n_pos == 0 or n_neg == 0 or math.floor(FPR_SUGGESTED * n_neg) < 1:
        return None
    sweep = _sweep(df, cls)
    row = metrics_at_fpr(sweep, [FPR_SUGGESTED]).set_index('fpr').loc[FPR_SUGGESTED]
    return None if pd.isna(row['threshold']) else row


def _count_events(df, cls):
    pos = df[TARGET_PREFIX + cls].astype(bool)
    return int(df.loc[pos, 'sample'].nunique())


def compute(fold_frames, events_source='sample'):
    """Suggested thresholds and their stats.

    Returns (thresholds, stats): `thresholds` {class: float} is the
    across-fold mean of each fold's threshold at FPR_SUGGESTED; `stats`
    {class: dict} carries the low/high range of sensitivity and precision
    observed across those folds, plus how many folds and events back it."""
    if not fold_frames:
        return {}, {}
    classes = _classes_in(next(iter(fold_frames.values())))

    thresholds, stats = {}, {}
    for cls in classes:
        points = []
        events = frames = 0
        for df in fold_frames.values():
            events += _count_events(df, cls)
            frames += int(df[TARGET_PREFIX + cls].sum())
            point = _fold_point(df, cls)
            if point is not None:
                points.append(point)

        entry = {
            'fpr_target': FPR_SUGGESTED,
            'folds': len(points),
            'folds_total': len(fold_frames),
            'events': events,
            'events_source': events_source,
            'frames': frames,
        }
        if len(points) >= MIN_FOLDS:
            here = pd.DataFrame(points)
            thresholds[cls] = round(float(here['threshold'].mean()), DIGITS)
            entry.update({
                'sensitivity_low': round(float(here['sensitivity'].min()), DIGITS),
                'sensitivity_high': round(float(here['sensitivity'].max()), DIGITS),
                'precision_low': round(float(here['precision'].min()), DIGITS),
                'precision_high': round(float(here['precision'].max()), DIGITS),
            })
        stats[cls] = entry

    return thresholds, stats


def load_fold_frames(dir_model, classes):
    """The rotations' per-class predictions, preferring predictions_classes.csv
    and falling back to surprisal/. Returns (frames, events_source)."""
    frames = read_fold_class_predictions(dir_model)
    if frames:
        return frames, 'sample'
    frames = read_surprisal_as_class_predictions(dir_model, classes)
    return frames, 'surprisal_runs'


def update_config(dir_model, thresholds, stats):
    """Merge thresholds and threshold_stats into the model's config_model.json.
    Every other key -- including a hand-written `description` -- is kept."""
    path = os.path.join(dir_model, FNAME_CONFIG)
    with open(path) as f:
        config = json.load(f)
    config['thresholds'] = thresholds
    config['threshold_stats'] = stats
    with open(path, 'w') as f:
        f.write(json.dumps(config))


def _fmt(v):
    return '' if pd.isna(v) else f'{v:.3f}'


def _markdown_table(df, cols):
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        cells = [(_fmt(r[c]) if isinstance(r[c], float) else str(r[c])) for c in cols]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


GENERATED = ('thresholds',)


def _block(name, body):
    return (f'<!-- generated:{name} -- rewritten by 03_train/thresholds.py; '
            f'edits inside are lost -->\n{body}\n<!-- /generated:{name} -->')


def render_blocks(thresholds, stats):
    """The generated parts of the README, by block name."""
    suggested = pd.DataFrame([{
        'class': cls,
        'threshold': thresholds.get(cls, np.nan),
        'sensitivity': (f"{s['sensitivity_low']:.3f} to {s['sensitivity_high']:.3f}"
                         if 'sensitivity_low' in s else ''),
        'precision': (f"{s['precision_low']:.3f} to {s['precision_high']:.3f}"
                       if 'precision_low' in s else ''),
        'deployments tested': f"{s['folds']}/{s['folds_total']}",
        'events': s['events'],
    } for cls, s in stats.items()])
    omitted = [c for c in stats if c not in thresholds]

    lines = [
        f'Suggested thresholds put each held-out deployment at {FPR_SUGGESTED:.1%} '
        f'false positive rate; the threshold is the mean across deployments that '
        f'reached it. Sensitivity and precision are given as the range observed '
        f'across those deployments, from lowest to highest. `events` counts '
        f'annotated samples.',
        '',
        _markdown_table(suggested, list(suggested.columns)) if len(suggested) else '_No rotations on disk._',
    ]
    if omitted:
        lines += ['', f'No suggestion for {", ".join(omitted)}: fewer than {MIN_FOLDS} '
                      f'deployments could reach the target.']

    return {'thresholds': _block('thresholds', '\n'.join(lines))}


def render_readme(modelname, blocks):
    """A sparse README: headings to fill in around the generated blocks. The
    thresholds are also in config_model.json, which is what programs should
    read; this is for people."""
    return '\n'.join([
        f'# {modelname}',
        '',
        '## At a glance',
        '_TODO_',
        '',
        '### Recommended threshold',
        blocks['thresholds'],
        '',
        '_TODO: interpretation_',
        '',
        '### Shortcomings',
        '_TODO_',
        '',
        '## Training',
        '_TODO_',
        '',
    ])


def refresh_blocks(text, blocks):
    """Replace each generated block already in `text`, leaving everything a
    person wrote around them alone. Returns (text, names replaced)."""
    replaced = []
    for name, block in blocks.items():
        pattern = re.compile(
            rf'<!-- generated:{re.escape(name)}\b.*?<!-- /generated:{re.escape(name)} -->',
            re.DOTALL)
        text, n = pattern.subn(lambda _m: block, text)
        if n:
            replaced.append(name)
    return text, replaced


def write_model_card(dir_model, modelname):
    """Compute thresholds for a trained model, merge them into its config, and
    write README.md -- a fresh skeleton if there is none, otherwise only its
    generated blocks. Returns (thresholds, stats)."""
    with open(os.path.join(dir_model, FNAME_CONFIG)) as f:
        classes = json.load(f)['classes']
    frames, source = load_fold_frames(dir_model, classes)
    if not frames:
        print(f'[{modelname}] no per-class fold predictions or surprisal on disk; '
              f'no thresholds written')
        return {}, {}
    thresholds, stats = compute(frames, source)
    update_config(dir_model, thresholds, stats)
    blocks = render_blocks(thresholds, stats)

    path_readme = os.path.join(dir_model, FNAME_README)
    if not os.path.exists(path_readme):
        with open(path_readme, 'w') as f:
            f.write(render_readme(modelname, blocks))
        print(f'[{modelname}] wrote {path_readme}')
    else:
        with open(path_readme) as f:
            text, replaced = refresh_blocks(f.read(), blocks)
        with open(path_readme, 'w') as f:
            f.write(text)
        missing = [b for b in blocks if b not in replaced]
        print(f'[{modelname}] {FNAME_README} exists; refreshed '
              f'{", ".join(replaced) or "no generated blocks"}'
              + (f' (no marker for {", ".join(missing)})' if missing else ''))
    print(f'[{modelname}] thresholds @ fpr {FPR_SUGGESTED}: {thresholds}'
          + ('' if source == 'sample' else f' (from {SUBDIR_SURPRISAL}/)'))
    return thresholds, stats
