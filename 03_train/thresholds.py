"""Suggested per-class thresholds for a shipped model, from its CV rotations.

sx.py reads one class (ins_buzz) at one target, because that is what an
experiment is judged on. A shipped model needs something broader: a starting
threshold for every class it outputs, and some sense of how far to trust it.
This module computes that from the rotations' held-out predictions and writes
it into the model's config_model.json, where buzzdetect's export carries it to
the engine. config_model.json is the only place this table lives -- the
README used to carry a duplicate generated one, which was one more place for
the numbers to go stale against each other; write_model_card() now leaves a
fresh README's threshold section pointing at the config instead.

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
                      target, the number of folds it was set on, the mean
                      sensitivity over those folds -- `sensitivity` counting
                      every positive frame and `sensitivity_exclquiet`
                      dropping frames whose buzz is only faint/quiet-tagged
                      (sx.py's sensitivity_exclquiet, generalized here from
                      ins_buzz to every class) -- and the buzz events
                      (annotation samples) and frames behind them.

Both sensitivities are the plain mean over the folds that reached the target,
same as the threshold -- a deployment-typical read, not a pooled one.
`sensitivity_exclquiet` is None where no fold carried a loudness tier (a run
predating it, or the surprisal fallback below).

The source is each rotation's `predictions_classes.csv`: every class's logit
and 0/1 target per held-out frame, the buzz loudness tier, and the `sample`
id. Models trained before that file existed fall back to `surprisal/`, which
carries the same frames as sigmoid probabilities (converted back to logits
here) but no sample id and no loudness tier, so events there are counted as
runs of positive frames -- an approximation, and the stats say so
(`events_source`) -- and sensitivity_exclquiet is left out entirely.

No TensorFlow: nothing here loads a model.
"""

import glob
import json
import math
import os

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
LOUDNESS_COL = 'loudness'

# Which loudness tiers don't count toward sensitivity. Mirrors
# train_utils.TIERS_EXCLUDED_FROM_HEADLINE, duplicated rather than imported --
# train_utils pulls in TensorFlow at module load, and this module's whole
# point is not to.
TIERS_EXCLUDED = ('faint', 'quiet')


def class_predictions_frame(logits, targets, sample_id, classes, loudness):
    """The per-frame table written beside predictions.csv: one logit and one
    0/1 target per class, the sample id, and the frame's buzz loudness tier
    (empty for a non-buzz frame; see train_utils.buzz_tier). loudness is a
    property of the annotation, not of any one class, so every class's column
    shares it."""
    data = {}
    for k, c in enumerate(classes):
        data[ACT_PREFIX + c] = logits[:, k]
    for k, c in enumerate(classes):
        data[TARGET_PREFIX + c] = targets[:, k].astype(np.int8)
    data['sample'] = sample_id
    data[LOUDNESS_COL] = loudness
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


def _sens_over(df, cls, positives, fprs):
    """Sensitivity at each target FPR over a restricted set of positives, at
    the SAME threshold sweep -- sx.py's `_sens_over`, generalized here from
    ins_buzz to any class. The negatives are untouched, so the threshold this
    reads sensitivity off is identical to the unrestricted sweep's; only the
    sensitivity numerator and denominator change."""
    correct = df[TARGET_PREFIX + cls].astype(bool)
    keep = df[(~correct) | positives]
    if not positives.any():
        return pd.Series({f: np.nan for f in fprs})
    one = pd.DataFrame({
        'activation_ins_buzz': keep[ACT_PREFIX + cls].to_numpy(),
        'correct': positives.loc[keep.index].to_numpy(),
    })
    return metrics_at_fpr(metrics_by_group(one), fprs).set_index('fpr')['sensitivity']


def _fold_point(df, cls):
    """One fold's threshold and its two sensitivities at FPR_SUGGESTED --
    `sensitivity` over every positive frame, `sensitivity_exclquiet` with
    frames tagged faint/quiet dropped (sx.py's sensitivity_exclquiet,
    generalized from ins_buzz to any class; NaN if this fold carries no
    loudness tier at all) -- or None if the fold can't reach the target,
    including, as in sx.py's _fold_sens, a target that corresponds to less
    than one negative frame, where metrics_at_fpr would otherwise interpolate
    inside one frame."""
    correct = df[TARGET_PREFIX + cls].astype(bool)
    n_pos = int(correct.sum())
    n_neg = int((~correct).sum())
    if n_pos == 0 or n_neg == 0 or math.floor(FPR_SUGGESTED * n_neg) < 1:
        return None
    sweep = _sweep(df, cls)
    row = metrics_at_fpr(sweep, [FPR_SUGGESTED]).set_index('fpr').loc[FPR_SUGGESTED]
    if pd.isna(row['threshold']):
        return None

    sens_exclquiet = float('nan')
    if LOUDNESS_COL in df.columns:
        tier = df[LOUDNESS_COL].fillna('').astype(str)
        excl = _sens_over(df, cls, correct & ~tier.isin(TIERS_EXCLUDED), [FPR_SUGGESTED])
        sens_exclquiet = excl.get(FPR_SUGGESTED, float('nan'))

    return pd.Series({
        'threshold': row['threshold'],
        'sensitivity': row['sensitivity'],
        'sensitivity_exclquiet': sens_exclquiet,
    })


def _count_events(df, cls):
    pos = df[TARGET_PREFIX + cls].astype(bool)
    return int(df.loc[pos, 'sample'].nunique())


def compute(fold_frames, events_source='sample'):
    """Suggested thresholds and their stats.

    Returns (thresholds, stats): `thresholds` {class: float} is the
    across-fold mean of each fold's threshold at FPR_SUGGESTED; `stats`
    {class: dict} carries the across-fold mean of both sensitivities at that
    threshold, plus how many folds and events back it."""
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
            entry['sensitivity'] = round(float(here['sensitivity'].mean()), DIGITS)
            entry['sensitivity_exclquiet'] = (
                round(float(here['sensitivity_exclquiet'].mean()), DIGITS)
                if here['sensitivity_exclquiet'].notna().any() else None
            )
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


def render_readme(modelname):
    """A sparse README: headings for a person to fill in. The suggested
    thresholds live only in config_model.json now -- that's what programs
    (and this README) point readers at, rather than duplicating the table
    here where it could go stale against the config."""
    return '\n'.join([
        f'# {modelname}',
        '',
        '## At a glance',
        '_TODO_',
        '',
        '### Recommended threshold',
        'See `thresholds` and `threshold_stats` in config_model.json.',
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


def write_model_card(dir_model, modelname):
    """Compute thresholds for a trained model and merge them into its config.
    Writes a fresh README.md if there is none; an existing one is left alone,
    since the thresholds it might once have echoed now live only in
    config_model.json. Returns (thresholds, stats)."""
    with open(os.path.join(dir_model, FNAME_CONFIG)) as f:
        classes = json.load(f)['classes']
    frames, source = load_fold_frames(dir_model, classes)
    if not frames:
        print(f'[{modelname}] no per-class fold predictions or surprisal on disk; '
              f'no thresholds written')
        return {}, {}
    thresholds, stats = compute(frames, source)
    update_config(dir_model, thresholds, stats)

    path_readme = os.path.join(dir_model, FNAME_README)
    if not os.path.exists(path_readme):
        with open(path_readme, 'w') as f:
            f.write(render_readme(modelname))
        print(f'[{modelname}] wrote {path_readme}')
    else:
        print(f'[{modelname}] {FNAME_README} exists; left as is '
              f'(thresholds live in config_model.json)')
    print(f'[{modelname}] thresholds @ fpr {FPR_SUGGESTED}: {thresholds}'
          + ('' if source == 'sample' else f' (from {SUBDIR_SURPRISAL}/)'))
    return thresholds, stats
