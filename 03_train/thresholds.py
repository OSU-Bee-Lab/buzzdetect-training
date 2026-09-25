"""The per-class activation centers for a shipped model, from its CV rotations.

A class's center is its suggested threshold in raw units; export slides the
output by it, so the shipped model's suggested threshold is 0 for every class.

sx.py reads ins_buzz at a target FPR, because FPR is base-rate free and that
is what an experiment is judged on. A shipped model needs a starting threshold
an end user can reason about, and that is precision: a false-positive signal
at 70% precision is almost half the size of the true one, and in a real
activity plot it shows up as spikes that can dwarf broad true buzz. So the
suggestion is set at PRECISION_SUGGESTED instead.

Every class is centered the same way ins_buzz is. Only ins_buzz is well
represented in the rotate folds; another class is centered on whatever folds it
appears in, even one, because a noisy center beats none. A class no fold
brings to the target is centered on each fold's best-precision threshold
instead (`precision_reached` in the stats says how far short it fell). Only a
class absent from every rotate fold ships raw.

Precision depends on how much of the audio is buzz, and the folds differ a lot
there, so a fold's own 95%-precision threshold swings with its buzz fraction.
The suggestion is the plain mean over folds -- the same totalizing as
folds_sx.csv's `total` row -- so it is right for a deployment with roughly the
folds' typical buzz fraction, and stricter or looser than needed elsewhere.

Each fold's threshold is the lowest one at which its precision reaches the
target, i.e. the most sensitive threshold that meets it. metrics_at_precision
is not used: precision is not monotonic in threshold, and its
nearest-point interpolation can latch onto a spurious crossing near the top
of the sweep.

What goes in config_model.json:

    activation_centers  {class: float}, in the head's raw logit units: the raw
                        activation that export moves to 0, i.e. the raw
                        threshold suggested for each class. Numeric and nothing
                        else, so any reader can use it without knowing this
                        module exists. Absent for a class no rotate fold
                        contains, unless FNAME_MANUAL (below) sets it.
    center_stats        {class: {...}}. `source` says where the center came
                        from: 'cv' or 'manual'. A 'cv' entry also carries the
                        precision target, the number of folds it was set on
                        (0: no center), the mean precision actually reached,
                        sensitivity and false positive rate over those folds -- `sensitivity`
                        counting every positive frame, and for ins_buzz only
                        `sensitivity_exclquiet` dropping frames whose buzz is
                        only faint/quiet-tagged -- the events and frames
                        behind them, and `fold_threshold_spread`: the lowest
                        and highest fold threshold minus the center. The spread
                        is relative, so it reads the same on either scale.

04_deploy/export_onnx.py subtracts `activation_centers` from each class's
logit inside the shipped graph, so the shipped model's suggested threshold is 0
for every class listed, and a class not listed ships raw. Everything in this
directory (predictions*.csv, folds_sx.csv) stays in raw units; subtract
`activation_centers[class]` to put them on the shipped scale.

Hand-set centers live in FNAME_MANUAL, {class: float} in raw units, beside
config_model.json: for a class the folds can't center, or one whose computed
center you don't trust. It survives regeneration (config_model.json's
`activation_centers` is rewritten on every card refresh), and a class it names
takes its value instead of the computed one.

Configs written before the rename carry `thresholds`/`threshold_stats` with the
same meaning; update_config drops them when it writes the new keys.

The source is each rotation's `predictions_classes.csv`. Models trained before
that file existed fall back to `surprisal/`, which carries the same frames as
sigmoid probabilities (converted back to logits here) but no sample id and no
loudness tier, so events there are counted as runs of positive frames -- an
approximation, and the stats say so (`events_source`) -- and
sensitivity_exclquiet is left out.

No TensorFlow: nothing here loads a model.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from metrics import metrics_by_group

FNAME_PREDICTIONS_CLASSES = 'predictions_classes.csv'
FNAME_CONFIG = 'config_model.json'
FNAME_MANUAL = 'centers_manual.json'
FNAME_README = 'README.md'
SUBDIR_FOLDS = 'folds'
SUBDIR_SURPRISAL = 'surprisal'

# Every class is centered at this precision. Only ins_buzz is well represented
# in the rotate folds; the rest are centered on whatever folds they appear in,
# because a noisy center beats none.
PRECISION_SUGGESTED = 0.95
# The class whose frames carry a loudness tier (see sensitivity_exclquiet).
CLASS_LOUDNESS = 'ins_buzz'
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


def _fold_point(df, cls):
    """One fold's threshold for `cls` at PRECISION_SUGGESTED and both
    sensitivities there -- `sensitivity` over every positive frame,
    `sensitivity_exclquiet` with faint/quiet-tagged frames dropped (NaN for a
    class other than CLASS_LOUDNESS, or a fold with no loudness tier) -- or
    None if the class is absent from the fold (or is every frame of it).

    A fold that never reaches the target falls back to the best precision it
    does reach; `precision` records what the point actually achieved.

    The threshold is the lowest one whose precision meets the target: the
    most sensitive operating point that does. No interpolation; the sweep
    steps one distinct activation at a time."""
    act = df[ACT_PREFIX + cls].to_numpy()
    correct = df[TARGET_PREFIX + cls].astype(bool).to_numpy()
    if not correct.any() or correct.all():
        return None
    sweep = metrics_by_group(pd.DataFrame({'activation_ins_buzz': act, 'correct': correct}))
    target = min(PRECISION_SUGGESTED, sweep['precision'].max())
    ok = sweep[sweep['precision'] >= target]
    row = ok.loc[ok['threshold'].idxmin()]

    sens_exclquiet = float('nan')
    if cls == CLASS_LOUDNESS and LOUDNESS_COL in df.columns:
        tier = df[LOUDNESS_COL].fillna('').astype(str).to_numpy()
        loud = correct & ~np.isin(tier, TIERS_EXCLUDED)
        if loud.any():
            sens_exclquiet = float((act[loud] >= row['threshold']).mean())

    return pd.Series({
        'threshold': row['threshold'],
        'sensitivity': row['sensitivity'],
        'fpr': row['fpr'],
        'precision': row['precision'],
        'sensitivity_exclquiet': sens_exclquiet,
    })


def compute(fold_frames, classes, events_source='sample'):
    """Every class's center and its stats.

    Returns (centers, stats): `centers` {class: float} is the across-fold mean
    of each fold's threshold at PRECISION_SUGGESTED, over the folds that
    reach it -- one is enough. If none does, it's the mean of each fold's
    best-precision threshold instead, over the folds where the class
    appears. A class absent from every fold is left out of `centers` (it ships
    raw) but still gets a stats entry saying so. `stats` {class: dict} carries the across-fold means at those
    thresholds, plus how many folds and events back them."""
    if not fold_frames:
        return {}, {}
    centers, stats = {}, {}
    for cls in classes:
        points = []
        events = frames = 0
        for df in fold_frames.values():
            pos = df[TARGET_PREFIX + cls].astype(bool)
            events += int(df.loc[pos, 'sample'].nunique())
            frames += int(pos.sum())
            point = _fold_point(df, cls)
            if point is not None:
                points.append(point)

        entry = {
            'source': 'cv',
            'precision_target': PRECISION_SUGGESTED,
            'folds': len(points),
            'folds_total': len(fold_frames),
            'events': events,
            'events_source': events_source,
            'frames': frames,
        }
        if points:
            here = pd.DataFrame(points)
            # Folds short of the target only count when no fold reaches it:
            # one weak fold would otherwise drag a well-set center looser.
            reached = here['precision'] >= PRECISION_SUGGESTED
            if reached.any():
                here = here[reached]
            entry['folds'] = len(here)
            mean = float(here['threshold'].mean())
            centers[cls] = round(mean, DIGITS)
            entry['sensitivity'] = round(float(here['sensitivity'].mean()), DIGITS)
            entry['false_positive_rate'] = round(float(here['fpr'].mean()), 5)
            entry['precision_reached'] = round(float(here['precision'].mean()), DIGITS)
            entry['fold_threshold_spread'] = [
                round(float(here['threshold'].min()) - mean, DIGITS),
                round(float(here['threshold'].max()) - mean, DIGITS),
            ]
            if here['sensitivity_exclquiet'].notna().any():
                entry['sensitivity_exclquiet'] = round(
                    float(here['sensitivity_exclquiet'].mean()), DIGITS)
        stats[cls] = entry
    return centers, stats


def load_fold_frames(dir_model, classes):
    """The rotations' per-class predictions, preferring predictions_classes.csv
    and falling back to surprisal/. Returns (frames, events_source)."""
    frames = read_fold_class_predictions(dir_model)
    if frames:
        return frames, 'sample'
    frames = read_surprisal_as_class_predictions(dir_model, classes)
    return frames, 'surprisal_runs'


def merge_manual(dir_model, centers, stats, classes):
    """Fold FNAME_MANUAL's hand-set centers into the computed ones. A manual
    value replaces a computed one for the same class."""
    path = os.path.join(dir_model, FNAME_MANUAL)
    if not os.path.exists(path):
        return centers, stats
    with open(path) as f:
        manual = json.load(f)
    unknown = sorted(set(manual) - set(classes))
    if unknown:
        raise SystemExit(f'{path}: not classes of this model: {unknown}')
    centers, stats = dict(centers), dict(stats)
    for cls, value in manual.items():
        if cls in centers:
            print(f'{FNAME_MANUAL}: {cls} = {value} replaces the computed {centers[cls]}')
        centers[cls] = round(float(value), DIGITS)
        stats[cls] = {'source': 'manual'}
    return centers, stats


def update_config(dir_model, centers, stats):
    """Merge activation_centers and center_stats into the model's
    config_model.json, dropping the pre-rename thresholds/threshold_stats.
    Every other key -- including a hand-written `description` -- is kept."""
    path = os.path.join(dir_model, FNAME_CONFIG)
    with open(path) as f:
        config = json.load(f)
    config.pop('thresholds', None)
    config.pop('threshold_stats', None)
    config['activation_centers'] = centers
    config['center_stats'] = stats
    with open(path, 'w') as f:
        f.write(json.dumps(config))


def render_readme(raw):
    """A sparse README: headings for a person to fill in, and the one fact an
    end user needs about the output scale. `raw` is the classes left
    uncentered. This is written once; re-centering a class later leaves an
    existing README's list as it was."""
    threshold = '0 for every class.'
    if raw:
        threshold = (f'0 for every class except {", ".join(raw)}. Those never '
                     f'appeared in the held-out data, so they are raw activations '
                     f'with no suggested threshold.')
    return '\n'.join([
        '## At a glance',
        '_TODO_',
        '',
        '### Recommended threshold',
        threshold,
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
    """Compute activation centers for a trained model and merge them into its config.
    Writes a fresh README.md if there is none; an existing one is left alone,
    since the numbers it might once have echoed now live only in
    config_model.json. Returns (centers, stats)."""
    with open(os.path.join(dir_model, FNAME_CONFIG)) as f:
        classes = json.load(f)['classes']
    frames, source = load_fold_frames(dir_model, classes)
    if not frames:
        print(f'[{modelname}] no per-class fold predictions or surprisal on disk; '
              f'no centers written')
        return {}, {}
    centers, stats = compute(frames, classes, source)
    centers, stats = merge_manual(dir_model, centers, stats, classes)
    update_config(dir_model, centers, stats)

    path_readme = os.path.join(dir_model, FNAME_README)
    if not os.path.exists(path_readme):
        with open(path_readme, 'w') as f:
            f.write(render_readme([c for c in classes if c not in centers]))
        print(f'[{modelname}] wrote {path_readme}')
    else:
        print(f'[{modelname}] {FNAME_README} exists; left as is '
              f'(centers live in config_model.json)')
    print(f'[{modelname}] activation centers (ins_buzz @ precision {PRECISION_SUGGESTED}): {centers} {stats}'
          + ('' if source == 'sample' else f' (from {SUBDIR_SURPRISAL}/)'))
    return centers, stats
