"""Build and append one 03_train/log.jsonl entry — LOOP.md step 5 — without hand-typing
JSON in a shell heredoc.

Pulls the `total` row's sensitivity straight from folds_sx.csv for both the
experiment and the baseline model (reusing compare_folds.py's reader) instead
of taking numbers by hand. The headline is `sensitivity_exclquiet`, and the
inclusive figure rides alongside it under `..._inclquiet`. `main_commit`,
`branch` and `date` are filled in from git and the name unless overridden.

`per_fold` carries each held-out fold's sensitivity under the same two keys,
so the log alone can be read (and graphed, tools/human/log_viewer.html) per
fold after the model dir is gone. The baseline's folds live on its own entry.

`set` is read straight off the model's own config_model.json (e.g. "medium",
"moderate") and omitted if that file has no `set` key.

    conda run -n buzzdetect-train python tools/log_entry.py \\
      --name class-weight-fix \\
      --model .local/worktrees/class-weight-fix/models/class_weight_fix \\
      --baseline-model models/<your matched control> \\
      --hypothesis "..." \\
      --trust caveated \\
      --conclusion "..."

Prints the entry by default; pass --write to append it to the training log (default
path: 03_train/log.jsonl, config.TRAIN_LOG) instead.

No TensorFlow: reads folds_sx.csv only, same as compare_folds.py.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare_folds import read_fold_sens, read_headline, resolve_dir_model, SENS_COL, SENS_COL_INCL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUST_VALUES = ('clean', 'caveated', 'artifact')


def _git_short_head(cwd):
    return subprocess.run(
        ['git', 'rev-parse', '--short', 'HEAD'], cwd=cwd,
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def read_set(model):
    """The training set name (e.g. "medium", "moderate") from the model's own
    config_model.json, so log entries can be told apart without re-opening the
    model dir — moderate-set runs are final-confirmation only (CLAUDE.md) and
    must not get mixed into a loop's medium-set comparisons."""
    path = os.path.join(resolve_dir_model(model), 'config_model.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get('set')


def build_entry(name, model, baseline_model, hypothesis, trust, conclusion,
                 branch=None, main_commit=None, entry_date=None,
                 baseline_name='cv-baseline-v3', fpr=0.005, method='cv'):
    if trust not in TRUST_VALUES:
        raise ValueError(f'trust must be one of {TRUST_VALUES}, got {trust!r}')

    # The headline is sensitivity_exclquiet (LOOP.md "Goal"): `_quiet`-only buzz
    # frames leave the sensitivity equation. The inclusive figure rides along
    # under a second key so a later reader can see both without reopening
    # folds_sx.csv — they share one threshold, so the pair is readable.
    sens = read_headline(model, fpr, SENS_COL)
    if sens is None:
        raise FileNotFoundError(f'no folds_sx.csv under {model!r} — run resummarize.py first if needed')
    baseline_sens = read_headline(baseline_model, fpr, SENS_COL)
    if baseline_sens is None:
        raise FileNotFoundError(f'no folds_sx.csv under {baseline_model!r}')
    sens_incl = read_headline(model, fpr, SENS_COL_INCL)
    baseline_sens_incl = read_headline(baseline_model, fpr, SENS_COL_INCL)

    metric_key = f'sens_at_fpr{fpr}_persite'
    metric_key_incl = f'{metric_key}_inclquiet'

    def metrics(headline, incl):
        out = {metric_key: round(float(headline), 3)}
        if incl is not None:
            out[metric_key_incl] = round(float(incl), 3)
        return out

    def per_fold(model):
        return {fold: {key: (None if v.get(col) is None else round(v[col], 3))
                       for key, col in ((metric_key, SENS_COL), (metric_key_incl, SENS_COL_INCL))
                       if col in v}
                for fold, v in read_fold_sens(model, fpr).items()}

    entry_set = read_set(model)

    entry = {
        'name': name,
        'branch': branch or f'exp/{name}',
        'date': entry_date or date.today().isoformat(),
        # Convention: main's HEAD at the moment you log, not the commit the
        # experiment branched from — those can diverge if main moves while a
        # long CV run is in flight. See LOOP.md's note on main_commit.
        'main_commit': main_commit or _git_short_head(ROOT),
        'method': method,
        'hypothesis': hypothesis,
        'metrics': metrics(sens, sens_incl),
        'baseline': {'model': baseline_name, **metrics(baseline_sens, baseline_sens_incl)},
        'per_fold': per_fold(model),
        'trust': trust,
        'conclusion': conclusion,
    }
    if entry_set is not None:
        entry['set'] = entry_set
    return entry


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', required=True, help='experiment slug, e.g. class-weight-fix')
    parser.add_argument('--model', required=True, help='trained model dir (name under models/, or a path — e.g. into a worktree)')
    parser.add_argument('--baseline-model', default='models/cv_baseline_v3',
                         help='baseline model dir to diff against; usually your matched control, not the era anchor')
    parser.add_argument('--baseline-name', default=None,
                        help="label for baseline.model in the entry; default: --baseline-model's dir name, _ -> -")
    parser.add_argument('--hypothesis', required=True)
    parser.add_argument('--trust', required=True, choices=TRUST_VALUES)
    parser.add_argument('--conclusion', required=True)
    parser.add_argument('--branch', default=None, help='default: exp/<name>')
    parser.add_argument('--main-commit', default=None, help='default: git rev-parse --short HEAD in the repo root')
    parser.add_argument('--date', default=None, help='default: today, YYYY-MM-DD')
    parser.add_argument('--fpr', type=float, default=0.005, choices=[0.001, 0.005, 0.01])
    parser.add_argument('--method', default='cv')
    parser.add_argument('--log', default=os.path.join(ROOT, '03_train', 'log.jsonl'))   # config.TRAIN_LOG
    parser.add_argument('--write', action='store_true', help='append to --log instead of just printing')
    args = parser.parse_args()
    if args.baseline_name is None:
        args.baseline_name = os.path.basename(os.path.normpath(args.baseline_model)).replace('_', '-')

    entry = build_entry(
        name=args.name, model=args.model, baseline_model=args.baseline_model,
        hypothesis=args.hypothesis, trust=args.trust, conclusion=args.conclusion,
        branch=args.branch, main_commit=args.main_commit, entry_date=args.date,
        baseline_name=args.baseline_name, fpr=args.fpr, method=args.method,
    )
    line = json.dumps(entry)

    if args.write:
        with open(args.log, 'a') as f:
            f.write(line + '\n')
        print(f'appended to {args.log}')
    print(line)
