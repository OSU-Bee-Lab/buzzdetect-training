"""Write an ensemble "model" from existing CV runs: per held-out frame, the mean of the
members' ins_buzz logits, scored by 03_train/sx.py exactly like a trained run.

    python tools_exp/ensemble_predictions.py <name> <member model dir> <member model dir> ...

Refuses unless every member's predictions.csv has the same rows per fold (correct,
loudness, sample): the members must have scored the same frames.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, '03_train'))

import pandas as pd  # noqa: E402

import config as cfg  # noqa: E402
from sx import (summarize_folds, format_sx_report, read_fold_predictions,  # noqa: E402
                FNAME_SX_SUMMARY, FNAME_PREDICTIONS)

KEYS = ['correct', 'loudness', 'sample']


def main(name, members):
    tables = [read_fold_predictions(os.path.join(m, 'folds')) for m in members]
    base = tables[0]
    for m, t in zip(members[1:], tables[1:]):
        same = (len(t) == len(base) and (t['fold'] == base['fold']).all()
                and all((t[k].fillna('') == base[k].fillna('')).all() for k in KEYS))
        if not same:
            sys.exit(f'{m}: rows differ from {members[0]}; members did not score the same frames')
    out = base.copy()
    out['activation_ins_buzz'] = sum(t['activation_ins_buzz'] for t in tables) / len(tables)

    dir_model = os.path.join(cfg.DIR_MODELS, name)
    for fold, df in out.groupby('fold'):
        d = os.path.join(dir_model, 'folds', fold)
        os.makedirs(d, exist_ok=True)
        df.drop(columns='fold').to_csv(os.path.join(d, FNAME_PREDICTIONS), index=False)
    sx = summarize_folds(out)
    sx.to_csv(os.path.join(dir_model, FNAME_SX_SUMMARY), index=False)
    with open(os.path.join(dir_model, 'members.txt'), 'w') as f:
        f.write('\n'.join(os.path.abspath(m) for m in members) + '\n')
    print(format_sx_report(name, sx))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
