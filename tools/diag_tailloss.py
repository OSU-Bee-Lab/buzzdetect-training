"""Diagnostic-only driver for the tail-loss-retest experiment: trains
TailBCELoss (fixed EarlyStopping monitor, fixed-size top_k) on a couple of
folds, not the full 11-fold CV, to confirm best_epoch lands in the normal
range before spending a full CV run. Not part of the pipeline; not committed
to main.

Ported from exp/tail-loss's tools/diag_tailloss.py; --frac replaced with --k
since ohem_k is now a fixed Python int, not a data-dependent fraction.

Usage (from this worktree, conda env buzzdetect-train):
    python tools/diag_tailloss.py --k 200 --folds \
        "JamesU - MustardBumbler/1_29" \
        "Luke - Various Opportunistic Recordings/2025-08-12/1_114"
"""
import tensorflow  # noqa: F401 -- import order, see 03_train/main.py

import argparse
import os
import sys

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
sys.path.insert(0, os.path.join(WT, '03_train'))

import config as cfg
import train
from dataset import read_fold_roles, folds_by_role, ROLE_TRAIN, ROLE_ROTATE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--weight', type=float, default=1.0)
    parser.add_argument('--folds', nargs='+', required=True)
    parser.add_argument('--set', default='medium')
    parser.add_argument('--embedder', default='yamnet')
    parser.add_argument('--translation', default='general')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--tag', default=None)
    args = parser.parse_args()

    train.OHEM_K = args.k
    train.OHEM_WEIGHT = args.weight

    roles = read_fold_roles(args.set, args.embedder)
    folds_rotate = folds_by_role(roles, ROLE_ROTATE)
    folds_train_always = folds_by_role(roles, ROLE_TRAIN)

    tag = args.tag or f'diag_k{args.k}_w{args.weight}'
    for held_out in args.folds:
        assert held_out in folds_rotate, f'{held_out!r} is not a rotate fold'
        folds_train = [f for f in folds_rotate if f != held_out] + folds_train_always

        data = train._load_data(args.set, args.embedder, folds_train, args.translation,
                                 None, val_fold=held_out)
        dir_model = os.path.join(cfg.DIR_MODELS, tag, held_out.replace('/', '_'))
        os.makedirs(dir_model, exist_ok=True)
        modelname = f'{tag}_{held_out.replace("/", "_")}'

        result, model = train._train_one(
            dir_model, modelname, args.embedder, args.set, args.translation,
            data, args.epochs, None, True,
            held_out, save_binary=False, patience=args.patience,
        )
        if result is None:
            print(f'[{held_out}] skipped (already trained)')
            continue

        scores, _, sens = train._write_scores(
            dir_model, model, args.set, args.embedder, held_out,
            data.translation, data.classes,
        )
        print(f'[{held_out}] k={args.k} weight={args.weight}: '
              f"{result['n_epochs']} epochs (best {result['best_epoch']}), "
              f"val_bce {result['best_val_bce']:.4f}, val_loss(tail) {result['best_val_loss']:.4f}, "
              f'{train._format_sens(sens)}')


if __name__ == '__main__':
    main()
