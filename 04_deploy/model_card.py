"""The activation centers and a README skeleton for an already-trained model.

A training run with --train-shipped does this itself (03_train/thresholds.py).
This is for a model trained before it did, or to regenerate after changing
thresholds.py. It merges `activation_centers` and `center_stats` into
models/<name>/config_model.json and writes a README.md skeleton if there is
none; an existing README is left alone.

    conda run -n buzzdetect-train python 04_deploy/model_card.py <name>...

No TensorFlow: it reads the rotations' predictions off disk.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, '03_train'))

import config as cfg
from thresholds import write_model_card


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('names', nargs='+', metavar='MODEL')
    args = parser.parse_args()
    for name in args.names:
        dir_model = os.path.join(cfg.DIR_MODELS, name)
        if not os.path.exists(os.path.join(dir_model, 'config_model.json')):
            print(f'[{name}] no shipped model (config_model.json) in {dir_model}; skipped')
            continue
        write_model_card(dir_model, name)


if __name__ == '__main__':
    main()
