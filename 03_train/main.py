# TensorFlow must be imported before pandas/pyarrow. pandas eagerly imports
# pyarrow, and pyarrow + TF each bundle their own statically-linked abseil; the
# shared lib that loads first wins abseil's weak synchronization symbols
# process-wide. If libarrow wins, TF's in-graph FFT (ducc0) threadpool ends up
# waiting on libarrow's incompatible semaphore impl and deadlocks. Keep this
# import first, ahead of anything (incl. train.py) that pulls in pandas.
import tensorflow  # noqa: F401  -- imported for load-order side effect only

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--set', required=True, dest='setname')
    parser.add_argument('--embedder', required=True)
    parser.add_argument('--translation', required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--augment', nargs='*', dest='aug_dirnames', metavar='AUG_DIRNAME')
    parser.add_argument('--dropout', type=float, default=0.2, dest='dropout_rate')
    parser.add_argument('--label-smoothing', type=float, default=0.2)
    parser.add_argument('--learning-rate', type=float, default=0.002)
    args = parser.parse_args()

    train_model(
        modelname=args.model,
        embeddername=args.embedder,
        setname=args.setname,
        name_translation=args.translation,
        epochs_max=args.epochs,
        aug_dirnames=args.aug_dirnames,
        dropout_rate=args.dropout_rate,
        label_smoothing=args.label_smoothing,
        learning_rate=args.learning_rate,
    )
