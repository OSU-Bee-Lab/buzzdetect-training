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
    args = parser.parse_args()

    train_model(
        modelname=args.model,
        embeddername=args.embedder,
        setname=args.setname,
        name_translation=args.translation,
        epochs_max=args.epochs,
        aug_dirnames=args.aug_dirnames,
    )
