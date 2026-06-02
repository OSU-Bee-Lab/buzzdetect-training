import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='test')
    parser.add_argument('--set', default='lite', dest='setname')
    parser.add_argument('--embedder', default='yamnet')
    parser.add_argument('--translation', default='general')
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()

    train_model(
        modelname=args.model,
        embeddername=args.embedder,
        setname=args.setname,
        name_translation=args.translation,
        epochs_max=args.epochs,
    )
