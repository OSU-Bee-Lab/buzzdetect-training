"""Generate YAMNet embeddings of white noise and save to augmentation directory.

Creates `augment_whitenoise/train/static.pickle` in the medium set's yamnet
embedding space. Each pickle entry is one 1024-d embedding frame.

White noise is generated at 16000 Hz (YAMNet native rate) and embedded with
framehop_prop=1 to match training extraction settings.
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from embedders.embedding import load_embedder

SETNAME = 'medium'
EMBEDDERNAME = 'yamnet'
AUG_DIRNAME = 'augment_whitenoise'
TARGET_FRAMES = 500
SR = 16000
RNG_SEED = 42

def main():
    embedder = load_embedder(EMBEDDERNAME, framehop_prop=1.0, initialize=True)

    rng = np.random.default_rng(RNG_SEED)

    # Generate white noise in 10s chunks and collect frames
    all_embeddings = []
    chunk_s = 10.0
    chunk_samples = int(chunk_s * SR)

    while len(all_embeddings) < TARGET_FRAMES:
        noise = rng.standard_normal(chunk_samples).astype(np.float32)
        # Normalize to unit RMS
        noise = noise / (np.sqrt(np.mean(noise ** 2)) + 1e-8)
        embs = embedder.embed(noise)  # shape: (n_frames, 1024)
        for i in range(embs.shape[0]):
            all_embeddings.append(embs[i])
            if len(all_embeddings) >= TARGET_FRAMES:
                break

    print(f'Generated {len(all_embeddings)} white noise frames')

    # Save to augment directory
    out_dir = os.path.join(
        cfg.dir_embeddings_augment(SETNAME, EMBEDDERNAME, AUG_DIRNAME),
        'train',
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'static.pickle')

    with open(out_path, 'wb') as f:
        for emb in all_embeddings:
            pickle.dump(emb, f)

    print(f'Saved to {out_path}')

if __name__ == '__main__':
    main()
