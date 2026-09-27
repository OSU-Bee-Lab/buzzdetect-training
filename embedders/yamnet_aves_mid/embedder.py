import os

import numpy as np

from embedders.recipe import Aves, Branch, Keras, Recipe, RecipeEmbedder, Slice

"""YAMNet + AVES, reading AVES layers 6, 9 and 12 instead of layer 12 alone.

IDEAS.md item 4. `embedders/yamnet_aves/embedder.py` reads only wav2vec2's
final layer (`layer_outputs[-1]`) -- the layer AVES's own pretraining task
shaped most, and per the SSL-transfer literature (cited in the idea) not
necessarily the layer richest in transferable information for a non-bird,
non-pretraining target. `extract_features` already computes every layer; the
existing embedder discards eleven of them at no saving. This mean-pools
layers 6, 9 and 12 (indices 5, 8, 11 into the 12-element `layer_outputs`
list) and concatenates them, letting the probe's own weights decide how much
of each to use -- the cheapest honest form of a layer sweep, one run instead
of three.

1024 YAMNet + 3*768 AVES (one per layer) = 3328-d, same shape as
`yamnet_aves_p3` (item 3) -- the two are mutually informative about where
AVES's capacity actually is (pooling statistics vs. depth) but were run as
separate experiments, not stacked.
"""

_YAMNET_SAMPLES = 15360   # 0.96 s at 16 kHz, centred in the 1.0 s frame
_CROP0 = (16000 - _YAMNET_SAMPLES) // 2   # 320
_LAYER_INDICES = (5, 8, 11)   # 0-indexed: AVES layers 6, 9, 12


class EmbedderYamnetAvesMid(RecipeEmbedder):
    embeddername = "yamnet_aves_mid"
    framelength_s = 1.0
    digits_time = 2
    samplerate = 16000
    n_embeddings = 1024 + 768 * len(_LAYER_INDICES)   # 1024 YAMNet + layers {6,9,12} mean-pooled
    dtype_in = 'float32'
    # YAMNet on the centred 0.96 s of each 1.0 s frame, AVES on all of it
    recipe = Recipe(branches=(Branch(Keras('yamnet'), (Slice(_CROP0, _CROP0 + _YAMNET_SAMPLES),)),
                              Branch(Aves(layers=_LAYER_INDICES))))

    def initialize(self):
        import tensorflow as tf
        from embedders.yamnet.yamnet import WaveformFeatures  # registers custom layer
        _ = WaveformFeatures.dtype

        if os.environ.get('BUZZDETECT_NO_GPU'):
            tf.config.set_visible_devices([], 'GPU')
        else:
            for gpu in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)

        here = os.path.dirname(os.path.realpath(__file__))
        yamnet_keras = os.path.join(here, '..', 'yamnet', 'yamnet.keras')
        self.yamnet = tf.keras.models.load_model(yamnet_keras, compile=False)
        self.yamnet.layers[1].params.patch_hop_seconds = 0.96

        import json
        import torch
        import torchaudio

        aves_dir = os.path.join(here, '..', 'aves')
        with open(os.path.join(aves_dir, 'model_config.json')) as f:
            config = json.load(f)
        self.aves = torchaudio.models.wav2vec2_model(**config, aux_num_out=None)
        state = torch.load(os.path.join(aves_dir, 'aves-base-bio.pt'),
                           map_location='cpu', weights_only=True)
        self.aves.load_state_dict(state)
        self.aves.eval()

        device = os.environ.get('BUZZDETECT_AVES_DEVICE', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.batch_size = int(os.environ.get('BUZZDETECT_AVES_BATCH', '64'))
        if device == 'cuda':
            torch.backends.cudnn.enabled = False
        self.aves.to(device)

        self.embed(np.zeros(self.samplerate, dtype=np.float32))
