import os

import numpy as np

from embedders.recipe import Aves, Branch, Keras, Recipe, RecipeEmbedder, Slice

"""YAMNet + AVES, with AVES pooled by [mean, max, std] over its token axis
instead of mean alone.

IDEAS.md item 3. `embedders/yamnet_aves/embedder.py` takes wav2vec2's
last-layer output `(B, T', 768)` and does `.mean(dim=1)` -- a mean over ~49
tokens of 20 ms each. Mean pooling is a matched filter for a *stationary*
signal and the worst pooling for a *transient* one; a buzz occupying 200 ms of
a 1.0 s frame has its evidence divided by ~5 before the probe ever sees it.
`max` is the parameter-free form of what attentive pooling recovers for
transformer probes in the literature (see the idea for citations); `std`
adds a cheap measure of how non-stationary the frame's activations are, which
a pure mean or max both discard. Concatenating all three costs nothing beyond
the trivial extra reduction ops -- the token sequence itself is never cached.

Re-aimed 2026-09-11 (see IDEAS.md): this targets `untagged`/`loud` sensitivity
on folds weak on *audible* buzz, not quiet-buzz recall, which is out of the
headline. Otherwise identical in structure to `yamnet_aves`: same YAMNet half
(centre-cropped 0.96 s of AVES's 1.0 s frame, on one shared grid), same model
weights, same device/memory-growth handling. n_embeddings = 1024 + 3*768 =
3328.

This is a standalone copy of `yamnet_aves/embedder.py`, not a subclass of it --
`load_embedder()` scans a module's namespace for exactly one `BaseEmbedder`
subclass, so importing `EmbedderYamnetAves` into this module's namespace would
give it two candidates to choose between.
"""

_YAMNET_SAMPLES = 15360   # 0.96 s at 16 kHz, centred in the 1.0 s frame
_CROP0 = (16000 - _YAMNET_SAMPLES) // 2   # 320


class EmbedderYamnetAvesP3(RecipeEmbedder):
    embeddername = "yamnet_aves_p3"
    framelength_s = 1.0
    digits_time = 2
    samplerate = 16000
    n_embeddings = 1024 + 768 * 3   # 1024 YAMNet + [mean, max, std] over AVES tokens
    dtype_in = 'float32'
    # YAMNet on the centred 0.96 s of each 1.0 s frame, AVES on all of it
    recipe = Recipe(branches=(Branch(Keras('yamnet'), (Slice(_CROP0, _CROP0 + _YAMNET_SAMPLES),)),
                              Branch(Aves(pool='stats'))))

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
