import os
import importlib

import numpy as np

from embedders.recipe import Branch, Keras, Recipe, RecipeEmbedder

# Imported through the module so load_embedder()'s scan for a BaseEmbedder
# subclass in this namespace only finds EmbedderYamnetTriple.
_yamnet = importlib.import_module('embedders.yamnet.embedder')

"""YAMNet at 3x frame length: one 2.88 s frame = three contiguous 0.96 s
YAMNet patches, their embeddings concatenated in time order (3072-d).

Not a context model: each row is its own frame with its own label, and rows
don't overlap or borrow from neighbours. The classifier just sees a wider
frame.

The trunk does the same work per second of audio as plain `yamnet` at
framehop 1 -- three YAMNet patches per 2.88 s either way. The saving is 3x
fewer rows downstream (head, thresholds, results) and nothing else. A frame
also dilutes a short buzz across 2.88 s; labels follow framelength.

Lean path: the whole thing is one TF call. The waveform is framed into
2.88 s windows with tf.signal.frame, flattened, handed to YAMNet with its
patch hop pinned at 0.96 s, and the (3n, 1024) output reshaped to
(n, 3072) -- all on-device, one host copy at the end. At framehop_prop 1 the
framing is a no-op reshape and the result is bit-identical to plain YAMNet's
patches regrouped by three.
"""

SUBFRAMES = 3


class EmbedderYamnetTriple(RecipeEmbedder, _yamnet.EmbedderYamnet):
    embeddername = "yamnet_triple"
    framelength_s = _yamnet.EmbedderYamnet.framelength_s * SUBFRAMES  # 2.88
    n_embeddings = _yamnet.EmbedderYamnet.n_embeddings * SUBFRAMES    # 3072
    # one 2.88 s frame is three YAMNet patches, laid side by side
    recipe = Recipe(branches=(Branch(Keras(patches=SUBFRAMES)),))

    def initialize(self):
        import tensorflow as tf
        # Not super().initialize(): it pins the patch hop to self.framehop_s
        # (2.88 s here) and warms up through embed() before we could fix it.
        from embedders.yamnet.yamnet import WaveformFeatures  # registers custom layer
        _ = WaveformFeatures.dtype
        model_path = os.path.join(os.path.dirname(os.path.realpath(_yamnet.__file__)), 'yamnet.keras')
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model.layers[1].params.patch_hop_seconds = _yamnet.EmbedderYamnet.framelength_s
        # warm the TF thread pool before librosa/Accelerate claims threads
        self.embed(np.zeros(int(round(self.framelength_s * self.samplerate)), dtype=np.float32))
