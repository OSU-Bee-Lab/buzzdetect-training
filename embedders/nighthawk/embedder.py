import os

import numpy as np

from embedders.embedding import BaseEmbedder

# ImageNet normalization constants applied after the spectrogram is scaled to [0,1]
_IMAGENET_MEAN = [-0.485, -0.456, -0.406]
_IMAGENET_STD  = [0.229,  0.224,  0.225]

_BATCH_SIZE = 64  # frames per backbone call


class EmbedderNighthawk(BaseEmbedder):
    embeddername = "nighthawk"
    framelength_s = 1.0
    digits_time = 2
    samplerate = 22050
    n_embeddings = 512
    dtype_in = 'float32'

    def initialize(self):
        import tensorflow as tf
        curdir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(curdir, 'saved_model_with_preprocessing')
        outer = tf.saved_model.load(model_path)

        self._preprocess = outer.preprocess  # (22050,) -> (128, 171) mel spectrogram

        inner = outer.model
        # Three normalization layers that precede the backbone in the Keras graph
        # (keras_metadata.pb: normalize_layer → db_layer → normalize_layer_1)
        self._norm1   = getattr(inner, 'layer-1')  # [0, 1]
        self._dblayer = getattr(inner, 'layer-3')  # [-80, 0] dB
        self._norm2   = getattr(inner, 'layer-4')  # [0, 1]

        # ResNet34 backbone; expects (N, 128, 171, 3) with ImageNet normalization
        self._backbone = getattr(inner, 'layer_with_weights-0')

        self._tf = tf
        self._imagenet_mean = tf.constant(_IMAGENET_MEAN, dtype=tf.float32)
        self._imagenet_std  = tf.constant(_IMAGENET_STD,  dtype=tf.float32)

        # Force TF thread pool initialization before any other ML library (e.g. librosa/Accelerate)
        # claims the threads; without this, the first embed() call after librosa.resample deadlocks.
        self.embed(np.zeros(self.samplerate, dtype=np.float32))

    def _frames_to_embeddings(self, frames):
        """
        frames: list of (22050,) float32 numpy arrays
        returns: (n, 512) numpy array
        """
        tf = self._tf
        specs = tf.stack([self._preprocess(tf.constant(f)) for f in frames])
        specs = tf.expand_dims(specs, axis=-1)          # (n, 128, 171, 1)

        x = self._norm1(specs)                          # [0, 1]
        x = self._dblayer(x)                            # [-80, 0]
        x = self._norm2(x)                              # [0, 1]
        x = tf.concat([x, x, x], axis=-1)              # (n, 128, 171, 3)
        x = tf.nn.bias_add(x, self._imagenet_mean)
        x = x / self._imagenet_std

        return self._backbone(x, False, None).numpy()   # (n, 512)

    def embed(self, audio):
        framelength_samples = int(self.framelength_s * self.samplerate)
        hop_samples = int(self.framehop_s * self.samplerate)

        n_frames = max(0, (len(audio) - framelength_samples) // hop_samples + 1)
        if n_frames == 0:
            return np.empty((0, self.n_embeddings), dtype=np.float32)

        results = []
        batch = []
        for i in range(n_frames):
            start = i * hop_samples
            batch.append(audio[start:start + framelength_samples])
            if len(batch) == _BATCH_SIZE:
                results.append(self._frames_to_embeddings(batch))
                batch = []
        if batch:
            results.append(self._frames_to_embeddings(batch))

        return np.concatenate(results, axis=0)
