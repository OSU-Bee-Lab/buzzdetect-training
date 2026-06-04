import os

import numpy as np

from embedders.embedding import BaseEmbedder

"""Feature computation for YAMNet."""

class EmbedderYamnet(BaseEmbedder):
    embeddername = "yamnet_doublerate"
    framelength_s = 0.48  # seconds
    digits_time = 2
    samplerate = 32000  # Hz
    n_embeddings = 1024
    dtype_in = 'float32'

    def initialize(self):
        """Load the YAMNet model from TensorFlow Hub"""
        import tensorflow as tf
        from tensorflow.keras import Model
        from embedders.yamnet.yamnet import WaveformFeatures  # unused import is NEEDED for model loading
        _ = WaveformFeatures.dtype
        curdir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(curdir, 'yamnet.keras')
        model = tf.keras.models.load_model(model_path, compile=False)
        model.layers[1].params.patch_hop_seconds = self.framehop_s
        self.model = model
        # Force TF thread pool init before librosa/Accelerate claims threads; without
        # this the first embed() call after librosa.resample deadlocks on macOS.
        self.embed(np.zeros(int(self.framelength_s * self.samplerate), dtype=np.float32))

    def embed(self, audio):
        """
        Generate embeddings for audio data

        Args:
            audio: numpy array of audio samples at self.samplerate

        Returns:
            numpy array of embeddings
        """
        return self.model(audio)