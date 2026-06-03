import os

import numpy as np

from embedders.embedding import BaseEmbedder

class YamnetK2(BaseEmbedder):
    # Class attributes - no config file needed!
    embeddername = "yamnet"
    framelength_s = 0.96  # seconds
    digits_time = 2
    samplerate = 16000  # Hz
    n_embeddings = 1024
    dtype_in = 'float32'

    def initialize(self):
        from keras.layers import TFSMLayer
        if self.framehop_prop == 1:
            dir_model = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models/yamnet_wholehop')
        elif self.framehop_prop == 0.5:
            dir_model = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models/yamnet_halfhop')
        else:
            raise ValueError('For Keras 2 YAMNet, framehop_prop must be 1 or 0.5')
            # TODO: hop on command; totally worth it with how much faster k2 is

        self.model = TFSMLayer(dir_model, call_endpoint='serving_default')
        # Force TF thread pool init before librosa/Accelerate claims threads; without
        # this the first embed() call after librosa.resample deadlocks on macOS.
        self.embed(np.zeros(int(self.framelength_s * self.samplerate), dtype=np.float32))

    def embed(self, audiosamples):
        """
        Generate embeddings for audio data

        Args:
            audiosamples: numpy array of audio samples at self.samplerate

        Returns:
            numpy array of embeddings
        """
        return self.model(audiosamples)['global_average_pooling2d']