import os
import numpy as np
from embedders.embedding import BaseEmbedder


class EmbedderBirdnet(BaseEmbedder):
    embeddername = "birdnet"
    framelength_s = 3.0
    digits_time = 1
    samplerate = 48000
    n_embeddings = 1024
    dtype_in = 'float32'

    def initialize(self):
        import tensorflow as tf
        curdir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(curdir, 'audio-model.tflite')
        # experimental_preserve_all_tensors is needed to read the embedding tensor
        # (GLOBAL_AVG_POOL output), which is intermediate and not preserved by default
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            experimental_preserve_all_tensors=True
        )
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.input_idx = input_details[0]['index']
        # Embedding tensor is one step before the classification head (Identity output)
        self.embedding_idx = output_details[0]['index'] - 1
        self.framelength_samples = int(self.framelength_s * self.samplerate)

    def embed(self, audio):
        # Frames are processed one at a time. TFLite supports variable batch sizes via
        # resize_tensor_input + allocate_tensors, but reallocating per chunk is expensive.
        # If this becomes a bottleneck, pre-allocate for a fixed batch size instead.
        n_frames = len(audio) // self.framelength_samples
        embeddings = np.empty((n_frames, self.n_embeddings), dtype='float32')
        for i in range(n_frames):
            frame = audio[i * self.framelength_samples:(i + 1) * self.framelength_samples]
            self.interpreter.set_tensor(self.input_idx, frame.reshape(1, -1))
            self.interpreter.invoke()
            embeddings[i] = self.interpreter.get_tensor(self.embedding_idx)[0]
        return embeddings
