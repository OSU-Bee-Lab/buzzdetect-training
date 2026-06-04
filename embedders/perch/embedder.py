import os

import numpy as np

from embedders.embedding import BaseEmbedder

_CPU_SUBDIR = 'perch_v2_cpu'
_GPU_SUBDIR = 'perch_v2_gpu'


class EmbedderPerch(BaseEmbedder):
    embeddername = "perch"
    framelength_s = 5.0
    digits_time = 1
    samplerate = 32000
    n_embeddings = 1536
    dtype_in = 'float32'

    def initialize(self):
        import tensorflow as tf
        from ml_collections import config_dict
        from perch_hoplite.zoo import taxonomy_model_tf

        curdir = os.path.dirname(os.path.realpath(__file__))
        has_gpu = bool(tf.test.gpu_device_name())
        subdir = _GPU_SUBDIR if has_gpu else _CPU_SUBDIR
        model_path = os.path.join(curdir, subdir)

        cfg = config_dict.ConfigDict({
            'model_path': model_path,
            'sample_rate': self.samplerate,
            'window_size_s': self.framelength_s,
            'hop_size_s': self.framehop_s,
            'target_peak': 0.25,
        })
        self.model = taxonomy_model_tf.TaxonomyModelTF.from_config(cfg)

        # Force TF thread pool initialization before librosa/Accelerate claims threads
        self.embed(np.zeros(int(self.framelength_s * self.samplerate), dtype=np.float32))

    def embed(self, audio):
        outputs = self.model.embed(audio)
        # embeddings shape: (n_frames, channels=1, features=1536)
        return outputs.embeddings[:, 0, :]
