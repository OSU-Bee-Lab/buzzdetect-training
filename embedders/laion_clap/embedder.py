import os

import numpy as np

from embedders.embedding import BaseEmbedder

_MODEL_SUBDIR = 'clap-htsat-fused'


class EmbedderLaionClap(BaseEmbedder):
    embeddername = "laion_clap"
    framelength_s = 10.0
    digits_time = 1
    samplerate = 48000
    n_embeddings = 768
    dtype_in = 'float32'

    def initialize(self):
        import torch
        from transformers import ClapModel, ClapProcessor

        curdir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(curdir, _MODEL_SUBDIR)
        self.processor = ClapProcessor.from_pretrained(model_path)
        self.clap = ClapModel.from_pretrained(model_path)
        self.clap.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clap.to(self.device)
        self.framelength_samples = int(self.framelength_s * self.samplerate)
        self._torch = torch

    def embed(self, audio):
        n_frames = len(audio) // self.framelength_samples
        embeddings = np.empty((n_frames, self.n_embeddings), dtype='float32')
        for i in range(n_frames):
            frame = audio[i * self.framelength_samples:(i + 1) * self.framelength_samples]
            inputs = self.processor(audio=frame, sampling_rate=self.samplerate, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self._torch.no_grad():
                audio_out = self.clap.audio_model(
                    input_features=inputs['input_features'],
                    is_longer=inputs['is_longer']
                )
            embeddings[i] = audio_out.pooler_output[0].cpu().numpy()
        return embeddings
