import os

import numpy as np

from embedders.embedding import BaseEmbedder

_BATCH_SIZE = 8


class EmbedderAst(BaseEmbedder):
    embeddername = "ast"
    framelength_s = 10.24   # 1024 frames × 10 ms hop
    digits_time = 2
    samplerate = 16000
    n_embeddings = 768
    dtype_in = 'float32'

    def initialize(self):
        import torch
        from transformers import AutoFeatureExtractor, ASTModel

        curdir = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.join(curdir, 'model')

        self.extractor = AutoFeatureExtractor.from_pretrained(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ASTModel.from_pretrained(model_dir)
        model.eval()
        model.to(self.device)
        self.model = model
        self._torch = torch

    def _embed_batch(self, chunks):
        """chunks: list of 1-D float32 arrays (each up to framelength_samples long)"""
        inputs = self.extractor(
            chunks,
            sampling_rate=self.samplerate,
            return_tensors='pt',
            padding='max_length',
        )
        input_values = inputs['input_values'].to(self.device)
        with self._torch.no_grad():
            out = self.model(input_values=input_values)
        return out.pooler_output.cpu().numpy()

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
                results.append(self._embed_batch(batch))
                batch = []
        if batch:
            results.append(self._embed_batch(batch))

        return np.concatenate(results, axis=0)
