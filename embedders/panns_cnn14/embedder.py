import os

import numpy as np

from embedders.embedding import BaseEmbedder

_BATCH_SIZE = 16


class EmbedderPannsCnn14(BaseEmbedder):
    embeddername = "panns_cnn14"
    framelength_s = 2.0
    digits_time = 1
    samplerate = 32000
    n_embeddings = 2048
    dtype_in = 'float32'

    def initialize(self):
        import torch
        from embedders.panns_cnn14.cnn14 import Cnn14
        curdir = os.path.dirname(os.path.realpath(__file__))
        checkpoint_path = os.path.join(curdir, 'CNN14_mAP=0.431.pth')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Cnn14()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        model.to(self.device)
        self.model = model

    def _embed_batch(self, frames):
        """frames: list of (framelength_samples,) float32 arrays → (n, 2048) numpy"""
        import torch
        x = torch.tensor(np.stack(frames), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            emb = self.model(x)
        return emb.cpu().numpy()

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
