import json
import os

import numpy as np

from embedders.embedding import BaseEmbedder


class EmbedderAves(BaseEmbedder):
    embeddername = "aves"
    framelength_s = 1.0
    digits_time = 2
    samplerate = 16000
    n_embeddings = 768
    dtype_in = 'float32'

    def initialize(self):
        import torch
        import torchaudio

        curdir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(curdir, 'model_config.json')
        model_path = os.path.join(curdir, 'aves-base-bio.pt')

        with open(config_path) as f:
            config = json.load(f)

        self.model = torchaudio.models.wav2vec2_model(**config, aux_num_out=None)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def embed(self, audio):
        framelength_samples = int(self.framelength_s * self.samplerate)
        hop_samples = int(self.framehop_s * self.samplerate)

        n_frames = max(0, (len(audio) - framelength_samples) // hop_samples + 1)
        if n_frames == 0:
            return np.empty((0, self.n_embeddings), dtype=np.float32)

        import torch

        results = []
        with torch.no_grad():
            for i in range(n_frames):
                start = i * hop_samples
                chunk = audio[start:start + framelength_samples]
                waveform = torch.from_numpy(chunk).unsqueeze(0)  # (1, T)
                # extract_features returns (list_of_layer_outputs, padding_mask)
                layer_outputs, _ = self.model.extract_features(waveform)
                # last layer: (1, T', 768) — mean-pool over time
                embedding = layer_outputs[-1].mean(dim=1).squeeze(0).numpy()
                results.append(embedding)

        return np.stack(results, axis=0)  # (n_frames, 768)
