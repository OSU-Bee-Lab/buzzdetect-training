import os
import sys

import numpy as np

from embedders.embedding import BaseEmbedder

_WEIGHTS_FILE = 'CLAP_weights_2023.pth'


class EmbedderMsclap(BaseEmbedder):
    embeddername = "msclap"
    framelength_s = 7.0
    digits_time = 1
    samplerate = 44100
    n_embeddings = 1024
    dtype_in = 'float32'

    def initialize(self):
        import torch
        curdir = os.path.dirname(os.path.realpath(__file__))

        # Add the embedder dir to sys.path so relative imports in the local models package work
        if curdir not in sys.path:
            sys.path.insert(0, curdir)

        from models.audio import get_audio_encoder
        from models.clap import AudioEncoder

        audio_encoder = AudioEncoder(
            audioenc_name='HTSAT',
            d_in=768,
            d_out=1024,
            sample_rate=44100,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=8000,
            classes_num=527,
        )

        weights_path = os.path.join(curdir, _WEIGHTS_FILE)
        ckpt = torch.load(weights_path, map_location='cpu')
        full_state = ckpt['model']
        audio_state = {
            k[len('audio_encoder.'):]: v
            for k, v in full_state.items()
            if k.startswith('audio_encoder.')
        }
        audio_encoder.load_state_dict(audio_state)
        audio_encoder.eval()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.audio_encoder = audio_encoder.to(self.device)
        self.framelength_samples = int(self.framelength_s * self.samplerate)

    def embed(self, audio):
        import torch
        hop_samples = int(self.framehop_s * self.samplerate)
        n_samples = len(audio)

        starts = range(0, n_samples - self.framelength_samples + 1, hop_samples)
        if not starts:
            return np.empty((0, self.n_embeddings), dtype='float32')

        frames = []
        for s in starts:
            frame = audio[s:s + self.framelength_samples]
            frames.append(frame)

        batch = torch.tensor(np.stack(frames), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            projected, _ = self.audio_encoder(batch)

        return projected.cpu().numpy()
