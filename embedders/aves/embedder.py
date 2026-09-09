import json
import os

import numpy as np

from embedders.embedding import BaseEmbedder


class EmbedderAves(BaseEmbedder):
    """AVES (aves-base-bio), a wav2vec2 pretrained on bioacoustics.

    Batched, and on the GPU when there is one. The original implementation ran
    one 1.0 s frame per forward pass on the CPU, which is 0.19 s/frame — 14 h
    for `medium`'s 268k frames. Batching to 64 on a GTX 1650 is 0.0139 s/frame,
    ~1.0 h, at 1.15 GB peak VRAM. Throughput is already flat at batch 16, so a
    larger batch buys only VRAM pressure, and a second worker process buys no
    throughput at all — the GPU is saturated by one.

    Tunable by env var: BUZZDETECT_AVES_DEVICE (auto/cpu/cuda),
    BUZZDETECT_AVES_BATCH (default 64).
    """

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

        device = os.environ.get('BUZZDETECT_AVES_DEVICE', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.batch_size = int(os.environ.get('BUZZDETECT_AVES_BATCH', '64'))

        if device == 'cuda':
            # cuDNN in the pinned env cannot finalize this model's conv1d
            # descriptors (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH). Torch's
            # native kernels are the same speed here — 0.0139 vs 0.0140
            # s/frame — so disable cuDNN rather than repin anything. Repinning
            # would drag TF 2.16.2 with it and break era comparability.
            torch.backends.cudnn.enabled = False

        self.model.to(device)

    def embed(self, audio):
        framelength_samples = int(self.framelength_s * self.samplerate)
        hop_samples = int(self.framehop_s * self.samplerate)

        n_frames = max(0, (len(audio) - framelength_samples) // hop_samples + 1)
        if n_frames == 0:
            return np.empty((0, self.n_embeddings), dtype=np.float32)

        import torch

        results = []
        with torch.no_grad():
            for start in range(0, n_frames, self.batch_size):
                stop = min(start + self.batch_size, n_frames)
                # Sliced per batch rather than all at once: a full ident is
                # ~8640 frames, and materialising every frame would be ~550 MB
                # of float32 before a single forward pass.
                batch = np.stack([
                    audio[i * hop_samples:i * hop_samples + framelength_samples]
                    for i in range(start, stop)
                ]).astype(np.float32)
                waveform = torch.from_numpy(batch).to(self.device)
                # extract_features returns (list_of_layer_outputs, padding_mask)
                layer_outputs, _ = self.model.extract_features(waveform)
                # last layer: (B, T', 768) — mean-pool over time
                results.append(layer_outputs[-1].mean(dim=1).cpu().numpy())

        return np.concatenate(results, axis=0).astype(np.float32)  # (n_frames, 768)
