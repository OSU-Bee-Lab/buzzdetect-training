import os

import numpy as np

from embedders.embedding import BaseEmbedder

"""YAMNet + AVES on one shared 1.0 s frame grid.

Each frame feeds the full 1.0 s to AVES (bioacoustic wav2vec2, dense 768-d) and
the centre 0.96 s to YAMNet (AudioSet-supervised, sparse 1024-d). The two blocks
are concatenated -> 1792-d, so the probe sees both representations for the same
audio window with the rows aligned by construction. The existing per-embedder
caches cannot be concatenated instead: hop is framelength_s * framehop_prop, so
the yamnet grid (0.96 s) and aves grid (1.0 s) diverge in frame count per chunk.

TF's GPU is hidden here when BUZZDETECT_NO_GPU=1 (YAMNet is tiny on CPU); AVES
keeps CUDA via its own torch path (BUZZDETECT_AVES_DEVICE), matching
embedders/aves. cuDNN is disabled for AVES for the same reason it is in
embedders/aves (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH under the pinned env).

--- RESTORE NOTE (2026-09-10) ---
This is the faithful original from the yamnet-aves session, rewritten from the
authoring transcript after embedders/yamnet_aves/embedder.py was lost with its
worktree (embedders/.gitignore was never unignored for it). `embeddername` is
kept as "yamnet_aves" so it maps onto the existing 560 MB medium cache.

The one detail the notes-only reconstruction (embedders/yamnet_aves/embedder.py,
"YAMNet block NOT exact") could not recover: the YAMNet block is produced by
feeding YAMNet ONE concatenated buffer of all the centred crops
(`self.yamnet(crops.reshape(-1))`) and letting its internal framing
(patch_hop = patch_window = 0.96 s) split it back into per-frame patches -- NOT
by calling `self.yamnet(crop)` once per frame. With a butt-joined buffer each
interior patch's trailing STFT window (~240 samples) pulls real audio from the
next crop; an isolated crop zero-pads that tail. That is the "~1-3% off on most
frames, exact on the occasional frame" the reconstruction measured -- the exact
ones are the last frame of each CHUNK_FRAMES (48) extraction block, where both
paths zero-pad. Verify byte-parity against the cache before trusting this for
re-extraction or shipping.

Reconciled 2026-09-10 on exp/yamnet-aves-head: this verified file replaced the
notes-only reconstruction at this path, embedders/.gitignore was unignored for
yamnet_aves, and both were committed. Byte-parity vs the cache confirmed
(YAMNet max |Δ| 5.6e-5 = GPU/CPU reduction-order noise; AVES exact) over all
3396 frames of the largest shared label pickle.
"""

_YAMNET_SAMPLES = 15360   # 0.96 s at 16 kHz, centred in the 1.0 s frame
_CROP0 = (16000 - _YAMNET_SAMPLES) // 2   # 320


class EmbedderYamnetAves(BaseEmbedder):
    embeddername = "yamnet_aves"
    framelength_s = 1.0
    digits_time = 2
    samplerate = 16000
    n_embeddings = 1792   # 1024 YAMNet + 768 AVES
    dtype_in = 'float32'

    def initialize(self):
        import tensorflow as tf
        from embedders.yamnet.yamnet import WaveformFeatures  # registers custom layer
        _ = WaveformFeatures.dtype

        if os.environ.get('BUZZDETECT_NO_GPU'):
            tf.config.set_visible_devices([], 'GPU')
        else:
            # Both blocks share the one 4 GB card with AVES's torch process. TF's
            # allocator is greedy by default and would claim the whole device,
            # OOMing AVES; memory growth holds it to what YAMNet inference needs
            # (~0.5-1 GB for the 1024-d conv net) and leaves the rest for torch.
            for gpu in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)

        here = os.path.dirname(os.path.realpath(__file__))
        yamnet_keras = os.path.join(here, '..', 'yamnet', 'yamnet.keras')
        self.yamnet = tf.keras.models.load_model(yamnet_keras, compile=False)
        # non-overlapping 0.96 s patches: one embedding per centred crop we feed
        self.yamnet.layers[1].params.patch_hop_seconds = 0.96

        import json
        import torch
        import torchaudio

        aves_dir = os.path.join(here, '..', 'aves')
        with open(os.path.join(aves_dir, 'model_config.json')) as f:
            config = json.load(f)
        self.aves = torchaudio.models.wav2vec2_model(**config, aux_num_out=None)
        state = torch.load(os.path.join(aves_dir, 'aves-base-bio.pt'),
                           map_location='cpu', weights_only=True)
        self.aves.load_state_dict(state)
        self.aves.eval()

        device = os.environ.get('BUZZDETECT_AVES_DEVICE', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.batch_size = int(os.environ.get('BUZZDETECT_AVES_BATCH', '64'))
        if device == 'cuda':
            torch.backends.cudnn.enabled = False
        self.aves.to(device)

        self.embed(np.zeros(self.samplerate, dtype=np.float32))

    def embed(self, audio):
        import torch

        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio) // self.samplerate
        if n == 0:
            return np.empty((0, self.n_embeddings), dtype=np.float32)
        frames = audio[:n * self.samplerate].reshape(n, self.samplerate)

        crops = frames[:, _CROP0:_CROP0 + _YAMNET_SAMPLES]
        yam = np.asarray(self.yamnet(crops.reshape(-1)), dtype=np.float32)
        if len(yam) != n:
            raise ValueError(
                f'yamnet_aves: YAMNet returned {len(yam)} frames for {n} crops'
            )

        out = []
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                stop = min(start + self.batch_size, n)
                wav = torch.from_numpy(frames[start:stop]).to(self.device)
                layer_outputs, _ = self.aves.extract_features(wav)
                out.append(layer_outputs[-1].mean(dim=1).cpu().numpy())
        av = np.concatenate(out, axis=0).astype(np.float32)

        return np.concatenate([yam, av], axis=1)
