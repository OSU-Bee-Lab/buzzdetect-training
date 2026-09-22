import importlib

import numpy as np

import config as cfg

# Through the module, so load_embedder()'s scan finds only this class.
_trunk = importlib.import_module('embedders.yamnet_trunk.embedder')

CONTEXT_FRAMES = 1


class EmbedderYamnetTrunkContext(_trunk.EmbedderYamnetTrunk):
    """yamnet_trunk (layer-12 maps, fine-tunable layers 13-14) with temporal context.

    Each frame carries concat(trunk t-1, trunk t, trunk t+1) = 3 x 12288 float16.
    build_head() runs ONE shared copy of layers 13-14 over the three frames, so
    fine-tuning sees the context (the stack of yamnet_context, but pre-GAP), then
    concatenates the three 1024-d codes into the Dense. Context comes from
    contiguous audio, as in yamnet_context (02_set/extract.py reads context_frames).
    """

    embeddername = "yamnet_trunk_context"
    context_frames = CONTEXT_FRAMES
    n_ctx = 2 * CONTEXT_FRAMES + 1
    n_embeddings = _trunk.EmbedderYamnetTrunk.n_embeddings * (2 * CONTEXT_FRAMES + 1)

    def n_frames(self, n_samples):
        length = int(round(self.framelength_s * self.samplerate))
        hop = int(round(self.framehop_s * self.samplerate))
        if n_samples < length:
            return 0
        return (n_samples - length) // hop + 1

    def embed_frames(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        length = int(round(self.framelength_s * self.samplerate))
        hop = int(round(self.framehop_s * self.samplerate))
        n = self.n_frames(len(audio))
        width = _trunk.EmbedderYamnetTrunk.n_embeddings
        if n == 0:
            return np.zeros((0, width), dtype=np.float16)
        out = []
        for lo in range(0, n, cfg.CHUNK_FRAMES):
            hi = min(lo + cfg.CHUNK_FRAMES, n)
            piece = audio[lo * hop: (hi - 1) * hop + length]
            emb = super().embed(piece)
            if len(emb) != hi - lo:
                raise ValueError(f'{self.embeddername}: got {len(emb)} frames, expected {hi - lo}')
            out.append(emb)
        return np.concatenate(out, axis=0)

    def stack_context(self, emb):
        k = self.context_frames
        n = len(emb)
        if n == 0:
            return np.zeros((0, self.n_embeddings), dtype=np.float16)
        idx = np.arange(n)
        return np.concatenate([emb[np.clip(idx + o, 0, n - 1)] for o in range(-k, k + 1)], axis=1)

    def embed(self, audio):
        return self.stack_context(self.embed_frames(audio))

    def to_onnx(self, opset=17):
        """The trunk's own Keras export, cropped to whole frames the way
        embed_frames() crops its input, plus the context stack built onto the
        graph (mirrors yamnet_context.to_onnx; 04_deploy's parity check is the test)."""
        from embedders.onnx_context import add_context_stack, crop_waveform_to_whole_frames

        frame_samples = int(round(self.framelength_s * self.samplerate))
        trunk_onnx = super().to_onnx(opset=opset)
        trunk_onnx = crop_waveform_to_whole_frames(trunk_onnx, frame_samples)
        width = _trunk.EmbedderYamnetTrunk.n_embeddings
        return add_context_stack(
            trunk_onnx, k=self.context_frames,
            widen_dim=width, total_dim=width, n_embeddings=self.n_embeddings)
