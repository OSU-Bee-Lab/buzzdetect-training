"""yamnet_trunk_pitchshift_depth12 with the octave-UP view swapped for the
octave-DOWN one: each frame is embedded as [plain, octave-down] through ONE
shared fine-tuned depth12 tail (TimeDistributed, n_ctx=2), codes concatenated
into the head.

Proposal (Luke, 2026-09-26): `v4-ft-psud` ([plain, up, down]) tied `v4-ft-ps`
([plain, up]) at 0.452, so adding down to up was flat. This isolates the down
direction. The mechanism Luke proposed for it is a filter: halving every
frequency pushes truck and prop-plane rumble below YAMNet's 125 Hz mel floor,
where it vanishes, while buzz harmonics stay in range (the fundamental drops out
too, see yamnet_trunk_pitchshift_updown_depth12). If down alone is flat against
`v4-ft` (plain only), the direction is weak and the both-halves and vocoder
variants are skipped. Comparators: `v4-ft` (no shift) for the direction,
`v4-ft-ps` (up) for up vs down.

Down view is the updown embedder's `_pitch_down_octave` unchanged: the centre
half of the frame, upsampled 2:1 and relabelled 16 kHz. The outer quarters of
the frame are not seen by the down view.

No to_onnx(): CV only. Add an export before deploying this embedder.
"""
import importlib

import numpy as np

_psud = importlib.import_module('embedders.yamnet_trunk_pitchshift_updown_depth12.embedder')
_trunk12 = importlib.import_module('embedders.yamnet_trunk_depth12.embedder')


class EmbedderYamnetTrunkPitchshiftDownDepth12(_psud.EmbedderYamnetTrunkPitchshiftUpdownDepth12):
    embeddername = "yamnet_trunk_pitchshift_down_depth12"
    n_ctx = 2  # [plain, octave-down] -- shared trunk tail, TimeDistributed in build_head
    n_embeddings = _trunk12.EmbedderYamnetTrunkDepth12.n_embeddings * 2  # 24576

    def embed(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio) // self._frame_samples
        if n == 0:
            return np.empty((0, self.n_embeddings), dtype=np.float16)
        usable = audio[:n * self._frame_samples]

        # go straight to the trunk; the parents' embed() would add an up view
        trunk_embed = _trunk12.EmbedderYamnetTrunkDepth12.embed
        plain = trunk_embed(self, usable)
        if len(plain) != n:
            raise ValueError(
                f'{self.embeddername}: trunk returned {len(plain)} frames for {n} input frames'
            )

        frames = usable.reshape(n, self._frame_samples)
        down_frames = np.stack([self._pitch_down_octave(f) for f in frames])
        down = trunk_embed(self, down_frames.reshape(-1))

        return np.concatenate([plain, down], axis=1).astype(np.float16)
