"""yamnet_trunk_pitchshift_depth12 plus a third, octave-DOWN view: each frame is
embedded as [plain, octave-up, octave-down], all three through ONE shared
fine-tuned depth12 tail (TimeDistributed, n_ctx=3), codes concatenated into the
head.

Proposal (Luke, 2026-09-26): `v4-ft-ps` (octave-up twin) won the v4 grid at
0.452 vs `v4-ft` 0.375. Octave-up moves buzz fundamentals (~100-400 Hz) into a
band YAMNet's AudioSet features resolve better. Octave-down is the other
direction: it pushes harmonics above 4 kHz back into range, and stretches each
buzz to twice its length. Whether the two views add is the question.
Comparator: `v4-ft-ps`, same config otherwise.

Octave-down mechanism, mirroring the up shift: take the centre half of the
0.96 s frame (0.24-0.72 s), upsample it 2:1 with librosa and relabel the
result as 16 kHz. That gives a full 0.96 s patch at half pitch and half speed,
centred on the same instant as the plain view, so the frame grid is unchanged.
The outer quarters of the frame are not in this view. The up view still covers
them, tiled.

No to_onnx(): CV only. Add an export before deploying this embedder.
"""
import importlib

import numpy as np

_ps12 = importlib.import_module('embedders.yamnet_trunk_pitchshift_depth12.embedder')
_trunk12 = importlib.import_module('embedders.yamnet_trunk_depth12.embedder')


class EmbedderYamnetTrunkPitchshiftUpdownDepth12(_ps12.EmbedderYamnetTrunkPitchshiftDepth12):
    embeddername = "yamnet_trunk_pitchshift_updown_depth12"
    n_ctx = 3  # [plain, octave-up, octave-down] -- shared trunk tail, TimeDistributed in build_head
    n_embeddings = _trunk12.EmbedderYamnetTrunkDepth12.n_embeddings * 3  # 36864

    def _pitch_down_octave(self, frame):
        q = self._frame_samples // 4
        centre = frame[q:q + self._frame_samples // 2]
        shifted = self._librosa.resample(
            centre, orig_sr=self.samplerate, target_sr=self.samplerate * 2,
        ).astype(np.float32)
        if len(shifted) < self._frame_samples:
            shifted = np.pad(shifted, (0, self._frame_samples - len(shifted)))
        return shifted[:self._frame_samples]

    def embed(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio) // self._frame_samples
        if n == 0:
            return np.empty((0, self.n_embeddings), dtype=np.float16)
        usable = audio[:n * self._frame_samples]

        # plain + octave-up, (n, 24576) float16, from the parent
        plain_up = super().embed(usable)
        if len(plain_up) != n:
            raise ValueError(
                f'{self.embeddername}: parent returned {len(plain_up)} frames for {n} input frames'
            )

        frames = usable.reshape(n, self._frame_samples)
        down_frames = np.stack([self._pitch_down_octave(f) for f in frames])
        # skip the parent's embed() (it would add its own up view); go straight to the trunk
        down = _trunk12.EmbedderYamnetTrunkDepth12.embed(self, down_frames.reshape(-1))

        return np.concatenate([plain_up, down], axis=1).astype(np.float16)
