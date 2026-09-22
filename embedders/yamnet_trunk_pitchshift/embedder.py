"""yamnet_trunk (layer-12 maps, fine-tunable layers 13-14), plus the same
octave-up pitch-shifted twin `yamnet_pitchshift` concatenates -- but run
through the SAME fine-tuned trunk tail via the shared-weight TimeDistributed
mechanism `yamnet_trunk_context` already built for temporal neighbours.

IDEAS proposal, new (2026-09-21): this era's two largest confirmed single
levers are `yamnet-pitchshift` (concat with an octave-up twin, +0.069, frozen
probe) and `trunk-ft-v3` (unfreezing layers 13-14, +0.077, plain audio). They
have never been combined. `trunk-ft-v3`'s own context variant
(`yamnet_trunk_context`) showed *temporal* context does not stack with
fine-tuning (+0.002 over trunk3-ft-1e5) -- but pitch-shift is a different
axis (same timestamp, transformed audio), already shown independently
additive with AVES depth (`pitchshift-aves-mid` beat both parents). This is
the first test of whether it is additive with backbone fine-tuning too.

Each frame's trunk (layer12) features are computed TWICE per frame -- once
on the plain 0.96 s audio, once on its octave-up resample+tile twin (same
mechanism as `yamnet_pitchshift`, ported unchanged) -- and cached as one
2 x 12288 float16 row. `build_head()`'s existing `n_ctx` path (a single
shared copy of layers 13-14, `TimeDistributed`, inherited unchanged from
`yamnet_trunk`) then fine-tunes ONE trunk tail jointly across both views
before concatenating their 1024-d codes into the Dense head. No temporal
neighbour is read; only the pitch-shift mechanism from `yamnet_pitchshift`
and the shared-tail mechanism from `yamnet_trunk_context`, combined.
"""
import importlib

import numpy as np

_trunk = importlib.import_module('embedders.yamnet_trunk.embedder')
_ps = importlib.import_module('embedders.yamnet_pitchshift.embedder')


class EmbedderYamnetTrunkPitchshift(_trunk.EmbedderYamnetTrunk):
    embeddername = "yamnet_trunk_pitchshift"
    n_ctx = 2  # [plain, octave-up] -- shared trunk tail, TimeDistributed in build_head
    n_embeddings = _trunk.EmbedderYamnetTrunk.n_embeddings * 2  # 24576

    def initialize(self):
        import librosa
        self._librosa = librosa
        self._frame_samples = int(round(self.framelength_s * self.samplerate))
        self._half_sr = self.samplerate // 2
        super().initialize()  # loads self.model = trunk through layer12, warms it (calls embed())

    # identical mechanism to yamnet_pitchshift._pitch_up_octave -- resample to
    # half-rate (relabelled 16 kHz), tile 2x, crop to one frame. Held as its
    # own copy (not inherited) since this class's MRO runs through
    # EmbedderYamnetTrunk, not EmbedderYamnetPitchshift.
    def _pitch_up_octave(self, frame):
        shifted = self._librosa.resample(
            frame, orig_sr=self.samplerate, target_sr=self._half_sr,
        ).astype(np.float32)
        tiled = np.tile(shifted, 2)
        if len(tiled) < self._frame_samples:
            tiled = np.pad(tiled, (0, self._frame_samples - len(tiled)))
        return tiled[:self._frame_samples]

    def embed(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio) // self._frame_samples
        if n == 0:
            return np.empty((0, self.n_embeddings), dtype=np.float16)
        usable = audio[:n * self._frame_samples]

        plain = super().embed(usable)  # (n, 12288) float16, layer12 features
        if len(plain) != n:
            raise ValueError(
                f'{self.embeddername}: trunk returned {len(plain)} frames for {n} input frames'
            )

        frames = usable.reshape(n, self._frame_samples)
        shifted_frames = np.stack([self._pitch_up_octave(f) for f in frames])
        shifted = super().embed(shifted_frames.reshape(-1))

        return np.concatenate([plain, shifted], axis=1).astype(np.float16)
