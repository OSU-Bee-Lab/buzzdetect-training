"""yamnet_trunk_depth12 (layer-11 cut, fine-tunable layers 12-14), plus the
same octave-up pitch-shifted twin `yamnet_pitchshift`/`yamnet_trunk_pitchshift`
concatenate -- run through the SAME fine-tuned trunk tail via the
TimeDistributed mechanism, exactly as `yamnet_trunk_pitchshift` does for the
layer-12 cut.

IDEAS-style proposal (2026-09-23, run inline, not queued): `trunk-ft-pitchshift`
combined this era's two largest single levers (pitch-shift concat, trunk
fine-tuning) at the layer-12 cut and got a mixed hard-fold signature (1_95 up,
1_114 down) relative to pitchshift-contrast's clean 1_114 sweep in the frozen
regime. Separately, `trunk-depth-headtohead` found the layer-11 cut
(depth12) statistically equivalent to the layer-12 cut (13-14) on plain
fine-tuning alone (-0.020 +/- 0.011, inside the era's noise floor). Whether
that equivalence holds once pitch-shift concat is stacked on top is untested
-- the extra trainable block could interact differently with two co-tuned
views than with one. Comparator: `trunk-ft-pitchshift`'s
`yamnet_trunk_pitchshift` model (layer-12 cut + concat), same day/config
otherwise.

Mechanism is a straight recombination of two already-merged embedders: the
depth-12 cut/tail from `yamnet_trunk_depth12`, the octave-up dual-view +
shared-tail TimeDistributed concat from `yamnet_trunk_pitchshift`. No new
audio transform, no new architecture piece.
"""
import importlib

import numpy as np

_trunk12 = importlib.import_module('embedders.yamnet_trunk_depth12.embedder')
_ps = importlib.import_module('embedders.yamnet_trunk_pitchshift.embedder')


class EmbedderYamnetTrunkPitchshiftDepth12(_trunk12.EmbedderYamnetTrunkDepth12):
    embeddername = "yamnet_trunk_pitchshift_depth12"
    n_ctx = 2  # [plain, octave-up] -- shared trunk tail, TimeDistributed in build_head
    n_embeddings = _trunk12.EmbedderYamnetTrunkDepth12.n_embeddings * 2  # 24576

    def initialize(self):
        import librosa
        self._librosa = librosa
        self._frame_samples = int(round(self.framelength_s * self.samplerate))
        self._half_sr = self.samplerate // 2
        super().initialize()  # loads self.model = trunk through layer11, warms it

    # identical mechanism to yamnet_trunk_pitchshift._pitch_up_octave -- held
    # as its own copy since this class's MRO runs through
    # EmbedderYamnetTrunkDepth12, not EmbedderYamnetTrunkPitchshift.
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

        plain = super().embed(usable)  # (n, 12288) float16, layer11 features
        if len(plain) != n:
            raise ValueError(
                f'{self.embeddername}: trunk returned {len(plain)} frames for {n} input frames'
            )

        frames = usable.reshape(n, self._frame_samples)
        shifted_frames = np.stack([self._pitch_up_octave(f) for f in frames])
        shifted = super().embed(shifted_frames.reshape(-1))

        return np.concatenate([plain, shifted], axis=1).astype(np.float16)
