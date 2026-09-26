"""[plain, phase-vocoder octave-up]: duration-preserving up shift
(librosa.effects.pitch_shift, +12 semitones), so no tiling seam. The method
comparison against `v4-ft-ps`'s resample up. Onsets smear.

Part of the 2026-09-26 pitch-shift method x direction grid (HANDOFF.md). Views
and the shared-tail mechanism live in embedders/trunk_views.py. Comparators:
`v4-ft` (no shift, 0.375), `v4-ft-ps` (resample up, 0.452), `v4-ft-psud`
(resample up + centre down, 0.452), `v4-ft-psd` (resample centre down).

No to_onnx(): CV only. Add an export before deploying this embedder.
"""
import importlib

_views = importlib.import_module('embedders.trunk_views')


class EmbedderYamnetTrunkVocoderUpDepth12(_views.TrunkViewsDepth12):
    embeddername = "yamnet_trunk_vocoder_up_depth12"
    views = ('vocoder_up',)
    n_ctx = 2  # plain + views -- shared trunk tail, TimeDistributed in build_head
    n_embeddings = _views.VIEW_WIDTH * 2
