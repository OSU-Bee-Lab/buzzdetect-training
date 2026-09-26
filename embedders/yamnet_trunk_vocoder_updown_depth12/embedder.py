"""[plain, phase-vocoder octave-up, phase-vocoder octave-down]: the vocoder
version of `v4-ft-psud`.

Part of the 2026-09-26 pitch-shift method x direction grid (HANDOFF.md). Views
and the shared-tail mechanism live in embedders/trunk_views.py. Comparators:
`v4-ft` (no shift, 0.375), `v4-ft-ps` (resample up, 0.452), `v4-ft-psud`
(resample up + centre down, 0.452), `v4-ft-psd` (resample centre down).

No to_onnx(): CV only. Add an export before deploying this embedder.
"""
import importlib

_views = importlib.import_module('embedders.trunk_views')


class EmbedderYamnetTrunkVocoderUpdownDepth12(_views.TrunkViewsDepth12):
    embeddername = "yamnet_trunk_vocoder_updown_depth12"
    views = ('vocoder_up', 'vocoder_down')
    n_ctx = 3  # plain + views -- shared trunk tail, TimeDistributed in build_head
    n_embeddings = _views.VIEW_WIDTH * 3
