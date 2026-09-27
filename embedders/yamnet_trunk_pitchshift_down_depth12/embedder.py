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

embed() and to_onnx() both come from the recipe (embedders/recipe.py).
"""
import importlib

from embedders.recipe import Branch, Keras, Recipe, RecipeEmbedder, DOWN_OCTAVE_CENTRE

_trunk12 = importlib.import_module('embedders.yamnet_trunk_depth12.embedder')


class EmbedderYamnetTrunkPitchshiftDownDepth12(RecipeEmbedder, _trunk12.EmbedderYamnetTrunkDepth12):
    embeddername = "yamnet_trunk_pitchshift_down_depth12"
    # [plain, octave-down (centre)], each through the depth12 trunk
    recipe = Recipe(branches=(Branch(Keras()),
                              Branch(Keras(), DOWN_OCTAVE_CENTRE)),
                    dtype='float16')
    n_ctx = len(recipe.branches)  # one shared trunk tail per view, TimeDistributed in build_head
    n_embeddings = _trunk12.EmbedderYamnetTrunkDepth12.n_embeddings * n_ctx
