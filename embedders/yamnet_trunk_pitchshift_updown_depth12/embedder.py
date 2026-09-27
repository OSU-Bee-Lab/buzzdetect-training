"""yamnet_trunk_pitchshift_depth12 plus a third, octave-DOWN view: each frame is
embedded as [plain, octave-up, octave-down], all three through ONE shared
fine-tuned depth12 tail (TimeDistributed, n_ctx=3), codes concatenated into the
head.

Proposal (Luke, 2026-09-26): `v4-ft-ps` (octave-up twin) won the v4 grid at
0.452 vs `v4-ft` 0.375. Octave-up moves buzz fundamentals (honey bee ~230 Hz)
off YAMNet's 125 Hz mel floor into denser filterbank (see yamnet_pitchshift).
Octave-down goes the other way, and so works against that mechanism: 230 Hz
lands at 115 Hz, below the mel floor, so this view sees harmonics only. The
audio is 16 kHz and YAMNet's mel top is 7.5 kHz, so shifting down adds only the
7.5-8 kHz sliver as new content. What it does add is time-stretch (each buzz
twice as long) and a second, coarser look at the harmonic stack. Whether that
helps is an empirical question. Comparator: `v4-ft-ps`, same config otherwise.

Octave-down mechanism, mirroring the up shift: take the centre half of the
0.96 s frame (0.24-0.72 s), upsample it 2:1 with librosa and relabel the
result as 16 kHz. That gives a full 0.96 s patch at half pitch and half speed,
centred on the same instant as the plain view, so the frame grid is unchanged.
The outer quarters of the frame are not in this view. The up view still covers
them, tiled.

embed() and to_onnx() both come from the recipe (embedders/recipe.py).
"""
import importlib

from embedders.recipe import Branch, Keras, Recipe, RecipeEmbedder, DOWN_OCTAVE_CENTRE, UP_OCTAVE

_trunk12 = importlib.import_module('embedders.yamnet_trunk_depth12.embedder')


class EmbedderYamnetTrunkPitchshiftUpdownDepth12(RecipeEmbedder, _trunk12.EmbedderYamnetTrunkDepth12):
    embeddername = "yamnet_trunk_pitchshift_updown_depth12"
    # [plain, octave-up, octave-down (centre)], each through the depth12 trunk
    recipe = Recipe(branches=(Branch(Keras()),
                              Branch(Keras(), UP_OCTAVE),
                              Branch(Keras(), DOWN_OCTAVE_CENTRE)),
                    dtype='float16')
    n_ctx = len(recipe.branches)  # one shared trunk tail per view, TimeDistributed in build_head
    n_embeddings = _trunk12.EmbedderYamnetTrunkDepth12.n_embeddings * n_ctx
