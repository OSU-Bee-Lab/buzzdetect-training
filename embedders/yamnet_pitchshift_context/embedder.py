import importlib

from embedders.embedding import BaseEmbedder  # noqa: F401  -- load_embedder's scan
from embedders.recipe import Branch, Context, Keras, Recipe, UP_OCTAVE

_pitchshift = importlib.import_module('embedders.yamnet_pitchshift.embedder')

"""IDEAS.md item 16: stack context onto the pitch-shift block.

This era's two best-confirmed levers, composed the way `yamnet_context_aves`
composed context + AVES: widen only ONE named half with temporal context and
leave the other as-is, so the run tests one new combination, not two new
levers at once.

`yamnet_pitchshift` (+0.069, 8/8 folds, twice confirmed) is
`[unshifted(t), shifted_up_octave(t)]`, 2048-d, both halves reading only frame
t's own audio. `yamnet_context` (+0.030, honest) widens plain YAMNet with its
real temporal neighbours. Widening the UNSHIFTED half here (not the shifted
one) mirrors `yamnet_context_aves`'s choice to widen the plain-YAMNet block
and leave AVES's already-temporally-integrated block alone: the pitch-shift
transform is a per-frame audio operation, not obviously suited to being
computed on someone else's neighbour, whereas the unshifted half is exactly
`yamnet_context`'s own input.

Per frame: `[yam(t-1), yam(t), yam(t+1), shifted(t)]` = 4096-d.

Context comes from `context_frames`, the same mechanism `yamnet_context` and
`yamnet_pitchshift_decimate` use: 02_set/extract.py hands embed() a buffer of
real contiguous audio, padded by one true frame on each side at chunk
boundaries, and discards the pad rows after.
"""

CONTEXT_FRAMES = 1
_YAMNET_DIMS = 1024


class EmbedderYamnetPitchshiftContext(_pitchshift.EmbedderYamnetPitchshift):
    embeddername = "yamnet_pitchshift_context"
    context_frames = CONTEXT_FRAMES
    # [plain, octave-up]; only the plain block is widened with its neighbours
    recipe = Recipe(branches=(Branch(Keras()),
                              Branch(Keras(), UP_OCTAVE)),
                    context=Context(k=CONTEXT_FRAMES, branches=1), chunk=True)
    n_embeddings = _YAMNET_DIMS * (2 * CONTEXT_FRAMES + 1) + _YAMNET_DIMS  # widened unshifted + shifted as-is
