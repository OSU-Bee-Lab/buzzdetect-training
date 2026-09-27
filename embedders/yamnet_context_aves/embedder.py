import importlib

from embedders.embedding import BaseEmbedder  # noqa: F401  -- load_embedder's scan
from embedders.recipe import Context, Recipe

"""YAMNet-with-context + AVES on one shared 1.0 s frame grid.

The composition of this era's two paying representation levers:

- `yamnet_context` (+0.058 xfold-pooled) widens a frame with its temporal
  neighbours, which a 0.96 s receptive field cannot see.
- `yamnet_aves` (+0.025, two draws) puts a second, differently-encoding
  backbone beside YAMNet.

Per frame: `[yam(t-1), yam(t), yam(t+1), aves(t)]` = 3840-d. The YAMNet block
is widened and the AVES block is not — AVES already integrates the whole 1.0 s
through a wav2vec2 stack, and widening it too would be a second untested lever
in the same run.

Imported through the module rather than `from ... import EmbedderYamnetAves`,
so load_embedder()'s scan for a BaseEmbedder subclass in this module's
namespace can only find EmbedderYamnetContextAves.
"""

_yav = importlib.import_module('embedders.yamnet_aves.embedder')

CONTEXT_FRAMES = 1
_YAMNET_DIMS = 1024


class EmbedderYamnetContextAves(_yav.EmbedderYamnetAves):
    """YAMNet+AVES, with the YAMNet block widened by its temporal neighbours.

    embed() takes a buffer of *contiguous* audio and returns one row per frame,
    so `context_frames` obliges 02_set/extract.py to hand it whole chunks in
    time order. Embedding the label buckets instead would give every frame
    neighbours that share its own label — the `context-stack` leak.

    Frames at the two ends of a buffer have no neighbour on one side and repeat
    themselves (edge clamping). **extract.py pads every buffer by
    `context_frames` frames of real audio and discards the pad rows**, so a
    clamped row only survives where the padding itself was unavailable — i.e.
    at a true edge of the source recording, which is the one place a deployment
    also has no neighbour. This is the fix for the clamp artifact described in
    notes.md: `yamnet_context`'s cache clamps at every annotation-cluster
    boundary, at a rate that correlates with the label.
    """

    embeddername = "yamnet_context_aves"
    context_frames = CONTEXT_FRAMES
    n_embeddings = _YAMNET_DIMS * (2 * CONTEXT_FRAMES + 1) + (
        _yav.EmbedderYamnetAves.n_embeddings - _YAMNET_DIMS)
    # yamnet_aves's branches; only the YAMNet block is widened with its neighbours
    recipe = Recipe(branches=_yav.EmbedderYamnetAves.recipe.branches,
                    context=Context(k=CONTEXT_FRAMES, branches=1), chunk=True)
