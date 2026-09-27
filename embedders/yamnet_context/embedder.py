import importlib

from embedders.recipe import Branch, Context, Keras, Recipe, RecipeEmbedder

# Imported through the module rather than `from ... import EmbedderYamnet`, so that
# load_embedder()'s scan for a BaseEmbedder subclass in this module's namespace can
# only find EmbedderYamnetContext. Two candidates and it picks whichever sorts first.
_yamnet = importlib.import_module('embedders.yamnet.embedder')

# Frames of real audio either side of the target frame folded into its embedding.
CONTEXT_FRAMES = 1


class EmbedderYamnetContext(RecipeEmbedder, _yamnet.EmbedderYamnet):
    """YAMNet, with each frame's embedding widened by its temporal neighbours.

    embed() takes a buffer of *contiguous* audio, embeds every frame in it with
    YAMNet, and returns concat(frame t-1, frame t, frame t+1) per frame — so the
    context a frame carries is the audio that actually sat either side of it.
    Frames at the two ends of a buffer have no neighbour on one side and repeat
    themselves (edge clamping); that is the same thing an inference buffer's edge
    does, and it costs 2 frames per buffer.

    Because the context lives in the embedder, the shipped model.py works
    unchanged: predict() calls embed() and gets n_embeddings-wide vectors.

    Extraction must hand embed() contiguous audio for any of this to be true.
    02_set/extract.py checks `context_frames` and does exactly that; embedding a
    buffer of frames that were first grouped by label would make each frame's
    "neighbours" other frames with its own label, which is the artifact this
    embedder exists to avoid.
    """

    embeddername = "yamnet_context"
    context_frames = CONTEXT_FRAMES
    recipe = Recipe(branches=(Branch(Keras()),),
                    context=Context(k=CONTEXT_FRAMES), chunk=True)
    n_embeddings = _yamnet.EmbedderYamnet.n_embeddings * (2 * CONTEXT_FRAMES + 1)
