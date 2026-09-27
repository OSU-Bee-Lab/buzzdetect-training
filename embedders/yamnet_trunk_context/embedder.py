import importlib

from embedders.recipe import Branch, Context, Keras, Recipe, RecipeEmbedder

# Through the module, so load_embedder()'s scan finds only this class.
_trunk = importlib.import_module('embedders.yamnet_trunk.embedder')

CONTEXT_FRAMES = 1


class EmbedderYamnetTrunkContext(RecipeEmbedder, _trunk.EmbedderYamnetTrunk):
    """yamnet_trunk (layer-12 maps, fine-tunable layers 13-14) with temporal context.

    Each frame carries concat(trunk t-1, trunk t, trunk t+1) = 3 x 12288 float16.
    build_head() runs ONE shared copy of layers 13-14 over the three frames, so
    fine-tuning sees the context (the stack of yamnet_context, but pre-GAP), then
    concatenates the three 1024-d codes into the Dense. Context comes from
    contiguous audio, as in yamnet_context (02_set/extract.py reads context_frames).
    """

    embeddername = "yamnet_trunk_context"
    context_frames = CONTEXT_FRAMES
    recipe = Recipe(branches=(Branch(Keras()),),
                    context=Context(k=CONTEXT_FRAMES), chunk=True, dtype='float16')
    n_ctx = 2 * CONTEXT_FRAMES + 1
    n_embeddings = _trunk.EmbedderYamnetTrunk.n_embeddings * (2 * CONTEXT_FRAMES + 1)
