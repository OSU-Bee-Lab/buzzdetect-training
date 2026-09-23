import importlib

# Imported through the module so load_embedder()'s scan for a BaseEmbedder
# subclass in this namespace only finds this class. See embedders/yamnet_pooled.py.
_pooled = importlib.import_module('embedders.yamnet_pooled')


class EmbedderYamnetTripleOverlay(_pooled.EmbedderYamnetPooled):
    embeddername = "yamnet_triple_overlay"
    mode = "overlay"
