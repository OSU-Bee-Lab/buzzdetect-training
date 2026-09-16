import numpy as np

from embedders.embedding import BaseEmbedder

"""IDEAS.md item 2a. [yamnet_pitchshift_aves_mid(t), perch(broadcast)] on the
lead's 1.0 s frame grid, 4352 + 1536 = 5888-d.

This embedder's cache is never produced by extract.py -- both source caches
already exist for `medium`, so `tools/join_broadcast.py` builds it directly
from their pickles, joining on frametimes.csv timestamps rather than
re-running any audio through a model (see that script's docstring for why a
plain j -> j // 5 index map doesn't hold). `embed()` is therefore
unreachable in the normal pipeline and only raises, matching the fact that
there is no raw-audio path that produces this embedder's output directly.
"""


class EmbedderYamnetPitchshiftAvesMidPerch(BaseEmbedder):
    embeddername = "yamnet_pitchshift_aves_mid_perch"
    framelength_s = 1.0
    digits_time = 2
    samplerate = 16000
    n_embeddings = 4352 + 1536
    dtype_in = 'float32'

    def initialize(self):
        pass

    def embed(self, audio):
        raise NotImplementedError(
            'yamnet_pitchshift_aves_mid_perch has no direct audio path; its '
            'cache is built by tools/join_broadcast.py from the '
            'yamnet_pitchshift_aves_mid and perch caches.'
        )
