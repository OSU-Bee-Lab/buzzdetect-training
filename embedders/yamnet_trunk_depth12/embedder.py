"""YAMNet split at layer 11, one block earlier than `yamnet_trunk`, so
layers 12-14 can be fine-tuned instead of just 13-14.

IDEAS proposal, new (2026-09-21): `trunk-ft-v3` found unfreezing layers 13-14
a robust +0.077 this era. Its own hypothesis section cited an old E2 result
(pre-cutover, early stopping, different metric) where unfreezing one layer
deeper (12-14) was *worse* than 13-14 (0.216 -> 13-14 0.262 -> 12-14 0.229).
LOOP.md's standing principle is to rerun rather than defer old-era verdicts
(`temporal-context` inverted between E1 and E2). This reruns the depth
question under the current era's fixed-epoch, no-early-stopping regime, where
`trunk-ft-v3` already found the previous depth-ladder E2 numbers plausible but
untested here.

layer11 and layer12 are both stride-1, 512-filter separable-conv blocks (the
four stride-2 blocks are layers 1/3/5/7; layer13 is the fifth and last), so
layer11's output has the exact same shape as layer12's: (6, 4, 512) -- cutting
one block earlier costs nothing in cache size or frame grid, only moves one
more conv block (plus its BatchNorm, held frozen as `yamnet_trunk` already
does for 13-14) into the trainable tail.

Everything else -- framing, audio cache key, the `n_ctx` TimeDistributed
mechanism for context variants, the per-variable backbone LR -- is inherited
from `yamnet_trunk` unchanged; only the cut point and tail lift shift by one
layer.
"""
import os

import numpy as np

import importlib
_trunk = importlib.import_module('embedders.yamnet_trunk.embedder')

_TRUNK_SHAPE = (6, 4, 512)   # layer11's spatial map, same as layer12's
_TRUNK_LAYER = 'layer11_pointwise_conv_relu'
_GAP_LAYER = 'global_average_pooling2d'
_TAIL_NAME = 'yamnet_tail_l12_13_14'


class EmbedderYamnetTrunkDepth12(_trunk.EmbedderYamnetTrunk):
    embeddername = "yamnet_trunk_depth12"
    n_embeddings = int(np.prod(_TRUNK_SHAPE))  # 12288, same width as yamnet_trunk

    def initialize(self):
        import keras
        full = self._load_full()
        self.model = keras.Model(
            inputs=full.input,
            outputs=full.get_layer(_TRUNK_LAYER).output,
            name='yamnet_trunk_l11',
        )
        self.embed(np.zeros(int(self.framelength_s * self.samplerate), dtype=np.float32))

    def build_head(self, n_classes, lr_backbone=0.0, lr_head=2e-4, dropout=0.2,
                   name=None):
        """Reshape(12288 -> 6,4,512) -> [YAMNet layers 12-14 + GAP] -> Dropout
        -> Dense(n_classes) logits. One block deeper than yamnet_trunk's tail."""
        import keras
        import tensorflow as tf

        full = self._load_full()
        tail = keras.Model(
            inputs=full.get_layer(_TRUNK_LAYER).output,
            outputs=full.get_layer(_GAP_LAYER).output,
            name=_TAIL_NAME,
        )

        train_backbone = lr_backbone > 0
        for layer in tail.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False          # keep AudioSet moving stats, all three blocks
            else:
                layer.trainable = train_backbone

        inp = keras.layers.Input(shape=(self.n_embeddings,), dtype=tf.float32, name='input')
        n_ctx = getattr(self, 'n_ctx', 1)
        if n_ctx == 1:
            x = keras.layers.Reshape(_TRUNK_SHAPE)(inp)
            x = tail(x)
        else:
            x = keras.layers.Reshape((n_ctx,) + _TRUNK_SHAPE)(inp)
            x = keras.layers.TimeDistributed(tail)(x)
            x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(dropout)(x)
        out = keras.layers.Dense(n_classes)(x)
        model = keras.Model(inp, out, name=name)

        backbone_mult = 1.0 if lr_backbone <= 0 else lr_backbone / lr_head
        model.compile(
            loss=keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.2),
            optimizer=self._optimizer(lr_head, backbone_mult, tail.trainable_variables),
            metrics=['accuracy'],
        )
        return model
