"""YAMNet split at layer 3, so layers 4-14 can be fine-tuned (ps-depth4).

Same recipe as `yamnet_trunk_depth12`, cut earlier. Layers 7-12 are
all stride-1, 512-filter blocks, so layer7's output is the same (6, 4, 512)
shape and the cache is the same width; only the cut point and the size of the
trainable tail change. Framing and the audio cache are inherited.
"""
import os

import numpy as np

import importlib
_trunk = importlib.import_module('embedders.yamnet_trunk.embedder')

_TRUNK_SHAPE = (24, 16, 128)   # layer3's spatial map
_TRUNK_LAYER = 'layer3_pointwise_conv_relu'
_GAP_LAYER = 'global_average_pooling2d'
_TAIL_NAME = 'yamnet_tail_l4_14'


class EmbedderYamnetTrunkDepth4(_trunk.EmbedderYamnetTrunk):
    embeddername = "yamnet_trunk_depth4"
    n_embeddings = int(np.prod(_TRUNK_SHAPE))  # flattened trunk map

    def initialize(self):
        import keras
        full = self._load_full()
        self.model = keras.Model(
            inputs=full.input,
            outputs=full.get_layer(_TRUNK_LAYER).output,
            name='yamnet_trunk_l3',
        )
        self.embed(np.zeros(int(self.framelength_s * self.samplerate), dtype=np.float32))

    def build_head(self, n_classes, lr_backbone=0.0, lr_head=2e-4, dropout=0.2,
                   name=None, hidden=0):
        """Reshape(flat -> trunk map) -> [YAMNet layers 4-14 + GAP] -> Dropout
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
        if hidden:
            # optional ReLU layer between the pooled trunk code and the logits
            x = keras.layers.Dense(hidden, activation='relu', name='hidden')(x)
        out = keras.layers.Dense(n_classes)(x)
        model = keras.Model(inp, out, name=name)

        backbone_mult = 1.0 if lr_backbone <= 0 else lr_backbone / lr_head
        model.compile(
            loss=keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.2),
            optimizer=self._optimizer(lr_head, backbone_mult, tail.trainable_variables),
            metrics=['accuracy'],
        )
        return model
