"""YAMNet split at layer 12, so layers 13-14 can be fine-tuned.

The frozen probe pipeline caches YAMNet's final 1024-d GlobalAveragePooling
output; you cannot fine-tune anything from that, the cache *is* the backbone's
output. This embedder moves the cut one block earlier: it caches
`layer12_pointwise_conv_relu`, shape (6, 4, 512) -> flattened 12288, stored
float16. Everything through layer 12 stays frozen and is computed once at
extraction.

`build_head()` then lifts YAMNet's layers 13-14 (+ the GlobalAveragePooling)
straight out of the loaded `yamnet.keras` -- the real layer objects, so kernels
and BatchNorm stats are the trained ones -- and stacks the usual
Dropout(0.2) -> Dense head on top. With the backbone frozen (lr_backbone == 0)
this is numerically the stock linear probe, differing only by the float16
cache. With it trainable, layers 13-14 move; their BatchNorm layers stay frozen
either way (their moving stats are AudioSet's, and letting a small set rewrite
them cost -5.8pp pre-rework).

audio_cache_key() is inherited unchanged, so this reuses the exact framed-audio
cache the `yamnet` embedder already built -- no re-framing, no re-reading source
audio.
"""
import os

import numpy as np

from embedders.embedding import BaseEmbedder

# layer12's spatial map, before layers 13-14's stride-2 downsample.
_TRUNK_SHAPE = (6, 4, 512)
_TRUNK_LAYER = 'layer12_pointwise_conv_relu'
_GAP_LAYER = 'global_average_pooling2d'
_TAIL_NAME = 'yamnet_tail_l13_14'
# backbone-tail trainable variables are named 'layer13_*' / 'layer14_*';
# the head weight is 'dense/*'. The optimizer scales anything matching these.
_TAG = ('layer13', 'layer14')


_ADAM_MULTILR_CLS = None


def _var_id(variable):
    # update_step receives the raw tf.Variable (keras backend.Variable.value);
    # match on identity, since tf.Variable has no `.path` and its `.name` is not
    # reliably layer-scoped. Unwrap a keras Variable exactly as keras'
    # own _var_key does, but never touch tf.Variable.value (that's a method).
    from keras.src import backend
    if isinstance(variable, backend.Variable):
        variable = variable.value
    return id(variable)


def _make_adam_multilr(lr_head, backbone_mult, backbone_ids):
    """Adam with a per-variable LR multiplier for the backbone-tail variables.

    Keras 3 has no parameter groups. Scaling the *gradient* instead is a no-op
    here -- Adam divides by its own RMS, so a constant factor on the gradient
    cancels. This scales the learning rate inside update_step, which does not.
    """
    global _ADAM_MULTILR_CLS
    import keras
    if _ADAM_MULTILR_CLS is None:
        class AdamMultiLR(keras.optimizers.Adam):
            def __init__(self, *a, backbone_mult=1.0, backbone_ids=(), **kw):
                super().__init__(*a, **kw)
                self._backbone_mult = float(backbone_mult)
                self._backbone_ids = set(backbone_ids)

            def update_step(self, gradient, variable, learning_rate):
                if _var_id(variable) in self._backbone_ids:
                    learning_rate = learning_rate * self._backbone_mult
                return super().update_step(gradient, variable, learning_rate)

            def get_config(self):
                cfg = super().get_config()
                cfg.update(backbone_mult=self._backbone_mult)
                return cfg
        _ADAM_MULTILR_CLS = AdamMultiLR
    return _ADAM_MULTILR_CLS(learning_rate=lr_head, clipnorm=1.0,
                             backbone_mult=backbone_mult, backbone_ids=backbone_ids)


class EmbedderYamnetTrunk(BaseEmbedder):
    embeddername = "yamnet_trunk"
    framelength_s = 0.96
    digits_time = 2
    samplerate = 16000
    n_embeddings = int(np.prod(_TRUNK_SHAPE))  # 12288
    dtype_in = 'float32'

    def _yamnet_dir(self):
        # embedders/yamnet is a sibling (a symlink in a worktree); its
        # yamnet.keras and yamnet.py are shared, not copied per embedder.
        return os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'yamnet')

    def _load_full(self):
        import tensorflow as tf  # noqa: F401 -- deferred, see yamnet/embedder.py
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        from embedders.yamnet.yamnet import WaveformFeatures  # registers custom layer
        _ = WaveformFeatures.dtype
        import keras
        model = keras.models.load_model(os.path.join(self._yamnet_dir(), 'yamnet.keras'), compile=False)
        # non-overlapping patches, matching how the frozen probe extracts
        model.layers[1].params.patch_hop_seconds = self.framehop_s
        return model

    def initialize(self):
        import keras
        full = self._load_full()
        self.model = keras.Model(
            inputs=full.input,
            outputs=full.get_layer(_TRUNK_LAYER).output,
            name='yamnet_trunk_l12',
        )
        # warm the TF thread pool before librosa/Accelerate claims threads
        self.embed(np.zeros(int(self.framelength_s * self.samplerate), dtype=np.float32))

    def embed(self, audio):
        out = np.asarray(self.model(audio))          # (n_frames, 6, 4, 512)
        return out.reshape(out.shape[0], -1).astype(np.float16)

    # --- fine-tunable head -------------------------------------------------

    @staticmethod
    def _optimizer(lr_head, backbone_mult, backbone_vars):
        import keras
        if backbone_mult == 1.0:
            return keras.optimizers.Adam(learning_rate=lr_head, clipnorm=1.0)
        return _make_adam_multilr(lr_head, backbone_mult,
                                  [_var_id(v) for v in backbone_vars])

    def build_head(self, n_classes, lr_backbone=0.0, lr_head=2e-4, dropout=0.2,
                   name=None):
        """Reshape(12288 -> 6,4,512) -> [YAMNet layers 13-14 + GAP] -> Dropout
        -> Dense(n_classes) logits.

        lr_backbone == 0 freezes layers 13-14 (stock linear probe, through the
        trunk pipeline). lr_backbone > 0 makes their conv kernels trainable at
        that rate while the head trains at lr_head; BatchNorm in that block
        stays frozen regardless. lr_backbone == lr_head gives a plain uniform
        Adam; anything smaller uses AdamMultiLR (per-variable LR scale).
        """
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
                layer.trainable = False          # keep AudioSet moving stats
            else:
                layer.trainable = train_backbone

        inp = keras.layers.Input(shape=(self.n_embeddings,), dtype=tf.float32, name='input')
        x = keras.layers.Reshape(_TRUNK_SHAPE)(inp)
        x = tail(x)
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
