"""`yamnet_trunk_pitchshift`, but the fine-tuned tail's two 1024-d codes are
combined by the SAME explicit signed contrast `yamnet_pitchshift_contrast`
uses on the frozen embedding, instead of a plain concatenation.

New proposal (2026-09-22), following two results from this era: `pitchshift-
contrast` (frozen YAMNet, `[e_t, e_t - e_shifted(t)]` beats plain concat,
+0.048, 1_114 the largest single-fold gain logged this era, +0.153) and
`trunk-ft-pitchshift` (fine-tuned trunk tail, CONCAT of the two views, +0.040
over the frozen concat but a mixed hard-fold signature -- 1_95 up, 1_114
DOWN, the opposite of pitchshift-contrast's clean sweep).

Hypothesis: `trunk-ft-pitchshift`'s concat design asks a linear Dense layer
to recover the contrast itself from two raw codes -- the same limitation
`asym-context-yamnet` demonstrated on the time axis, where an explicit
`[e_t, e_t - mean(neighbours)]` beat raw concatenation. Computing the
contrast explicitly, downstream of the SAME fine-tuned tail, should recover
`pitchshift-contrast`'s 1_114 win while keeping fine-tuning's own gain --
unlike `trunk-ft-pitchshift`'s concat, which got 1_95 (aircraft) but lost
1_114 (trill).

Mechanism: `embed()`/`initialize()`/`_pitch_up_octave` are inherited
byte-identical from `yamnet_trunk_pitchshift` -- same 2 x 12288 float16 cache
row ([plain layer12 features, octave-up layer12 features]), so this reuses
that embedder's cache directly (same `embeddername` prefix would collide, so
it is NOT reused as-is -- a fresh cache under this name is still a real
extraction over the same audio/frames, per LOOP.md's no-synthetic-cache
rule). Only `build_head()` differs: after the shared TimeDistributed tail
produces two 1024-d codes, this concatenates `[code_plain, code_plain -
code_shifted]` (2048-d) instead of flattening both codes side by side
(2048-d either way -- same head width, same trainable tail, only the join
changes from concat to contrast).
"""
import importlib

_trunk = importlib.import_module('embedders.yamnet_trunk.embedder')
_trunk_ps = importlib.import_module('embedders.yamnet_trunk_pitchshift.embedder')


class EmbedderYamnetTrunkPitchshiftContrast(_trunk_ps.EmbedderYamnetTrunkPitchshift):
    embeddername = "yamnet_trunk_pitchshift_contrast"
    # embed()/initialize()/_pitch_up_octave inherited unchanged from
    # yamnet_trunk_pitchshift -- n_ctx=2, n_embeddings=24576, same cache shape.

    def build_head(self, n_classes, lr_backbone=0.0, lr_head=2e-4, dropout=0.2,
                   name=None):
        import keras
        import tensorflow as tf

        full = self._load_full()
        tail = keras.Model(
            inputs=full.get_layer(_trunk._TRUNK_LAYER).output,
            outputs=full.get_layer(_trunk._GAP_LAYER).output,
            name=_trunk._TAIL_NAME,
        )

        train_backbone = lr_backbone > 0
        for layer in tail.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False          # keep AudioSet moving stats
            else:
                layer.trainable = train_backbone

        inp = keras.layers.Input(shape=(self.n_embeddings,), dtype=tf.float32, name='input')
        x = keras.layers.Reshape((2,) + _trunk._TRUNK_SHAPE)(inp)
        x = keras.layers.TimeDistributed(tail)(x)          # (batch, 2, 1024)
        plain = x[:, 0, :]
        shifted = x[:, 1, :]
        contrast = keras.layers.Subtract()([plain, shifted])
        joined = keras.layers.Concatenate()([plain, contrast])
        x = keras.layers.Dropout(dropout)(joined)
        out = keras.layers.Dense(n_classes)(x)
        model = keras.Model(inp, out, name=name)

        backbone_mult = 1.0 if lr_backbone <= 0 else lr_backbone / lr_head
        model.compile(
            loss=keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.2),
            optimizer=self._optimizer(lr_head, backbone_mult, tail.trainable_variables),
            metrics=['accuracy'],
        )
        return model
