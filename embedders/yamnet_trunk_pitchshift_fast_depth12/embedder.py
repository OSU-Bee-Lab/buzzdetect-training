"""yamnet_trunk_pitchshift_depth12 (`v4-ft-ps`), rebuilt for inference speed.

Luke, 2026-09-28: `v4-ft-ps-e60-moderate` is the best model yet but costs
~2.3x a single-view trunk at inference (two STFTs, a 383-tap resample filter,
two full CNN passes). This embedder keeps the idea -- [plain, octave-up] views
of each frame, one shared fine-tuned depth12 tail -- and builds the up view
cheaply:

  1. The up view comes off the plain view's STFT. Playing audio an octave up
     doubles every frequency and halves the duration, so its spectrogram is
     every second STFT frame of the plain one, read through a mel matrix built
     at twice the sample rate (bin k taken as frequency 2k, i.e. the bin that
     lands on mel frequency f is the original bin at f/2). No resampler and no
     second STFT. Not the resampled audio's spectrogram exactly: its window
     spans 25 ms of the original audio where the resampled audio's spanned 50,
     so frequency resolution is coarser, and its magnitude scale differs.
  2. The up view is not tiled. `v4-ft-ps` tiled 0.48 s of shifted audio twice
     to fill YAMNet's 96-frame patch; the CNN is fully convolutional, so the
     48-frame patch goes through as it is, giving a (3, 4, 512) layer-11 map
     instead of (6, 4, 512). Half the up view's CNN cost.

Untrained prototypes (onnxruntime CPU, 120 s audio): single view 204 ms, the
v4-ft-ps design 463 ms, this one ~270 ms.

The plain view is exactly yamnet_trunk_depth12's (same padding, STFT, mel,
patching and layers 1-11). Embedding per frame: [plain (6,4,512), up (3,4,512)]
flattened, 18432 float16. build_head runs the shared layer 12-14 tail on each
map and concatenates the two pooled codes, as v4-ft-ps's TimeDistributed tail
does.
"""
import importlib

import numpy as np

from embedders.recipe import Branch, Keras, Recipe, RecipeEmbedder

_trunk12 = importlib.import_module('embedders.yamnet_trunk_depth12.embedder')

_PLAIN_SHAPE = (6, 4, 512)
_UP_SHAPE = (3, 4, 512)
_FIRST_LAYER = 'layer1_conv'


def _dual_features_layer(params):
    """Waveform -> (plain patches (n, 96, 64, 1), up patches (n, 48, 64, 1))
    from one STFT. The plain half repeats yamnet/features.py op for op."""
    import keras
    import tensorflow as tf

    from embedders.yamnet import features as features_lib

    window = int(round(params.sample_rate * params.stft_window_seconds))
    hop = int(round(params.sample_rate * params.stft_hop_seconds))
    fft = 2 ** int(np.ceil(np.log(window) / np.log(2.0)))
    bins = fft // 2 + 1
    patch = int(round(params.patch_window_seconds / params.stft_hop_seconds))
    patch_hop = int(round(params.patch_hop_seconds / params.stft_hop_seconds))
    if patch_hop != patch or patch % 2:
        raise ValueError('the up view assumes whole, non-overlapping, even patches')
    sr = params.sample_rate
    mel_plain = tf.signal.linear_to_mel_weight_matrix(
        params.mel_bands, bins, sr, params.mel_min_hz, params.mel_max_hz).numpy()
    mel_up = tf.signal.linear_to_mel_weight_matrix(
        params.mel_bands, bins, 2 * sr, params.mel_min_hz, params.mel_max_hz).numpy()

    class DualFeatures(keras.layers.Layer):
        def call(self, waveform):
            padded = features_lib.pad_waveform(waveform, params)
            mag = tf.abs(tf.signal.stft(padded, window, hop, fft))
            plain = tf.math.log(tf.matmul(mag, tf.constant(mel_plain)) + params.log_offset)
            plain = tf.signal.frame(plain, patch, patch_hop, axis=0)
            up = tf.math.log(tf.matmul(mag[::2], tf.constant(mel_up)) + params.log_offset)
            up = tf.signal.frame(up, patch // 2, patch_hop // 2, axis=0)[:tf.shape(plain)[0]]
            return plain[..., tf.newaxis], up[..., tf.newaxis]

    return DualFeatures(name='dual_features')


class EmbedderYamnetTrunkPitchshiftFastDepth12(RecipeEmbedder, _trunk12.EmbedderYamnetTrunkDepth12):
    embeddername = "yamnet_trunk_pitchshift_fast_depth12"
    # one model computes both views: the up view is read off the plain STFT
    recipe = Recipe(branches=(Branch(Keras()),), dtype='float16')
    n_embeddings = int(np.prod(_PLAIN_SHAPE) + np.prod(_UP_SHAPE))  # 18432

    def initialize(self):
        import keras
        import tensorflow as tf

        full = self._load_full()
        names = [l.name for l in full.layers]
        cnn = full.layers[names.index(_FIRST_LAYER):names.index(_trunk12._TRUNK_LAYER) + 1]
        wave = keras.Input(shape=(), dtype=tf.float32, name='waveform')
        plain, up = _dual_features_layer(full.layers[1].params)(wave)
        for layer in cnn:
            plain, up = layer(plain), layer(up)
        out = keras.layers.Concatenate()([keras.layers.Flatten()(plain),
                                          keras.layers.Flatten()(up)])
        self.model = keras.Model(wave, out, name='yamnet_trunk_l11_psfast')
        self.embed(np.zeros(int(self.framelength_s * self.samplerate), dtype=np.float32))

    def build_head(self, n_classes, lr_backbone=0.0, lr_head=2e-4, dropout=0.2,
                   name=None, hidden=0):
        """[plain (6,4,512) | up (3,4,512)] -> one shared [layers 12-14 + GAP]
        on each -> concat -> Dropout -> Dense(n_classes) logits."""
        import keras
        import tensorflow as tf

        full = self._load_full()
        names = [l.name for l in full.layers]
        tail_layers = full.layers[names.index(_trunk12._TRUNK_LAYER) + 1:
                                  names.index(_trunk12._GAP_LAYER) + 1]
        t_in = keras.Input(shape=(None,) + _PLAIN_SHAPE[1:], name='tail_in')
        t = t_in
        for layer in tail_layers:
            t = layer(t)
        tail = keras.Model(t_in, t, name=_trunk12._TAIL_NAME)

        train_backbone = lr_backbone > 0
        for layer in tail.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False          # keep AudioSet moving stats
            else:
                layer.trainable = train_backbone

        n_plain = int(np.prod(_PLAIN_SHAPE))
        inp = keras.layers.Input(shape=(self.n_embeddings,), dtype=tf.float32, name='input')
        plain = keras.layers.Reshape(_PLAIN_SHAPE)(inp[:, :n_plain])
        up = keras.layers.Reshape(_UP_SHAPE)(inp[:, n_plain:])
        x = keras.layers.Concatenate()([tail(plain), tail(up)])
        x = keras.layers.Dropout(dropout)(x)
        if hidden:
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
