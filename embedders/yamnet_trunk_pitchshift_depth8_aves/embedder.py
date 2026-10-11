"""ps-depth8's two trunk views plus a frozen AVES block read by the same linear head.

`yamnet_trunk_pitchshift_depth8` (plain + octave-up through YAMNet layers 1-7, layers 8-14
fine-tuned in the head) with AVES layers 6, 9 and 12 mean-pooled over the same 0.96 s frame
appended (`yamnet_aves_mid`'s block, which there needed a 1.0 s frame; here AVES sees the
trunk's own frame, 47 tokens). The trainable tail pools each view to 1024; the class Dense
reads [2 x 1024 tail code, 3 x 768 AVES]. AVES is frozen: its columns are cached features.

One-off test allowed by Luke on 2026-10-11 under LOOP.md's AVES injunction (exp/ps-d8-aves).
"""
import importlib
import os

from embedders.recipe import Aves, Branch, Keras, Recipe, RecipeEmbedder, UP_OCTAVE

_d8 = importlib.import_module('embedders.yamnet_trunk_depth8.embedder')

_LAYER_INDICES = (5, 8, 11)   # 0-indexed: AVES layers 6, 9, 12 (as yamnet_aves_mid)
_TRUNK_WIDTH = _d8.EmbedderYamnetTrunkDepth8.n_embeddings
_N_VIEWS = 2
_AVES_WIDTH = 768 * len(_LAYER_INDICES)


class EmbedderYamnetTrunkPitchshiftDepth8Aves(RecipeEmbedder, _d8.EmbedderYamnetTrunkDepth8):
    embeddername = "yamnet_trunk_pitchshift_depth8_aves"
    # [plain, octave-up] through the depth8 trunk, then AVES on the plain frame
    recipe = Recipe(branches=(Branch(Keras()),
                              Branch(Keras(), UP_OCTAVE),
                              Branch(Aves(layers=_LAYER_INDICES))),
                    dtype='float16')
    n_embeddings = _TRUNK_WIDTH * _N_VIEWS + _AVES_WIDTH

    def initialize(self):
        import json

        import tensorflow as tf
        import torch
        import torchaudio

        if os.environ.get('BUZZDETECT_NO_GPU'):      # the trunk on CPU, the card left to AVES
            tf.config.set_visible_devices([], 'GPU')
        aves_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'aves')
        with open(os.path.join(aves_dir, 'model_config.json')) as f:
            config = json.load(f)
        self.aves = torchaudio.models.wav2vec2_model(**config, aux_num_out=None)
        self.aves.load_state_dict(torch.load(os.path.join(aves_dir, 'aves-base-bio.pt'),
                                             map_location='cpu', weights_only=True))
        self.aves.eval()
        device = os.environ.get('BUZZDETECT_AVES_DEVICE', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.batch_size = int(os.environ.get('BUZZDETECT_AVES_BATCH', '64'))
        if device == 'cuda':
            torch.backends.cudnn.enabled = False     # as embedders/aves: cuDNN version mismatch in this env
        self.aves.to(device)
        super().initialize()

    def build_head(self, n_classes, lr_backbone=0.0, lr_head=2e-4, dropout=0.2, name=None, hidden=0):
        """ps-depth8's head with the AVES columns concatenated to the pooled tail code before the Dense."""
        import keras
        import tensorflow as tf

        full = self._load_full()
        tail = keras.Model(inputs=full.get_layer(_d8._TRUNK_LAYER).output,
                           outputs=full.get_layer(_d8._GAP_LAYER).output, name=_d8._TAIL_NAME)
        for layer in tail.layers:
            layer.trainable = lr_backbone > 0 and not isinstance(layer, keras.layers.BatchNormalization)

        split = _TRUNK_WIDTH * _N_VIEWS
        inp = keras.layers.Input(shape=(self.n_embeddings,), dtype=tf.float32, name='input')
        x = keras.layers.Reshape((_N_VIEWS,) + _d8._TRUNK_SHAPE)(inp[:, :split])
        x = keras.layers.Flatten()(keras.layers.TimeDistributed(tail)(x))
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.Concatenate(name='code_aves')([x, inp[:, split:]])
        if hidden:
            x = keras.layers.Dense(hidden, activation='relu', name='hidden')(x)
        model = keras.Model(inp, keras.layers.Dense(n_classes)(x), name=name)
        backbone_mult = 1.0 if lr_backbone <= 0 else lr_backbone / lr_head
        model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.2),
                      optimizer=self._optimizer(lr_head, backbone_mult, tail.trainable_variables),
                      metrics=['accuracy'])
        return model
