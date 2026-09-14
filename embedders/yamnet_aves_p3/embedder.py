import os

import numpy as np

from embedders.embedding import BaseEmbedder

"""YAMNet + AVES, with AVES pooled by [mean, max, std] over its token axis
instead of mean alone.

IDEAS.md item 3. `embedders/yamnet_aves/embedder.py` takes wav2vec2's
last-layer output `(B, T', 768)` and does `.mean(dim=1)` -- a mean over ~49
tokens of 20 ms each. Mean pooling is a matched filter for a *stationary*
signal and the worst pooling for a *transient* one; a buzz occupying 200 ms of
a 1.0 s frame has its evidence divided by ~5 before the probe ever sees it.
`max` is the parameter-free form of what attentive pooling recovers for
transformer probes in the literature (see the idea for citations); `std`
adds a cheap measure of how non-stationary the frame's activations are, which
a pure mean or max both discard. Concatenating all three costs nothing beyond
the trivial extra reduction ops -- the token sequence itself is never cached.

Re-aimed 2026-09-11 (see IDEAS.md): this targets `untagged`/`loud` sensitivity
on folds weak on *audible* buzz, not quiet-buzz recall, which is out of the
headline. Otherwise identical in structure to `yamnet_aves`: same YAMNet half
(centre-cropped 0.96 s of AVES's 1.0 s frame, on one shared grid), same model
weights, same device/memory-growth handling. n_embeddings = 1024 + 3*768 =
3328.

This is a standalone copy of `yamnet_aves/embedder.py`, not a subclass of it --
`load_embedder()` scans a module's namespace for exactly one `BaseEmbedder`
subclass, so importing `EmbedderYamnetAves` into this module's namespace would
give it two candidates to choose between.
"""

_YAMNET_SAMPLES = 15360   # 0.96 s at 16 kHz, centred in the 1.0 s frame
_CROP0 = (16000 - _YAMNET_SAMPLES) // 2   # 320


class EmbedderYamnetAvesP3(BaseEmbedder):
    embeddername = "yamnet_aves_p3"
    framelength_s = 1.0
    digits_time = 2
    samplerate = 16000
    n_embeddings = 1024 + 768 * 3   # 1024 YAMNet + [mean, max, std] over AVES tokens
    dtype_in = 'float32'

    def initialize(self):
        import tensorflow as tf
        from embedders.yamnet.yamnet import WaveformFeatures  # registers custom layer
        _ = WaveformFeatures.dtype

        if os.environ.get('BUZZDETECT_NO_GPU'):
            tf.config.set_visible_devices([], 'GPU')
        else:
            for gpu in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)

        here = os.path.dirname(os.path.realpath(__file__))
        yamnet_keras = os.path.join(here, '..', 'yamnet', 'yamnet.keras')
        self.yamnet = tf.keras.models.load_model(yamnet_keras, compile=False)
        self.yamnet.layers[1].params.patch_hop_seconds = 0.96

        import json
        import torch
        import torchaudio

        aves_dir = os.path.join(here, '..', 'aves')
        with open(os.path.join(aves_dir, 'model_config.json')) as f:
            config = json.load(f)
        self.aves = torchaudio.models.wav2vec2_model(**config, aux_num_out=None)
        state = torch.load(os.path.join(aves_dir, 'aves-base-bio.pt'),
                           map_location='cpu', weights_only=True)
        self.aves.load_state_dict(state)
        self.aves.eval()

        device = os.environ.get('BUZZDETECT_AVES_DEVICE', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.batch_size = int(os.environ.get('BUZZDETECT_AVES_BATCH', '64'))
        if device == 'cuda':
            torch.backends.cudnn.enabled = False
        self.aves.to(device)

        self.embed(np.zeros(self.samplerate, dtype=np.float32))

    def embed(self, audio):
        import torch

        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio) // self.samplerate
        if n == 0:
            return np.empty((0, self.n_embeddings), dtype=np.float32)
        frames = audio[:n * self.samplerate].reshape(n, self.samplerate)

        crops = frames[:, _CROP0:_CROP0 + _YAMNET_SAMPLES]
        yam = np.asarray(self.yamnet(crops.reshape(-1)), dtype=np.float32)
        if len(yam) != n:
            raise ValueError(
                f'yamnet_aves_p3: YAMNet returned {len(yam)} frames for {n} crops'
            )

        out = []
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                stop = min(start + self.batch_size, n)
                wav = torch.from_numpy(frames[start:stop]).to(self.device)
                layer_outputs, _ = self.aves.extract_features(wav)
                tokens = layer_outputs[-1]  # (batch, T', 768)
                pooled = torch.cat(
                    [tokens.mean(dim=1), tokens.amax(dim=1), tokens.std(dim=1)],
                    dim=1,
                )
                out.append(pooled.cpu().numpy())
        av = np.concatenate(out, axis=0).astype(np.float32)

        return np.concatenate([yam, av], axis=1)

    def to_onnx(self, opset=17):
        """Same bridge as `yamnet_aves.to_onnx()`; only the AVES-side pooling
        wrapper differs (mean+max+std instead of mean alone). See that
        docstring for the template this follows."""
        import tempfile

        import numpy as np
        import onnx
        import tensorflow as tf
        import tf2onnx
        import torch
        from onnx import TensorProto, compose, helper

        samplerate = self.samplerate

        spec = (tf.TensorSpec((None,), tf.float32, name='crops'),)
        yam_proto, _ = tf2onnx.convert.from_keras(
            self.yamnet, input_signature=spec, opset=opset)

        class _AvesStatsPool(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, waveform):
                layer_outputs, _ = self.model.extract_features(waveform)
                tokens = layer_outputs[-1]
                return torch.cat(
                    [tokens.mean(dim=1), tokens.amax(dim=1), tokens.std(dim=1)],
                    dim=1,
                )

        wrapper = _AvesStatsPool(self.aves.cpu()).eval()
        dummy = torch.zeros(2, samplerate, dtype=torch.float32)
        try:
            with tempfile.TemporaryDirectory() as d:
                path_av = os.path.join(d, 'aves.onnx')
                with torch.no_grad():
                    torch.onnx.export(
                        wrapper, dummy, path_av,
                        input_names=['frames'], output_names=['embedding'],
                        dynamic_axes={'frames': {0: 'batch'}, 'embedding': {0: 'batch'}},
                        opset_version=opset, dynamo=False,
                    )
                av_proto = onnx.load(path_av)
        finally:
            self.aves.to(self.device)

        yam_proto = compose.add_prefix(yam_proto, prefix='yam_')
        av_proto = compose.add_prefix(av_proto, prefix='av_')
        yam_in, yam_out = yam_proto.graph.input[0].name, yam_proto.graph.output[0].name
        av_in, av_out = av_proto.graph.input[0].name, av_proto.graph.output[0].name

        waveform_in = helper.make_tensor_value_info(
            'waveform', TensorProto.FLOAT, ['samples'])
        embedding_out = helper.make_tensor_value_info(
            'embedding', TensorProto.FLOAT, ['n_frames', self.n_embeddings])

        c_sr = helper.make_tensor('c_samplerate', TensorProto.INT64, [1], [samplerate])
        c_zero = helper.make_tensor('c_zero', TensorProto.INT64, [1], [0])
        c_one = helper.make_tensor('c_one', TensorProto.INT64, [1], [1])
        c_axis0 = helper.make_tensor('c_axis0', TensorProto.INT64, [1], [0])
        c_frame_shape = helper.make_tensor(
            'c_frame_shape', TensorProto.INT64, [2], [-1, samplerate])
        nodes = [
            helper.make_node('Shape', ['waveform'], ['total_len'], name='bridge_shape'),
            helper.make_node('Div', ['total_len', 'c_samplerate'], ['n_frames_real'],
                             name='bridge_n_frames_real'),
            helper.make_node('Max', ['n_frames_real', 'c_one'], ['n_frames_safe'],
                             name='bridge_n_frames_safe'),
            helper.make_node('Mul', ['n_frames_safe', 'c_samplerate'], ['usable_len_safe'],
                             name='bridge_usable_len_safe'),
            helper.make_node('Sub', ['usable_len_safe', 'total_len'], ['pad_amount_signed'],
                             name='bridge_pad_amount_signed'),
            helper.make_node('Max', ['pad_amount_signed', 'c_zero'], ['pad_amount'],
                             name='bridge_pad_amount'),
            helper.make_node('Concat', ['c_zero', 'pad_amount'], ['pads'], axis=0,
                             name='bridge_pads'),
            helper.make_node('Pad', ['waveform', 'pads'], ['padded'], name='bridge_pad'),
            helper.make_node('Slice', ['padded', 'c_zero', 'usable_len_safe', 'c_axis0'],
                             ['cropped'], name='bridge_crop_tail'),
            helper.make_node('Reshape', ['cropped', 'c_frame_shape'], ['frames'],
                             name='bridge_frames'),
        ]

        c_crop_starts = helper.make_tensor('c_crop_starts', TensorProto.INT64, [1], [_CROP0])
        c_crop_ends = helper.make_tensor(
            'c_crop_ends', TensorProto.INT64, [1], [_CROP0 + _YAMNET_SAMPLES])
        c_crop_axis = helper.make_tensor('c_crop_axis', TensorProto.INT64, [1], [1])
        c_flat_shape = helper.make_tensor('c_flat_shape', TensorProto.INT64, [1], [-1])
        nodes += [
            helper.make_node('Slice', ['frames', 'c_crop_starts', 'c_crop_ends', 'c_crop_axis'],
                             ['crop'], name='bridge_crop_yamnet'),
            helper.make_node('Reshape', ['crop', 'c_flat_shape'], [yam_in],
                             name='bridge_flatten_for_yamnet'),
            helper.make_node('Identity', ['frames'], [av_in],
                             name='bridge_identity_for_aves'),
        ]

        trailer = [
            helper.make_node('Concat', [yam_out, av_out], ['embedding_safe'],
                             axis=1, name='bridge_concat'),
            helper.make_node('Slice', ['embedding_safe', 'c_zero', 'n_frames_real', 'c_axis0'],
                             ['embedding'], name='bridge_trim_to_real'),
        ]

        graph = helper.make_graph(
            nodes=nodes + list(yam_proto.graph.node) + list(av_proto.graph.node) + trailer,
            name='yamnet_aves_p3_trunk',
            inputs=[waveform_in],
            outputs=[embedding_out],
            initializer=[c_sr, c_zero, c_one, c_axis0, c_frame_shape, c_crop_starts,
                        c_crop_ends, c_crop_axis, c_flat_shape]
                       + list(yam_proto.graph.initializer) + list(av_proto.graph.initializer),
            value_info=list(yam_proto.graph.value_info) + list(av_proto.graph.value_info),
        )

        opset_imports = {}
        for imp in list(yam_proto.opset_import) + list(av_proto.opset_import):
            opset_imports[imp.domain] = max(opset_imports.get(imp.domain, 0), imp.version)

        model = helper.make_model(
            graph, producer_name='buzzdetect-training',
            opset_imports=[helper.make_opsetid(d, v) for d, v in opset_imports.items()],
            ir_version=max(yam_proto.ir_version, av_proto.ir_version),
        )
        onnx.checker.check_model(model, full_check=False)
        return model
