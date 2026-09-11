import os

import numpy as np

from embedders.embedding import BaseEmbedder

"""YAMNet + AVES on one shared 1.0 s frame grid.

Each frame feeds the full 1.0 s to AVES (bioacoustic wav2vec2, dense 768-d) and
the centre 0.96 s to YAMNet (AudioSet-supervised, sparse 1024-d). The two blocks
are concatenated -> 1792-d, so the probe sees both representations for the same
audio window with the rows aligned by construction. The existing per-embedder
caches cannot be concatenated instead: hop is framelength_s * framehop_prop, so
the yamnet grid (0.96 s) and aves grid (1.0 s) diverge in frame count per chunk.

TF's GPU is hidden here when BUZZDETECT_NO_GPU=1 (YAMNet is tiny on CPU); AVES
keeps CUDA via its own torch path (BUZZDETECT_AVES_DEVICE), matching
embedders/aves. cuDNN is disabled for AVES for the same reason it is in
embedders/aves (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH under the pinned env).

--- RESTORE NOTE (2026-09-10) ---
This is the faithful original from the yamnet-aves session, rewritten from the
authoring transcript after embedders/yamnet_aves/embedder.py was lost with its
worktree (embedders/.gitignore was never unignored for it). `embeddername` is
kept as "yamnet_aves" so it maps onto the existing 560 MB medium cache.

The one detail the notes-only reconstruction (embedders/yamnet_aves/embedder.py,
"YAMNet block NOT exact") could not recover: the YAMNet block is produced by
feeding YAMNet ONE concatenated buffer of all the centred crops
(`self.yamnet(crops.reshape(-1))`) and letting its internal framing
(patch_hop = patch_window = 0.96 s) split it back into per-frame patches -- NOT
by calling `self.yamnet(crop)` once per frame. With a butt-joined buffer each
interior patch's trailing STFT window (~240 samples) pulls real audio from the
next crop; an isolated crop zero-pads that tail. That is the "~1-3% off on most
frames, exact on the occasional frame" the reconstruction measured -- the exact
ones are the last frame of each CHUNK_FRAMES (48) extraction block, where both
paths zero-pad. Verify byte-parity against the cache before trusting this for
re-extraction or shipping.

Reconciled 2026-09-10 on exp/yamnet-aves-head: this verified file replaced the
notes-only reconstruction at this path, embedders/.gitignore was unignored for
yamnet_aves, and both were committed. Byte-parity vs the cache confirmed
(YAMNet max |Δ| 5.6e-5 = GPU/CPU reduction-order noise; AVES exact) over all
3396 frames of the largest shared label pickle.
"""

_YAMNET_SAMPLES = 15360   # 0.96 s at 16 kHz, centred in the 1.0 s frame
_CROP0 = (16000 - _YAMNET_SAMPLES) // 2   # 320


class EmbedderYamnetAves(BaseEmbedder):
    embeddername = "yamnet_aves"
    framelength_s = 1.0
    digits_time = 2
    samplerate = 16000
    n_embeddings = 1792   # 1024 YAMNet + 768 AVES
    dtype_in = 'float32'

    def initialize(self):
        import tensorflow as tf
        from embedders.yamnet.yamnet import WaveformFeatures  # registers custom layer
        _ = WaveformFeatures.dtype

        if os.environ.get('BUZZDETECT_NO_GPU'):
            tf.config.set_visible_devices([], 'GPU')
        else:
            # Both blocks share the one 4 GB card with AVES's torch process. TF's
            # allocator is greedy by default and would claim the whole device,
            # OOMing AVES; memory growth holds it to what YAMNet inference needs
            # (~0.5-1 GB for the 1024-d conv net) and leaves the rest for torch.
            for gpu in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)

        here = os.path.dirname(os.path.realpath(__file__))
        yamnet_keras = os.path.join(here, '..', 'yamnet', 'yamnet.keras')
        self.yamnet = tf.keras.models.load_model(yamnet_keras, compile=False)
        # non-overlapping 0.96 s patches: one embedding per centred crop we feed
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
                f'yamnet_aves: YAMNet returned {len(yam)} frames for {n} crops'
            )

        out = []
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                stop = min(start + self.batch_size, n)
                wav = torch.from_numpy(frames[start:stop]).to(self.device)
                layer_outputs, _ = self.aves.extract_features(wav)
                out.append(layer_outputs[-1].mean(dim=1).cpu().numpy())
        av = np.concatenate(out, axis=0).astype(np.float32)

        return np.concatenate([yam, av], axis=1)

    def to_onnx(self, opset=17):
        """Waveform-in/embeddings-out ONNX graph, built by hand from both halves.

        Neither half is a Keras model on its own -- self.yamnet is, but
        self.aves is a PyTorch module -- so there is no single graph object to
        hand to BaseEmbedder.to_onnx()'s Keras exporter. Each half is exported
        through its own framework's exporter instead (tf2onnx for YAMNet,
        torch.onnx for AVES with the mean-pool folded in), and the two graphs
        are onnx.compose'd together with the same bridge embed() does in
        numpy: reshape the raw waveform into whole 1.0 s frames, dropping any
        remainder (no padding -- see embed() above), centre-crop 0.96 s of
        each frame for YAMNet, run both halves, concatenate.

        This is the template for any future embedder built from more than one
        model or framework: export each piece separately, bridge the shared
        input into each piece's expected shape with a few Slice/Reshape
        nodes, run onnx.compose.add_prefix on each piece to avoid name
        collisions, and Concat the outputs. Nothing in tools/export_onnx.py
        needs to know any of this -- it only calls to_onnx() and checks the
        result against embed().
        """
        import tempfile

        import numpy as np
        import onnx
        import tensorflow as tf
        import tf2onnx
        import torch
        from onnx import TensorProto, compose, helper

        samplerate = self.samplerate

        # --- YAMNet half: flattened crops (n * _YAMNET_SAMPLES,) -> (n, 1024) ---
        # Matches embed(): one concatenated buffer, not one call per crop (see
        # the module docstring's "YAMNet block NOT exact" note above).
        spec = (tf.TensorSpec((None,), tf.float32, name='crops'),)
        yam_proto, _ = tf2onnx.convert.from_keras(
            self.yamnet, input_signature=spec, opset=opset)

        # --- AVES half: frames (n, samplerate) -> (n, 768), mean-pooled ---
        class _AvesMeanPool(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, waveform):
                layer_outputs, _ = self.model.extract_features(waveform)
                return layer_outputs[-1].mean(dim=1)

        # torch.onnx.export traces on whatever device the dummy input and
        # model share; CPU keeps this independent of embed()'s device choice.
        # self.aves is moved back to self.device (unchanged if already CPU)
        # once the export is done, so embed() still runs where it did before.
        wrapper = _AvesMeanPool(self.aves.cpu()).eval()
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

        # waveform -> whole frames, dropping the ragged tail: n = len // sr,
        # then take the first n*sr samples and reshape to (n, sr). All of
        # ends/starts/axes below are int64 tensors because Slice's bounds are
        # graph inputs here, not python ints -- the waveform length isn't
        # known until run time.
        #
        # AVES's own exported graph -- not this bridge -- can't run on a
        # zero-row batch: onnxruntime's Reshape kernel refuses to resolve a
        # -1 dim against a tensor that already has zero elements (seen inside
        # its conv feature extractor's internal layer_norm reshape). An
        # earlier version of this worked around it with an If node that
        # skipped the sub-models entirely on zero frames; that broke
        # optimize()'s batchnorm-fold/Conv+Relu passes and write_fp16() in
        # tools/export_onnx.py, both of which only look at the top-level node
        # list and so silently saw zero convolutions once the real ones were
        # nested inside an If branch. This version stays a single flat graph
        # instead: always run at least one frame (right-padded with silence
        # when there are none), then slice the output back down to the real
        # frame count -- 0 rows if that's what it is -- afterward. A Slice
        # that trims an already-nonempty tensor down to zero rows is not the
        # same operation as a Reshape's -1 trying to resolve against a tensor
        # that's zero-sized _going in_, and onnxruntime is fine with it.
        c_sr = helper.make_tensor('c_samplerate', TensorProto.INT64, [1], [samplerate])
        c_zero = helper.make_tensor('c_zero', TensorProto.INT64, [1], [0])
        c_one = helper.make_tensor('c_one', TensorProto.INT64, [1], [1])
        c_axis0 = helper.make_tensor('c_axis0', TensorProto.INT64, [1], [0])
        c_frame_shape = helper.make_tensor(
            'c_frame_shape', TensorProto.INT64, [2], [-1, samplerate])
        nodes = [
            helper.make_node('Shape', ['waveform'], ['total_len'], name='bridge_shape'),
            # real frame count -- int64 Div truncates -> floor, since len>=0
            helper.make_node('Div', ['total_len', 'c_samplerate'], ['n_frames_real'],
                             name='bridge_n_frames_real'),
            # at least one frame's worth, so the sub-models never see batch=0
            helper.make_node('Max', ['n_frames_real', 'c_one'], ['n_frames_safe'],
                             name='bridge_n_frames_safe'),
            helper.make_node('Mul', ['n_frames_safe', 'c_samplerate'], ['usable_len_safe'],
                             name='bridge_usable_len_safe'),
            # zero-pad the waveform up to usable_len_safe if it's short (a
            # no-op Pad, pad amount 0, whenever n_frames_real is already >=1)
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

        # centre-crop each frame to _YAMNET_SAMPLES for YAMNet, flatten to
        # match its butt-joined-buffer framing (patch_hop == patch_window)
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

        # embedding_safe always has >=1 row (padded when there were none);
        # trim back down to the real frame count, which may be 0. These two
        # nodes consume yam_out/av_out, so they must come after -- not before
        # -- yam_proto's and av_proto's own nodes in the flattened list below;
        # ONNX requires a graph's nodes in topological order.
        trailer = [
            helper.make_node('Concat', [yam_out, av_out], ['embedding_safe'],
                             axis=1, name='bridge_concat'),
            helper.make_node('Slice', ['embedding_safe', 'c_zero', 'n_frames_real', 'c_axis0'],
                             ['embedding'], name='bridge_trim_to_real'),
        ]

        graph = helper.make_graph(
            nodes=nodes + list(yam_proto.graph.node) + list(av_proto.graph.node) + trailer,
            name='yamnet_aves_trunk',
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
