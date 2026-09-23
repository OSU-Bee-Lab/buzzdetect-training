import os
import importlib

import numpy as np

# Imported through the module so load_embedder()'s scan for a BaseEmbedder
# subclass in this namespace only finds EmbedderYamnetTriple.
_yamnet = importlib.import_module('embedders.yamnet.embedder')

"""YAMNet at 3x frame length: one 2.88 s frame = three contiguous 0.96 s
YAMNet patches, their embeddings concatenated in time order (3072-d).

Not a context model: each row is its own frame with its own label, and rows
don't overlap or borrow from neighbours. The classifier just sees a wider
frame.

The trunk does the same work per second of audio as plain `yamnet` at
framehop 1 -- three YAMNet patches per 2.88 s either way. The saving is 3x
fewer rows downstream (head, thresholds, results) and nothing else. A frame
also dilutes a short buzz across 2.88 s; labels follow framelength.

Lean path: the whole thing is one TF call. The waveform is framed into
2.88 s windows with tf.signal.frame, flattened, handed to YAMNet with its
patch hop pinned at 0.96 s, and the (3n, 1024) output reshaped to
(n, 3072) -- all on-device, one host copy at the end. At framehop_prop 1 the
framing is a no-op reshape and the result is bit-identical to plain YAMNet's
patches regrouped by three.
"""

SUBFRAMES = 3


class EmbedderYamnetTriple(_yamnet.EmbedderYamnet):
    embeddername = "yamnet_triple"
    framelength_s = _yamnet.EmbedderYamnet.framelength_s * SUBFRAMES  # 2.88
    n_embeddings = _yamnet.EmbedderYamnet.n_embeddings * SUBFRAMES    # 3072

    def initialize(self):
        import tensorflow as tf
        self._tf = tf
        self._sub_samples = int(round(_yamnet.EmbedderYamnet.framelength_s * self.samplerate))
        self._frame_samples = self._sub_samples * SUBFRAMES
        self._hop_samples = int(round(self.framehop_s * self.samplerate))
        # Not super().initialize(): it pins the patch hop to self.framehop_s
        # (2.88 s here) and warms up through embed() before we could fix it.
        from embedders.yamnet.yamnet import WaveformFeatures  # registers custom layer
        _ = WaveformFeatures.dtype
        model_path = os.path.join(os.path.dirname(os.path.realpath(_yamnet.__file__)), 'yamnet.keras')
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model.layers[1].params.patch_hop_seconds = _yamnet.EmbedderYamnet.framelength_s
        # warm the TF thread pool before librosa/Accelerate claims threads
        self.embed(np.zeros(self._frame_samples, dtype=np.float32))

    def embed(self, audio):
        """Audio at self.samplerate -> (n_frames, 3072); ragged tail dropped."""
        tf = self._tf
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        if int(audio.shape[0]) < self._frame_samples:
            return np.zeros((0, self.n_embeddings), dtype=np.float32)
        if self._hop_samples == self._frame_samples:
            # Non-overlapping: frames already tile the buffer; crop, don't copy.
            n = int(audio.shape[0]) // self._frame_samples
            sub = self.model(audio[:n * self._frame_samples])
        else:
            frames = tf.signal.frame(audio, self._frame_samples, self._hop_samples)
            n = int(frames.shape[0])
            sub = self.model(tf.reshape(frames, [-1]))
        # Wrong patch count here means a patch-hop mismatch, which would
        # silently shuffle subframes across rows.
        if int(sub.shape[0]) != n * SUBFRAMES:
            raise ValueError(f'{self.embeddername}: YAMNet returned {sub.shape[0]} '
                             f'patches for {n} frames, expected {n * SUBFRAMES}')
        return tf.reshape(sub, [n, self.n_embeddings]).numpy()

    def to_onnx(self, opset=17):
        """YAMNet's trunk (patch hop 0.96 s) with the waveform cropped to whole
        2.88 s frames in front and a Slice + Reshape to (n, 3072) behind.
        Assumes framehop_prop=1, which is what export_onnx.py exports with."""
        from onnx import TensorProto, helper

        trunk = super().to_onnx(opset=opset)
        graph = trunk.graph
        p = 'triple'
        old_in = graph.input[0].name
        new_in = f'{p}_waveform'
        trunk_out = graph.output[0].name

        def const(name, values):
            return helper.make_tensor(f'{p}_{name}', TensorProto.INT64, [len(values)], values)

        graph.initializer.extend([
            const('zero', [0]), const('frame_samples', [self._frame_samples]),
            const('subframes', [SUBFRAMES]), const('shape_out', [-1, self.n_embeddings]),
        ])
        front = [
            helper.make_node('Shape', [new_in], [f'{p}_len']),
            helper.make_node('Div', [f'{p}_len', f'{p}_frame_samples'], [f'{p}_n']),
            helper.make_node('Mul', [f'{p}_n', f'{p}_frame_samples'], [f'{p}_usable']),
            helper.make_node('Slice', [new_in, f'{p}_zero', f'{p}_usable', f'{p}_zero'], [old_in]),
        ]
        # Trim to 3n patches first: on a sub-frame input YAMNet pads up to one
        # patch, which must become zero rows, not a reshape error.
        back = [
            helper.make_node('Mul', [f'{p}_n', f'{p}_subframes'], [f'{p}_n_sub']),
            helper.make_node('Slice', [trunk_out, f'{p}_zero', f'{p}_n_sub', f'{p}_zero'], [f'{p}_trim']),
            helper.make_node('Reshape', [f'{p}_trim', f'{p}_shape_out'], [f'{p}_out']),
        ]
        for node in reversed(front):
            graph.node.insert(0, node)
        graph.node.extend(back)
        del graph.input[:]
        graph.input.append(helper.make_tensor_value_info(new_in, TensorProto.FLOAT, ['samples']))
        del graph.output[:]
        graph.output.append(helper.make_tensor_value_info(
            f'{p}_out', TensorProto.FLOAT, ['frames', self.n_embeddings]))
        return trunk
