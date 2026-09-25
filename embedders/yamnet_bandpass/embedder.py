import os

import numpy as np

from embedders.embedding import BaseEmbedder


class EmbedderYamnetBandpass(BaseEmbedder):
    embeddername = "yamnet_bandpass"
    framelength_s = 0.96
    digits_time = 2
    samplerate = 16000
    n_embeddings = 1024
    dtype_in = 'float32'

    LOWCUT_HZ  = 100
    HIGHCUT_HZ = 3000
    NUM_TAPS   = 201  # odd → linear phase, group delay = (N-1)/2 = 100 samples

    def initialize(self):
        import tensorflow as tf
        from embedders.yamnet.yamnet import WaveformFeatures  # needed for model loading
        _ = WaveformFeatures.dtype
        curdir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(curdir, '..', 'yamnet', 'yamnet.keras')
        model = tf.keras.models.load_model(model_path, compile=False)
        model.layers[1].params.patch_hop_seconds = self.framehop_s
        self.model = model
        self._filter_kernel = self._build_filter_kernel()
        # Force TF thread pool init before librosa/Accelerate claims threads; without
        # this the first embed() call after librosa.resample deadlocks on macOS.
        self.embed(np.zeros(int(self.framelength_s * self.samplerate), dtype=np.float32))

    def _build_filter_kernel(self):
        """Hamming-windowed sinc bandpass FIR, returned as a tf constant."""
        import tensorflow as tf
        sr   = self.samplerate
        low  = self.LOWCUT_HZ
        high = self.HIGHCUT_HZ
        n    = self.NUM_TAPS

        # Centered sample indices
        t = np.arange(n) - (n - 1) / 2.0

        # Bandpass = LP(high) − LP(low); np.sinc is normalized (sin(πx)/(πx))
        h = (2 * high / sr) * np.sinc(2 * high / sr * t) \
          - (2 * low  / sr) * np.sinc(2 * low  / sr * t)
        h *= np.hamming(n)

        # Normalize so passband gain ≈ 1 at center frequency
        center = (low + high) / 2.0
        response_at_center = np.dot(h, np.exp(2j * np.pi * center / sr * np.arange(n)))
        h /= np.abs(response_at_center)

        # Shape expected by tf.nn.conv1d: [filter_width, in_channels, out_channels]
        return tf.constant(h.astype(np.float32).reshape(n, 1, 1), dtype=tf.float32)

    def embed(self, audio):
        import tensorflow as tf
        x = tf.cast(audio, tf.float32)
        # conv1d wants [batch, width, channels]
        x = x[tf.newaxis, :, tf.newaxis]
        filtered = tf.nn.conv1d(x, self._filter_kernel, stride=1, padding='SAME')
        filtered = tf.squeeze(filtered, axis=[0, 2])
        return self.model(filtered)

    def to_onnx(self, opset=17):
        """Waveform -> bandpass FIR -> YAMNet trunk, as one ONNX graph.

        The default BaseEmbedder.to_onnx() exports self.model alone, i.e. the
        *unfiltered* trunk -- embed()'s conv1d bandpass never makes it into the
        graph. This prepends the same Hamming-windowed sinc FIR as a
        'SAME'-padded Conv (stride 1, odd kernel -> pad NUM_TAPS//2 each side,
        matching TF's SAME rule for this case) ahead of the trunk's input, so
        the graph sees the same filtered audio the model does in embed().
        TF's conv1d and ONNX's Conv are both cross-correlation with the same
        weight-order convention, so the kernel carries over unchanged.
        """
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        trunk = super().to_onnx(opset=opset)
        graph = trunk.graph
        yam_in = graph.input[0].name

        kernel = self._filter_kernel.numpy().reshape(1, 1, self.NUM_TAPS)
        half = (self.NUM_TAPS - 1) // 2
        new_in = 'bp_waveform'

        graph.initializer.extend([
            numpy_helper.from_array(kernel.astype(np.float32), 'bp_kernel'),
            helper.make_tensor('bp_frame_shape', TensorProto.INT64, [3], [1, 1, -1]),
            helper.make_tensor('bp_flat_shape', TensorProto.INT64, [1], [-1]),
        ])
        front = [
            helper.make_node('Reshape', [new_in, 'bp_frame_shape'], ['bp_framed']),
            helper.make_node('Conv', ['bp_framed', 'bp_kernel'], ['bp_filtered'],
                             pads=[half, half], strides=[1], kernel_shape=[self.NUM_TAPS]),
            helper.make_node('Reshape', ['bp_filtered', 'bp_flat_shape'], [yam_in]),
        ]
        del graph.input[:]
        graph.input.append(helper.make_tensor_value_info(new_in, TensorProto.FLOAT, ['samples']))
        for node in reversed(front):
            graph.node.insert(0, node)

        return trunk
