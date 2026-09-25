import os

import numpy as np

from embedders.embedding import BaseEmbedder

"""YAMNet, plus a pitch-shifted copy of the same frame, concatenated.

IDEAS.md item 5. Honey-bee flight is ~230 Hz with harmonics; YAMNet's mel
filterbank starts at 125 Hz and is near-linear (coarse) below 1 kHz, where
AudioSet's training mass is thin. This shifts each frame up an octave before
a second embedding pass, landing the fundamental and first harmonics at
460/920/1380 Hz where the filterbank is denser and AudioSet has more mass --
then concatenates that block with the unshifted one, so the probe can use
whichever (or both) it finds useful. A straight replacement was tried and
sank (`yamnet-bandpass` -0.047, `yamnet-mask` -0.143, both E1) -- this is a
concat specifically to avoid that failure mode.

The shift is by resampling, not a formant-preserving pitch shifter: each 0.96 s
frame (15360 samples at 16 kHz) is downsampled to 8 kHz (halves the sample
count, low-pass filtering content above 4 kHz of the ORIGINAL signal in the
process) and the result is relabelled as 16 kHz -- so playing it back takes
half the real time, and every frequency in it is doubled. That resampled clip
is then tiled 2x and cropped back to 15360 samples, so YAMNet still sees a
full 0.96 s patch (its patch_hop == patch_window == 0.96 s; feeding it a
shorter buffer changes the frame grid, which is exactly the confound that sank
`perch-probe` -- see framehop-overlap in the archive).

`embedders/yamnet_doublerate/` looks like a shortcut for this but is NOT one:
it changes framelength_s to 0.48s, which changes the frame grid, frame count
and overlap_event_s system-wide. This embedder holds framelength_s at YAMNet's
native 0.96 s and does the resampling entirely inside embed(), so the frame
grid and every label stay byte-identical to plain `yamnet`.
"""

_FRAME_SAMPLES_HALF_SR = None  # set in initialize() once samplerate is fixed


class EmbedderYamnetPitchshift(BaseEmbedder):
    embeddername = "yamnet_pitchshift"
    framelength_s = 0.96  # seconds -- YAMNet's native grid, untouched
    digits_time = 2
    samplerate = 16000  # Hz
    n_embeddings = 2048  # 1024 unshifted + 1024 pitch-shifted
    dtype_in = 'float32'

    def initialize(self):
        import tensorflow as tf
        from embedders.yamnet.yamnet import WaveformFeatures  # registers custom layer
        _ = WaveformFeatures.dtype

        curdir = os.path.dirname(os.path.realpath(__file__))
        yamnet_keras = os.path.join(curdir, '..', 'yamnet', 'yamnet.keras')
        model = tf.keras.models.load_model(yamnet_keras, compile=False)
        model.layers[1].params.patch_hop_seconds = self.framehop_s
        self.model = model

        import librosa
        self._librosa = librosa
        self._frame_samples = int(round(self.framelength_s * self.samplerate))
        self._half_sr = self.samplerate // 2

        # Force TF thread pool init before librosa/Accelerate claims threads --
        # see embedders/yamnet/embedder.py for why.
        self.embed(np.zeros(self._frame_samples, dtype=np.float32))

    def _pitch_up_octave(self, frame):
        """One 0.96 s frame -> its octave-shifted, re-tiled twin. See module
        docstring for the mechanism."""
        shifted = self._librosa.resample(
            frame, orig_sr=self.samplerate, target_sr=self._half_sr,
        ).astype(np.float32)
        tiled = np.tile(shifted, 2)
        # resample's output length can be off by a sample or two from the
        # exact half; pad or trim back to precisely one frame so the
        # concatenated multi-frame buffer stays on YAMNet's exact patch grid.
        if len(tiled) < self._frame_samples:
            tiled = np.pad(tiled, (0, self._frame_samples - len(tiled)))
        return tiled[:self._frame_samples]

    def embed(self, audio):
        """
        Generate embeddings for audio data

        Args:
            audio: numpy array of audio samples at self.samplerate, a whole
                number of self.framelength_s frames (extraction always calls
                with framehop_prop=1, i.e. non-overlapping frames -- see
                02_set/extract.py)

        Returns:
            numpy array of embeddings, (n_frames, 2048)
        """
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio) // self._frame_samples
        if n == 0:
            return np.empty((0, self.n_embeddings), dtype=np.float32)
        usable = audio[:n * self._frame_samples]

        unshifted = np.asarray(self.model(usable), dtype=np.float32)
        if len(unshifted) != n:
            raise ValueError(
                f'yamnet_pitchshift: YAMNet returned {len(unshifted)} frames '
                f'for {n} input frames'
            )

        frames = usable.reshape(n, self._frame_samples)
        shifted_frames = np.stack([self._pitch_up_octave(f) for f in frames])
        shifted = np.asarray(self.model(shifted_frames.reshape(-1)), dtype=np.float32)

        return np.concatenate([unshifted, shifted], axis=1)

    def _resample_kernel(self):
        """The FIR that librosa.resample(sr -> sr/2) applies, as a conv kernel.

        librosa's resampler is linear and, for a 2:1 ratio, time-invariant
        up to the stride: y[m] = sum_k h[2m - k] x[k] with zero padding at the
        frame edges. h is read off by resampling two unit impulses (one on an
        even input sample, one on an odd one) rather than re-deriving
        soxr's filter, so the ONNX graph tracks whatever librosa does.
        Returned flipped (Conv is cross-correlation), with its half-width L;
        max deviation from librosa.resample on noise is ~6e-7.
        """
        n = self._frame_samples
        centre = n // 2
        taps = {}
        for k0 in (centre, centre + 1):
            impulse = np.zeros(n, dtype=np.float32)
            impulse[k0] = 1
            out = self._librosa.resample(
                impulse, orig_sr=self.samplerate, target_sr=self._half_sr)
            for m in np.nonzero(out)[0]:
                taps[2 * int(m) - k0] = out[m]
        half = max(abs(o) for o in taps)
        h = np.zeros(2 * half + 1, dtype=np.float32)
        for o, v in taps.items():
            h[o + half] = v
        return h[::-1].copy(), half

    def to_onnx(self, opset=17):
        """Waveform -> [YAMNet | YAMNet(pitch-shifted)] as one ONNX graph.

        The default BaseEmbedder.to_onnx() exports self.model alone, i.e. the
        unshifted 1024 block only. This wraps that trunk: whole 0.96 s frames
        are downsampled 2:1 with the librosa-equivalent FIR (_resample_kernel)
        as a stride-2 Conv, tiled 2x, and appended to the unshifted frames, so
        a second copy of the YAMNet trunk embeds them, and the two outputs are
        concatenated along the feature axis (embed()'s two separate calls).
        """
        import onnx
        from onnx import TensorProto, compose, helper, numpy_helper

        from embedders.onnx_context import crop_waveform_to_whole_frames

        trunk = super().to_onnx(opset=opset)
        # Second copy of the trunk for the shifted block, merged into the same
        # graph. One copy over the concatenated 2n frames is NOT equivalent:
        # YAMNet's edge padding means the last unshifted frame would see the
        # first shifted one as context (error ~0.4 on that row).
        twin = compose.add_prefix(onnx.ModelProto.FromString(trunk.SerializeToString()),
                                  prefix='ps_twin_')
        kernel, half = self._resample_kernel()
        n_s = self._frame_samples
        graph = trunk.graph
        yam_in = graph.input[0].name
        yam_out = graph.output[0].name
        twin_in = twin.graph.input[0].name
        twin_out = twin.graph.output[0].name
        new_in = 'ps_waveform'
        graph.node.extend(twin.graph.node)
        graph.initializer.extend(twin.graph.initializer)
        graph.value_info.extend(twin.graph.value_info)

        graph.initializer.extend([
            numpy_helper.from_array(kernel.reshape(1, 1, -1), 'ps_kernel'),
            helper.make_tensor('ps_frames_shape', TensorProto.INT64, [3], [-1, 1, n_s]),
            helper.make_tensor('ps_flat_shape', TensorProto.INT64, [1], [-1]),
        ])
        front = [
            helper.make_node('Identity', [new_in], [yam_in]),
            helper.make_node('Reshape', [new_in, 'ps_frames_shape'], ['ps_frames']),
            helper.make_node('Conv', ['ps_frames', 'ps_kernel'], ['ps_down'],
                             pads=[half, half], strides=[2], kernel_shape=[len(kernel)]),
            helper.make_node('Concat', ['ps_down', 'ps_down'], ['ps_tiled'], axis=2),
            helper.make_node('Reshape', ['ps_tiled', 'ps_flat_shape'], [twin_in]),
        ]
        back = [helper.make_node('Concat', [yam_out, twin_out], ['ps_embeddings'], axis=1)]
        del graph.input[:]
        graph.input.append(helper.make_tensor_value_info(new_in, TensorProto.FLOAT, ['samples']))
        for node in reversed(front):
            graph.node.insert(0, node)
        graph.node.extend(back)
        graph.output[0].name = 'ps_embeddings'
        del graph.output[0].type.tensor_type.shape.dim[:]

        # embed() drops the ragged tail and returns 0 rows under one frame
        return crop_waveform_to_whole_frames(trunk, n_s)
