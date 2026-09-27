"""Shared base for depth12 trunk embedders that add transformed views of each
frame: [plain, view_1, ..., view_k], every view through ONE shared fine-tuned
tail (TimeDistributed, n_ctx = k + 1), codes concatenated into the head.

Not an embedder itself (no directory, never loaded by name). Subclasses set
`views` (method names below, in order) and `n_ctx`/`n_embeddings` to match.

Built 2026-09-26 for the pitch-shift method x direction grid (see HANDOFF.md):
resample (the v4-ft-ps mechanism), resample over both halves of the frame, and
phase vocoder (librosa.effects.pitch_shift, duration-preserving).

`views_to_onnx()` exports the resample views (2026-09-27); it is also what the
older multi-view classes (`yamnet_trunk_pitchshift_depth12` and its updown/down
subclasses) call. Vocoder views have no ONNX form yet: CV only.
"""
import importlib

import numpy as np

_trunk12 = importlib.import_module('embedders.yamnet_trunk_depth12.embedder')

VIEW_WIDTH = _trunk12.EmbedderYamnetTrunkDepth12.n_embeddings  # 12288


# --- ONNX export ---------------------------------------------------------------
#
# Each resample view is (crop to a slice of the frame) -> (2:1 resample) ->
# (tile or pad back to one frame). librosa.resample is linear and, for a 2:1
# ratio, time-invariant up to the stride with zero padding at the slice edges,
# so each direction is one FIR read off librosa by impulses (as
# yamnet_pitchshift._resample_kernel does for the up shift):
#   'down' (sr -> sr/2, the octave-UP view): y[m] = sum_k h(2m - k) x[k]
#   'up'   (sr -> 2 sr, the octave-DOWN views): y[m] = sum_k h(m - 2k) x[k]
# The first is a stride-2 Conv; the second is zero-stuffing then a Conv.

# view name -> (slice start, slice stop) as fractions of the frame, resample
# direction ('down' = sr/2, 'up' = 2 sr), and how the result fills the frame.
ONNX_VIEWS = {
    'resample_up': ((0, 1), 'down', 'tile'),
    'resample_down_centre': ((0.25, 0.75), 'up', 'fit'),
    'resample_down_first': ((0, 0.5), 'up', 'fit'),
    'resample_down_second': ((0.5, 1), 'up', 'fit'),
}


def _fir(samplerate, L, direction):
    """(Conv kernel, left pad, right pad) reproducing librosa.resample on L samples."""
    import librosa

    target = samplerate // 2 if direction == 'down' else samplerate * 2
    taps = {}
    for k0 in (L // 2, L // 2 + 1):
        impulse = np.zeros(L, dtype=np.float32)
        impulse[k0] = 1
        out = librosa.resample(impulse, orig_sr=samplerate, target_sr=target)
        for m in np.nonzero(out)[0]:
            d = 2 * int(m) - k0 if direction == 'down' else int(m) - 2 * k0
            taps[d] = out[m]
    lo, hi = min(taps), max(taps)
    # Conv is cross-correlation: w[t] = h(pad_left - t), pad_left = max offset
    w = np.zeros(hi - lo + 1, dtype=np.float32)
    for d, v in taps.items():
        w[hi - d] = v
    return w, hi, len(w) - 1 - hi


def views_to_onnx(embedder, views, opset=17):
    """Waveform -> [trunk(plain) | trunk(view_1) | ...] as one ONNX graph.

    Matches the numpy embed() of every multi-view depth12 embedder: whole
    frames only, each view computed per frame, and each view's frames run
    through the trunk in their own call (so a separate trunk copy here: one
    copy over all views' frames would let YAMNet's STFT see across view
    boundaries). Output (n_frames, VIEW_WIDTH * (1 + len(views))).
    """
    import onnx
    from onnx import TensorProto, compose, helper, numpy_helper

    from embedders.onnx_context import crop_waveform_to_whole_frames

    S = int(round(embedder.framelength_s * embedder.samplerate))
    trunk = _trunk12.EmbedderYamnetTrunkDepth12.to_onnx(embedder, opset=opset)
    serial = trunk.SerializeToString()
    graph = trunk.graph

    inits, nodes, back = [], [], []  # nodes run before the trunks, back after

    def const(name, values):
        inits.append(helper.make_tensor(name, TensorProto.INT64, [len(values)], values))
        return name

    new_in = 'tv_waveform'
    nodes += [
        helper.make_node('Identity', [new_in], [graph.input[0].name]),
        helper.make_node('Reshape', [new_in, const('tv_frames_shape', [-1, 1, S])], ['tv_frames']),
    ]
    const('tv_flat', [-1])
    const('tv_axis2', [2])
    const('tv_view_width', [-1, VIEW_WIDTH])
    flats = []

    def flatten(out_name, p):
        back.append(helper.make_node('Reshape', [out_name, 'tv_view_width'], [f'{p}flat']))
        flats.append(f'{p}flat')

    flatten(graph.output[0].name, 'tv0_')

    for i, view in enumerate(views, start=1):
        if view not in ONNX_VIEWS:
            raise NotImplementedError(
                f"{embedder.embeddername}: view '{view}' has no ONNX form "
                f'(exportable: {sorted(ONNX_VIEWS)})')
        (a, b), direction, fill = ONNX_VIEWS[view]
        p = f'tv{i}_'
        start, stop = int(round(a * S)), int(round(b * S))
        L = stop - start
        nodes.append(helper.make_node(
            'Slice', ['tv_frames', const(f'{p}start', [start]), const(f'{p}stop', [stop]),
                      'tv_axis2'], [f'{p}slice']))
        w, pad_l, pad_r = _fir(embedder.samplerate, L, direction)
        inits.append(numpy_helper.from_array(w.reshape(1, 1, -1), f'{p}kernel'))
        if direction == 'down':
            nodes.append(helper.make_node(
                'Conv', [f'{p}slice', f'{p}kernel'], [f'{p}resampled'],
                pads=[pad_l, pad_r], strides=[2], kernel_shape=[len(w)]))
            n_out = L // 2
        else:
            # zero-stuff: x[k] -> u[2k], u[2k + 1] = 0
            nodes += [
                helper.make_node('Unsqueeze', [f'{p}slice', const(f'{p}axis3', [3])], [f'{p}col']),
                helper.make_node('Sub', [f'{p}col', f'{p}col'], [f'{p}zeros']),
                helper.make_node('Concat', [f'{p}col', f'{p}zeros'], [f'{p}pairs'], axis=3),
                helper.make_node('Reshape', [f'{p}pairs', const(f'{p}stuffed_shape', [-1, 1, 2 * L])],
                                 [f'{p}stuffed']),
                helper.make_node('Conv', [f'{p}stuffed', f'{p}kernel'], [f'{p}resampled'],
                                 pads=[pad_l, pad_r], strides=[1], kernel_shape=[len(w)]),
            ]
            n_out = 2 * L
        if fill == 'tile':
            reps = -(-S // n_out)
            nodes.append(helper.make_node('Concat', [f'{p}resampled'] * reps, [f'{p}tiled'], axis=2))
            filled = f'{p}tiled'
        else:
            filled = f'{p}resampled'
        if n_out * (reps if fill == 'tile' else 1) != S:
            raise NotImplementedError(f'{view}: resampled length does not fill {S} exactly')

        twin = compose.add_prefix(onnx.ModelProto.FromString(serial), prefix=p)
        nodes.append(helper.make_node('Reshape', [filled, 'tv_flat'], [twin.graph.input[0].name]))
        graph.node.extend(twin.graph.node)
        graph.initializer.extend(twin.graph.initializer)
        graph.value_info.extend(twin.graph.value_info)
        flatten(twin.graph.output[0].name, p)

    back.append(helper.make_node('Concat', flats, ['tv_embeddings'], axis=1))
    for node in reversed(nodes):
        graph.node.insert(0, node)
    graph.node.extend(back)
    graph.initializer.extend(inits)

    del graph.input[:]
    graph.input.append(helper.make_tensor_value_info(new_in, TensorProto.FLOAT, ['samples']))
    del graph.output[:]
    graph.output.append(helper.make_tensor_value_info('tv_embeddings', TensorProto.FLOAT, None))

    # embed() drops the ragged tail and returns 0 rows under one frame
    return crop_waveform_to_whole_frames(trunk, S)


class TrunkViewsDepth12(_trunk12.EmbedderYamnetTrunkDepth12):
    views = ()

    def initialize(self):
        import librosa
        self._librosa = librosa
        self._frame_samples = int(round(self.framelength_s * self.samplerate))
        super().initialize()  # loads self.model = trunk through layer11, warms it

    # --- views: (n, S) float32 frames -> (n, S) float32 ---------------------

    def _fit(self, x):
        """Pad or crop a 1-D signal to one frame."""
        S = self._frame_samples
        if len(x) < S:
            x = np.pad(x, (0, S - len(x)))
        return x[:S].astype(np.float32)

    def _up2(self, x):
        # resample to half rate, relabel at full rate: one octave up, half length
        return self._librosa.resample(x, orig_sr=self.samplerate, target_sr=self.samplerate // 2)

    def _down2(self, x):
        # resample to double rate, relabel at full rate: one octave down, double length
        return self._librosa.resample(x, orig_sr=self.samplerate, target_sr=self.samplerate * 2)

    def resample_up(self, frames):
        """v4-ft-ps's up view: whole frame an octave up, tiled twice to fill."""
        return np.stack([self._fit(np.tile(self._up2(f), 2)) for f in frames])

    def resample_down_first(self, frames):
        """First half of the frame (0-0.48 s), an octave down, filling the frame."""
        h = self._frame_samples // 2
        return np.stack([self._fit(self._down2(f[:h])) for f in frames])

    def resample_down_second(self, frames):
        """Second half of the frame (0.48-0.96 s), an octave down, filling the frame."""
        h = self._frame_samples // 2
        return np.stack([self._fit(self._down2(f[h:2 * h])) for f in frames])

    def vocoder_up(self, frames):
        """Phase vocoder, +12 semitones, same duration. Batched over frames."""
        return self._librosa.effects.pitch_shift(
            frames, sr=self.samplerate, n_steps=12).astype(np.float32)

    def vocoder_down(self, frames):
        """Phase vocoder, -12 semitones, same duration. Batched over frames."""
        return self._librosa.effects.pitch_shift(
            frames, sr=self.samplerate, n_steps=-12).astype(np.float32)

    def to_onnx(self, opset=17):
        return views_to_onnx(self, self.views, opset=opset)

    # --- embed ----------------------------------------------------------------

    def embed(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio) // self._frame_samples
        if n == 0:
            return np.empty((0, self.n_embeddings), dtype=np.float16)
        usable = audio[:n * self._frame_samples]

        trunk_embed = _trunk12.EmbedderYamnetTrunkDepth12.embed
        plain = trunk_embed(self, usable)  # (n, 12288) float16, layer11 features
        if len(plain) != n:
            raise ValueError(
                f'{self.embeddername}: trunk returned {len(plain)} frames for {n} input frames'
            )

        frames = usable.reshape(n, self._frame_samples)
        out = [plain]
        for v in self.views:
            shifted = getattr(self, v)(frames)
            out.append(trunk_embed(self, shifted.reshape(-1)))

        return np.concatenate(out, axis=1).astype(np.float16)
