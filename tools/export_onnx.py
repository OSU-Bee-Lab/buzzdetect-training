"""Export a trained model into buzzdetect's engine/models/.

    conda run -n buzzdetect-train python tools/export_onnx.py cv_baseline

Reads models/<name>/ here and writes <dest>/<name>/ over in buzzdetect. The
ONNX half is one graph that takes a waveform and returns predictions -- the
embedder's log-mel front end, the embedder's trunk and the trained classifier
head, all fused into a single model.onnx. buzzdetect runs that graph and
nothing else: no embedder plugin, no NumPy front end, no TensorFlow.

The Keras weights stay here. buzzdetect gets the ONNX graph and nothing else,
which is why the export is worth checking rather than trusting: every run
compares the graph against the Keras model it was built from, on real audio and
on the lengths that exercise the front end's padding, and refuses to write a
mismatch.

Why fuse the front end in rather than keeping the embedder separate, which is
how buzzdetect worked before:

  - It was the single largest cost in the pipeline. The NumPy front end took
    118 ms of a 170 ms chunk on macOS and 147 ms of 177 ms on the Linux box,
    all of it on the CPU, competing with the audio decoder threads for cores.
    In the graph it runs wherever the rest of the graph runs; the same chunk
    now takes 68 ms on CoreML.
  - It removes the second implementation. The front end existed twice, once in
    TensorFlow and once in NumPy, held together by a parity test.
  - One file is the whole model, so a newly trained model is shippable the
    moment it is exported.

See buzzdetect's benchmarks/onnx-vs-tf/{RESULTS.md,COREML.md} for the numbers.
"""

# IMPORTANT: `import tensorflow` must come first, before anything that pulls in
# pandas. Keras/TF and pandas both link a copy of the same native libraries, and
# whichever loads second can bind against the first one's symbols; importing TF
# first is the ordering that works. See any 02_/03_ entry point for the full
# explanation.
import tensorflow  # noqa: F401  (import-order guard, see above)

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg  # noqa: E402
from tools.onnx_passes import count_fusable, optimize  # noqa: E402

# Per-machine -- buzzdetect's checkout lives wherever this box put it. Set
# "buzzdetect_dest" in paths.local.json (gitignored; see
# paths.local.example.json) rather than hardcoding it here.
DEST_DEFAULT = cfg.local('buzzdetect_dest')

# Five minutes of field audio from Lily - Fit+Fast/2023_R3_Marysville/53. Real
# audio matters here: the synthetic lengths below run on gaussian noise, which
# proves the graph exported correctly but says nothing about behaviour on the
# post-ReLU, sparse, non-negative activations the model actually sees.
FIXTURE_AUDIO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'fixtures', '230808_1208_s89520.flac')

# Loose enough for float32 reassociation between two runtimes -- and for the
# batchnorm fold, which is exact in real arithmetic but moves the last bits --
# tight enough to catch a mis-exported graph. The fixture lands around 4e-5.
TOL = 1e-4

# The graph's input and output are renamed to these on the way out. Nothing
# requires it -- the engine reads them by index -- but the engine also has to
# name the input's symbolic dimension to fix it before session creation, and a
# contract worth relying on is worth writing down.
NAME_IN = 'waveform'
NAME_OUT = 'predictions'
DIM_SAMPLES = 'samples'

# The trunk and the head are exported separately now (see build_trunk_onnx()
# and head_to_onnx()) and then onnx.compose'd together, which requires them to
# share an opset. Pinned rather than left to each exporter's default so a
# future embedder's to_onnx() and this file agree without coordinating.
OPSET = 17

# The reduced-precision sibling. buzzdetect loads it instead of model.onnx when
# a run sets BUZZDETECT_GPU_FP16=1 and the provider can act on it -- today only
# CoreML, where it reaches the Neural Engine and is worth 1.9x end to end at
# ~2e-2 on the predictions (buzzdetect's benchmarks/onnx-vs-tf/COREML.md).
# Reduced precision is a different file rather than a provider option because
# handing an fp32 graph to the Neural Engine means CoreML's NeuralNetwork
# format, which declines the FusedConv nodes and ends up slower than the CPU.
FNAME_FP16 = 'model.fp16.onnx'

# What the fp16 graph is allowed to differ by. Two orders of magnitude past
# TOL, and deliberately so: this is not a parity check, it is a check that the
# conversion produced the model rather than mush. The number it actually lands
# on is printed, and the engine warns the user when it loads this file.
TOL_FP16 = 5e-2

# And how much of the top class it is allowed to change. A shifted activation
# is expected; a different answer on one frame in fifty is not.
AGREE_FP16 = 0.98

# The engine's config_model.json is the whole model manifest -- there is no
# model.py any more. It carries the class list (which names the result columns),
# the rounding for the time and activation columns, and the framing parameters
# the engine needs to turn a chunk into frames. `classes` and `digits_results`
# come from the training config; the rest is measured off the exported graph and
# the embedder here (see where config_out is built in export()). Keep it in step
# with the engine's src/inference/models.py::REQUIRED_CONFIG_KEYS. Training
# metadata is not shipped.


def load_head(dir_src, n_embeddings):
    """The trained classifier, from a model.keras or from a SavedModel directory.

    Everything trained here is a model.keras. The SavedModel branch is for
    models released before that -- model_general_v3 is one -- whose weights
    exist only in the form TensorFlow 2 saved them in, and which would
    otherwise have to be retrained to be shipped again.
    """
    import keras

    path_keras = os.path.join(dir_src, 'model.keras')
    if os.path.exists(path_keras):
        head = keras.saving.load_model(path_keras, compile=False)
        # Keras refuses to hand out .inputs for a model it has never seen
        # called, and refuses to export one either.
        head(np.zeros((1, n_embeddings), dtype=np.float32))
        return head

    if os.path.exists(os.path.join(dir_src, 'saved_model.pb')):
        from keras.layers import TFSMLayer
        return TFSMLayer(dir_src, call_endpoint='serving_default')

    raise SystemExit(f'no model.keras and no saved_model.pb in {dir_src}')


def build_trunk_onnx(embeddername):
    """The embedder's own waveform -> embeddings ONNX graph.

    Loaded through the embedder plugin interface rather than by reaching for
    a .keras path, so whatever that plugin does at load time -- retuning the
    patch hop, in yamnet's case -- is done here too. The actual ONNX-building
    is the embedder's own job (BaseEmbedder.to_onnx(), overridden per
    embedder as needed -- see embedders/yamnet_aves/embedder.py for a trunk
    spanning two frameworks); a new embedder, however it's built, needs no
    change in this file, only in its own to_onnx().
    """
    from embedders.embedding import load_embedder

    # framehop_prop=1: the graph is exported with the patch hop welded to the
    # patch window. An overlapping framehop would have to be a graph parameter,
    # and buzzdetect's ONNX path has never supported one.
    embedder = load_embedder(embeddername, framehop_prop=1, initialize=True)
    try:
        trunk_onnx = embedder.to_onnx(opset=OPSET)
    except NotImplementedError as e:
        raise SystemExit(str(e))

    trunk_onnx = flatten_trunk_output(trunk_onnx, embedder)
    return trunk_onnx, embedder


def flatten_trunk_output(trunk_onnx, embedder):
    """Make sure the trunk graph's output is (frames, n_embeddings).

    A trunk-fine-tune embedder (yamnet_trunk, yamnet_trunk11, ...) hands back
    its frozen layers' raw spatial map -- e.g. (frames, 6, 4, 512) -- because
    that's the trunk's own graph; flattening to n_embeddings only happens in
    the embedder's *numpy* embed() path used at extraction time, and the head
    was trained on that flat vector. Checked and fixed here by actually
    running the graph on a throwaway input rather than by trusting any one
    framework's static shape, since to_onnx() may come from a framework (or a
    hand-built graph, per embedders/yamnet_aves) that doesn't expose one.
    """
    import onnx
    import onnxruntime as ort

    session = ort.InferenceSession(trunk_onnx.SerializeToString(),
                                   providers=['CPUExecutionProvider'])
    dummy = np.zeros(int(embedder.framelength_s * embedder.samplerate) * 3,
                     dtype=np.float32)
    name_in = session.get_inputs()[0].name
    out = session.run(None, {name_in: dummy})[0]

    if out.ndim > 2:
        flat_dim = int(np.prod(out.shape[1:]))
        output_name = trunk_onnx.graph.output[0].name
        flat_name = output_name + '_flattened'
        shape_init = onnx.helper.make_tensor(
            f'{output_name}_flatten_shape', onnx.TensorProto.INT64, [2], [0, -1])
        trunk_onnx.graph.initializer.append(shape_init)
        trunk_onnx.graph.node.append(onnx.helper.make_node(
            'Reshape', [output_name, shape_init.name], [flat_name],
            name='export_flatten_trunk'))
        trunk_onnx.graph.output[0].name = flat_name
        del trunk_onnx.graph.output[0].type.tensor_type.shape.dim[:]
        print(f'  trunk output {out.shape[1:]} flattened to ({flat_dim},)')
    else:
        flat_dim = out.shape[-1]

    if flat_dim != embedder.n_embeddings:
        raise SystemExit(
            f"embedder '{embedder.embeddername}' trunk output flattens to "
            f'{flat_dim}, but n_embeddings is {embedder.n_embeddings}')
    return trunk_onnx


def head_to_onnx(head, n_embeddings):
    """The trained classifier alone, as its own ONNX graph: embeddings in,
    predictions out. Wrapping it in a throwaway keras.Model first is what
    makes this work for both a model.keras head (already a keras.Model, so
    this is a no-op wrapper) and the legacy SavedModel/TFSMLayer head (which
    has no .export() of its own but can still be called inside one)."""
    import keras
    import onnx
    import tempfile

    inp = keras.Input((n_embeddings,), dtype='float32', name='embeddings')
    out = head(inp)
    # A SavedModel hands its outputs back in a dict keyed by layer name.
    if isinstance(out, dict):
        (out,) = out.values()
    head_model = keras.Model(inp, out, name='head')
    head_model(np.zeros((1, n_embeddings), dtype=np.float32))

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'head.onnx')
        head_model.export(path, format='onnx', verbose=False, opset_version=OPSET)
        return onnx.load(path)


def merge_trunk_head(trunk_onnx, head_onnx):
    """One graph: the trunk's embeddings feed straight into the head.

    onnx.compose does the actual wiring and the name-collision avoidance
    (prefix1/prefix2) -- this just satisfies its precondition that both
    graphs agree on IR version and opset, which they do by construction here
    (both exported at OPSET) but might not for a trunk graph that was built
    or cached some other way, so it's checked rather than assumed.
    """
    from onnx import compose

    if trunk_onnx.ir_version != head_onnx.ir_version:
        target = max(trunk_onnx.ir_version, head_onnx.ir_version)
        trunk_onnx.ir_version = target
        head_onnx.ir_version = target

    trunk_ops = {i.domain: i.version for i in trunk_onnx.opset_import}
    head_ops = {i.domain: i.version for i in head_onnx.opset_import}
    if trunk_ops != head_ops:
        raise SystemExit(
            f'trunk and head were exported at different opsets ({trunk_ops} vs '
            f'{head_ops}); export both at the same opset before merging.')

    trunk_out = trunk_onnx.graph.output[0].name
    head_in = head_onnx.graph.input[0].name
    return compose.merge_models(
        trunk_onnx, head_onnx, io_map=[(trunk_out, head_in)],
        prefix1='trunk_', prefix2='head_')


def rename_io(model):
    """Give the graph's input, output and symbolic length stable names."""
    graph = model.graph
    old_in, old_out = graph.input[0].name, graph.output[0].name
    for node in graph.node:
        node.input[:] = [NAME_IN if i == old_in else i for i in node.input]
        node.output[:] = [NAME_OUT if o == old_out else o for o in node.output]
    graph.input[0].name = NAME_IN
    graph.output[0].name = NAME_OUT
    dim = graph.input[0].type.tensor_type.shape.dim[0]
    if dim.HasField('dim_param'):
        dim.dim_param = DIM_SAMPLES
    return model


def export_graph(model, path_onnx):
    """The merged trunk+head graph -> folded and fused, written to path_onnx."""
    import onnx

    print(f'exporting {path_onnx}')
    n_before = len(model.graph.node)
    model, n_folded, n_fused, n_dropped = optimize(model)
    model = rename_io(model)
    onnx.checker.check_model(model)
    onnx.save(model, path_onnx)

    n_conv = sum(1 for n in model.graph.node if n.op_type in ('Conv', 'FusedConv'))
    print(f'  {n_before} -> {len(model.graph.node)} nodes: '
          f'{n_folded} batchnorms folded into their convolutions, '
          f'{n_fused} of {n_conv} convolutions fused with their Relu, '
          f'{n_dropped} orphaned initializers dropped')

    # A pair the pass should have taken and did not is a bug in the pass.
    n_left = count_fusable(model)
    if n_left:
        raise SystemExit(f'{n_left} Conv+Relu pairs were left unfused; '
                         f'tools/onnx_passes.py did not do its job.')
    # Nothing to fuse is a different thing, and not necessarily wrong: a
    # backbone using Relu6 or HardSwish has no plain Conv->Relu, and neither
    # does one whose batchnorms never folded. Worth saying out loud, because on
    # YAMNet it would mean the graph that ships is not the graph that was
    # measured -- but not worth refusing an export over, since the parity
    # checks below are what decide whether the model is right.
    if n_fused == 0 and n_conv:
        print(f'  WARNING: no Conv+Relu pairs matched. If this model is meant '
              f'to fuse, the export is slower than it should be on CUDA; check '
              f'what activation the backbone uses.')
    return model


def verify(path_onnx, embedder, head, path_audio):
    """Run the ONNX graph against embed() and the trained head it was built from.

    embed() -- not a fused Keras model -- is the ground truth here because
    it's the one thing every embedder already has and already agrees with its
    own to_onnx(): whatever framework or combination of frameworks a trunk
    uses, embed() is its numpy-in/numpy-out reference implementation. This is
    also what makes the check meaningful for a multi-framework trunk like
    yamnet_aves, which has no single fused model to compare against.

    Real audio first, then synthetic lengths chosen to cover the ragged cases:
    several whole frames, exactly one, one sample under the framing floor, a
    ragged tail, and a clip too short to make a frame at all. Those last three
    exercise the front end's framing rule (padding or truncating), which is
    inside the graph now.
    """
    import librosa
    import onnxruntime as ort

    session = ort.InferenceSession(path_onnx, providers=['CPUExecutionProvider'])
    name_in = session.get_inputs()[0].name
    print(f'  {os.path.getsize(path_onnx) / 1e6:.2f} MB, '
          f'in {session.get_inputs()[0].shape} out {session.get_outputs()[0].shape}')

    def reference(samples):
        embeddings = np.asarray(embedder.embed(samples), dtype=np.float32)
        predictions = head(embeddings)
        if isinstance(predictions, dict):
            (predictions,) = predictions.values()
        return np.asarray(predictions)

    cases = []
    if path_audio is not None:
        samples, _ = librosa.load(path_audio, sr=embedder.samplerate, mono=True)
        cases.append((os.path.basename(path_audio), samples.astype(np.float32)))

    rng = np.random.default_rng(0)
    n_frame = int(embedder.framelength_s * embedder.samplerate)
    for n in (n_frame * 8, n_frame, n_frame - 1, n_frame * 3 + 137, n_frame // 4):
        cases.append((f'noise n={n}',
                      (rng.standard_normal(n) * 0.1).astype(np.float32)))

    worst = 0.0
    for label, samples in cases:
        expected = reference(samples)
        got = session.run(None, {name_in: samples})[0]
        if expected.shape != got.shape:
            raise SystemExit(f'{label}: shape mismatch, keras {expected.shape} '
                             f'vs onnx {got.shape}')
        d = float(np.abs(expected - got).max()) if got.size else 0.0
        agree = ((expected.argmax(1) == got.argmax(1)).mean()
                 if got.size else float('nan'))
        worst = max(worst, d)
        print(f'  {label:<34} {str(got.shape):<12} max|d|={d:.2e}  agree={agree:.4f}')

    if worst > TOL:
        raise SystemExit(f'parity FAILED: {worst:.2e} > {TOL}; nothing shipped')
    print(f'parity OK: {worst:.2e}')
    return worst


def frames_for(n_samples, samples_hop, samples_min, floor_nonzero, float32_quotient):
    """How many frames the graph returns for n_samples of audio.

    Three things here are not what they look like.

    Below floor_nonzero, the answer is 0, not 1 -- for a front end that pads a
    short clip up to a frame (YAMNet: the patch window plus the STFT window's
    overhang) floor_nonzero is 1 and this never bites; for one that only ever
    returns whole frames (embedders/yamnet_aves's embed(), which truncates a
    ragged tail rather than padding it) floor_nonzero is samples_hop, and a
    clip shorter than one frame really does yield nothing.

    Above that, the first frame needs samples_min samples rather than exactly
    samples_hop -- for a padding front end that's the padding floor described
    above; for a truncating one it's the largest input that still rounds down
    to one whole frame -- so this is not ceil(n_samples / samples_hop) either.

    And the hop division past that floor is not always an honest integer
    division. tf2onnx emits YAMNet's patch framing as a float32 multiply by
    the reciprocal of the hop, and 1/15360 is not exact in float32, so at some
    exact multiples of the hop the quotient lands just above the integer and
    the ceil returns one more frame than real arithmetic would (61680 samples
    and 3210480 samples are two of them). A hand-built bridge doesn't have to
    share that quirk -- embedders/yamnet_aves's front end does the division in
    int64, which is exact -- so which one applies is a property of how a given
    embedder's to_onnx() computes it, not a constant. probe_framing() decides
    which by testing both against the graph, rather than assuming either.
    """
    if n_samples < floor_nonzero:
        return 0
    after = max(0, n_samples - samples_min)
    if float32_quotient:
        q = np.ceil(np.float32(after) * (np.float32(1.0) / np.float32(samples_hop)))
        return 1 + int(q)
    return 1 + -(-after // samples_hop)  # exact ceiling division, no float roundoff


def probe_framing(path_onnx, embedder):
    """Find the graph's framing rule by asking it, rather than by assuming it.

    samples_hop follows from the frame length, but whether -- and how far --
    the front end pads a short clip up to a frame, versus just returning
    nothing, is a property of that front end, not something this file or the
    engine should be expected to know per embedder. Binary-searching for it
    costs a handful of runs on inputs under two frames long, and the result is
    checked against the graph at awkward lengths before it is shipped.
    """
    import time

    import onnxruntime as ort

    session = ort.InferenceSession(path_onnx, providers=['CPUExecutionProvider'])
    call_seconds = []

    def n_frames(n):
        x = np.zeros(n, dtype=np.float32)
        t0 = time.perf_counter()
        out = session.run(None, {NAME_IN: x})[0].shape[0]
        call_seconds.append(time.perf_counter() - t0)
        return out

    samples_hop = int(round(embedder.framelength_s * embedder.samplerate))
    hi_check = 4 * samples_hop
    if n_frames(hi_check) < 2:
        raise SystemExit(f'{hi_check} samples still gives one frame; framing is not '
                         f'what this assumes')

    # The smallest input that yields at least one frame: 1 for a front end
    # that pads, samples_hop for one that only ever returns whole frames.
    if n_frames(1) >= 1:
        floor_nonzero = 1
    else:
        lo, hi = 1, hi_check
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if n_frames(mid) >= 1:
                hi = mid
            else:
                lo = mid
        floor_nonzero = hi

    # The largest input that still yields exactly one frame.
    lo, hi = floor_nonzero, hi_check
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if n_frames(mid) == 1:
            lo = mid
        else:
            hi = mid
    samples_min = lo

    # Every exact multiple of the hop, either side of it, plus a few ragged
    # lengths and the floor itself. The multiples are the ones that matter:
    # that is where a float32 reciprocal would disagree with exact integer
    # arithmetic, and checking a handful of round numbers would miss it.
    #
    # How many multiples to check is scaled to how expensive one graph call
    # actually is, rather than fixed at the 220 that suits a small conv net
    # like YAMNet's: an embedder whose trunk includes a full transformer (see
    # embedders/yamnet_aves) can be three orders of magnitude slower per call
    # on CPU, and 220 * 3 calls of that would turn a export into a
    # multi-minute wait for a check that's only ever caught one specific
    # class of bug. The budget below keeps this step to roughly ten seconds
    # regardless of how heavy the trunk is, while still checking dozens of
    # multiples -- plenty to catch a systematic rounding quirk, which shows
    # up at every multiple it affects, not one in a thousand.
    avg_call_s = (sum(call_seconds) / len(call_seconds)) if call_seconds else 0.0
    budget_s = 10.0
    n_multiples = max(5, min(220, int(budget_s / (3 * max(avg_call_s, 1e-6)))))
    if n_multiples < 220:
        print(f'  trunk call ~{avg_call_s * 1e3:.1f} ms; checking {n_multiples} '
              f'hop multiples instead of 220 to keep this under {budget_s:g}s')

    checks = [floor_nonzero - 1, floor_nonzero, samples_min - 1, samples_min,
             samples_min + 1]
    for m in range(0, n_multiples):
        base = samples_min + samples_hop * m
        checks += [base - 1, base, base + 1]
    checks += [samples_hop * 7 + 137, samples_hop * 40 + 1]
    checks = sorted({n for n in checks if n >= 1})

    # Which arithmetic the graph actually uses is decided empirically -- try
    # exact integer division first (it's what most hand-built bridges will
    # use), fall back to the float32-reciprocal quirk (what tf2onnx emits for
    # YAMNet's patch framing), and refuse to ship if neither matches
    # everywhere, since that means something else is wrong.
    graph_counts = {n: n_frames(n) for n in checks}
    float32_quotient = None
    for candidate in (False, True):
        if all(frames_for(n, samples_hop, samples_min, floor_nonzero, candidate)
              == graph_counts[n] for n in checks):
            float32_quotient = candidate
            break
    if float32_quotient is None:
        n = next(n for n in checks if frames_for(
            n, samples_hop, samples_min, floor_nonzero, False) != graph_counts[n])
        raise SystemExit(f'framing rule wrong at n={n}: predicted '
                         f'{frames_for(n, samples_hop, samples_min, floor_nonzero, False)} '
                         f'frames, graph returned {graph_counts[n]}')

    print(f'framing rule checked at {len(checks)} lengths '
          f'({"float32-reciprocal" if float32_quotient else "exact integer"} division)')
    if floor_nonzero == 1:
        print(f'framing: hop {samples_hop} samples, first frame needs '
              f'{samples_min} (pads a short clip up to one frame)')
    else:
        print(f'framing: hop {samples_hop} samples, first frame needs '
              f'{floor_nonzero} (no padding: a shorter clip returns zero frames)')
    return samples_hop, samples_min, floor_nonzero, float32_quotient


def verify_fixed_length(path_onnx, embedder, samples_hop, samples_min, floor_nonzero,
                        float32_quotient, seconds=200):
    """Check the graph still agrees with itself once its input length is pinned.

    This is how buzzdetect runs it. CoreML's MLProgram format cannot compile a
    graph with an unbounded dimension at all, so the engine fixes the waveform
    length before creating the session and zero-pads short chunks up to it
    (see COREML.md). That makes the padding contract part of what ships, so it
    is checked here rather than only over there: a chunk padded up to the fixed
    length, then truncated to the frames the real audio covers, must give what
    the unpadded graph gives.
    """
    import onnx
    import onnxruntime as ort
    from onnxruntime.tools.onnx_model_utils import fix_output_shapes, make_dim_param_fixed

    n_fixed = int(seconds * embedder.samplerate)
    model = onnx.load(path_onnx)
    make_dim_param_fixed(model.graph, DIM_SAMPLES, n_fixed)
    fix_output_shapes(model)

    dynamic = ort.InferenceSession(path_onnx, providers=['CPUExecutionProvider'])
    fixed = ort.InferenceSession(model.SerializeToString(),
                                 providers=['CPUExecutionProvider'])

    rng = np.random.default_rng(1)
    worst = 0.0
    for n in (n_fixed, samples_hop * 7, samples_hop * 7 + 137, samples_min - 1):
        samples = (rng.standard_normal(n) * 0.1).astype(np.float32)
        padded = np.zeros(n_fixed, dtype=np.float32)
        padded[:n] = samples
        n_expected = frames_for(n, samples_hop, samples_min, floor_nonzero, float32_quotient)
        got = fixed.run(None, {NAME_IN: padded})[0][:n_expected]
        expected = dynamic.run(None, {NAME_IN: samples})[0]
        if expected.shape != got.shape:
            raise SystemExit(f'padded n={n}: {got.shape} against '
                             f'{expected.shape} unpadded')
        worst = max(worst, float(np.abs(expected - got).max()))
    print(f'fixed-length parity OK at {seconds:g}s: {worst:.2e}')
    return n_fixed


def write_fp16(path_onnx, path_fp16, samples):
    """Convert the trunk and head to fp16, leaving the front end -- and every
    explicit Cast node, wherever it sits -- alone.

    Two reasons the front end stays in fp32. It is where the dynamic range is
    -- a log of a mel spectrogram, before any normalisation -- and it is
    cheap, so converting it would buy little. "Everything from the first
    convolution on" is the rule for what to convert; it is a rule about this
    shape of model -- a signal front end followed by a convolutional trunk --
    rather than about YAMNet specifically.

    Every Cast node is excluded for a different, narrower reason: the
    converter mistypes them, producing a node whose declared output type
    doesn't match what it actually writes, and a graph onnxruntime refuses to
    load. First seen in YAMNet's framing code (hence catching it by cutting
    the front end off at the first Conv), then again inside embedders/aves's
    attention blocks -- unrelated code, same converter bug -- which is why
    this excludes Cast nodes everywhere rather than by position. A Cast's
    output dtype is already explicit in its `to` attribute; converting it is
    also the one case where leaving a node in fp32 changes nothing about
    memory or speed, since it was never doing arithmetic.
    """
    import onnx
    import onnxruntime as ort
    from onnxconverter_common import float16

    model = onnx.load(path_onnx)
    nodes = list(model.graph.node)
    first_conv = next((i for i, n in enumerate(nodes)
                       if n.op_type in ('Conv', 'FusedConv')), None)
    if first_conv is None:
        raise SystemExit('no convolution in the graph; fp16 conversion has no '
                         'sensible boundary to stop at')
    frontend = {n.name for n in nodes[:first_conv]}
    casts = {n.name for n in nodes if n.op_type == 'Cast'}
    node_block_list = sorted(frontend | casts)

    converted = float16.convert_float_to_float16(
        model, keep_io_types=True, node_block_list=node_block_list)
    onnx.save(converted, path_fp16)

    # It loads, it runs, and it still resembles the model it came from. Run
    # both sessions on their native dynamic shape rather than pinning
    # DIM_SAMPLES via SessionOptions.add_free_dimension_override_by_name --
    # verify_fixed_length() already checked that fixing the shape doesn't
    # change the fp32 graph's output, so this doesn't need to repeat that at
    # fp16. It emphatically should not repeat it: on a trunk with attention
    # layers (embedders/yamnet_aves), pinning the shape on a converted fp16
    # session measured 5+ minutes for a forward pass that takes 70s on the
    # same session left dynamic -- some onnxruntime interaction between a
    # fixed batch dimension and fp16 attention ops, triggered by either
    # add_free_dimension_override_by_name or the graph-level
    # make_dim_param_fixed used elsewhere in this file. Root cause not
    # chased further since the fix is the same either way: don't fix the
    # shape here, there is no need to.
    rng = np.random.default_rng(2)
    x = (rng.standard_normal(samples) * 0.1).astype(np.float32)
    reference = ort.InferenceSession(path_onnx, providers=['CPUExecutionProvider'])
    session = ort.InferenceSession(path_fp16, providers=['CPUExecutionProvider'])
    expected = reference.run(None, {NAME_IN: x})[0]
    got = session.run(None, {NAME_IN: x})[0]
    d = float(np.abs(expected - got).max())
    agree = float((expected.argmax(1) == got.argmax(1)).mean())
    print(f'  {os.path.getsize(path_fp16) / 1e6:.2f} MB, '
          f'{len(node_block_list)} of {len(nodes)} nodes left in fp32 '
          f'({len(frontend)} front end, {len(casts - frontend)} Cast), '
          f'max|d|={d:.2e}, top-class agreement={agree:.4f}')
    # Both bounds are loose on purpose. This is not a parity check -- reduced
    # precision is a deliberate trade the operator opts into -- it is a check
    # that the conversion produced the model rather than mush, and in
    # particular that the fp32/fp16 boundary landed somewhere sensible. A model
    # whose feature extractor is itself convolutional would put that boundary
    # in the wrong place, and this is what would say so.
    if d > TOL_FP16 or agree < AGREE_FP16:
        raise SystemExit(
            f'fp16 conversion looks wrong: max|d|={d:.2e} (limit {TOL_FP16}), '
            f'top-class agreement={agree:.4f} (limit {AGREE_FP16}). The fp32 '
            f'prefix is {len(frontend)} nodes; if this model\'s front end is '
            f'itself convolutional, that boundary is in the wrong place.')


def export(modelname, dir_dest, force=False, path_audio=None,
           dir_src=None, embeddername=None):
    """Build the export in a staging dir, check it, and only then move it into place.

    Staging is what makes a failed check harmless: nothing lands in the
    destination -- and no existing export there is disturbed -- unless every
    check passes.

    The staged files are then copied over the destination rather than replacing
    the directory. A model directory is not only the export: it also holds the
    README, the test plots, the training history and the weights table, none of
    which this tool produces and all of which replacing the directory would
    delete.
    """
    dir_src = dir_src or os.path.join(cfg.DIR_MODELS, modelname)
    if not os.path.isdir(dir_src):
        raise SystemExit(f'no such model: {dir_src}')

    with open(os.path.join(dir_src, 'config_model.json')) as f:
        config = json.load(f)
    embeddername = embeddername or config['embeddername']

    dir_out = os.path.join(dir_dest, modelname)
    if os.path.exists(dir_out) and not force:
        raise SystemExit(f'{dir_out} already exists; pass --force to overwrite')

    dir_staging = tempfile.mkdtemp(prefix='export_onnx_')
    dir_stage = os.path.join(dir_staging, modelname)
    os.makedirs(dir_stage)

    print(f"building {modelname} from {dir_src} on embedder '{embeddername}'")
    trunk_onnx, embedder = build_trunk_onnx(embeddername)
    head = load_head(dir_src, embedder.n_embeddings)
    head_onnx = head_to_onnx(head, embedder.n_embeddings)
    model = merge_trunk_head(trunk_onnx, head_onnx)

    path_onnx = os.path.join(dir_stage, 'model.onnx')
    export_graph(model, path_onnx)

    print('checking the graph against embed() and the head it came from')
    verify(path_onnx, embedder, head, path_audio)
    samples_hop, samples_min, floor_nonzero, float32_quotient = probe_framing(
        path_onnx, embedder)
    n_session = verify_fixed_length(path_onnx, embedder, samples_hop, samples_min,
                                    floor_nonzero, float32_quotient)

    print('writing the reduced-precision sibling')
    write_fp16(path_onnx, os.path.join(dir_stage, FNAME_FP16), n_session)

    config_out = {
        'classes': config['classes'],
        'samplerate': embedder.samplerate,
        'framelength_s': embedder.framelength_s,
        'digits_time': embedder.digits_time,
        'digits_results': config['digits_results'],
        'samples_hop': samples_hop,
        'samples_min': samples_min,
        # New key, from probe_framing()'s generalisation for a front end that
        # truncates a ragged tail instead of padding it (embedders/yamnet_aves
        # is the first of these): samples_min alone no longer says what a
        # chunk shorter than one frame returns. samples_min == floor_nonzero
        # (== 1) reproduces every existing model's engine-side behaviour
        # exactly, so this is additive -- but the engine's chunking logic
        # (src/inference/models.py) needs to read it once a model ships with
        # floor_nonzero > 1, or it will assume a padded single frame where
        # the graph actually returns none.
        'samples_floor_nonzero': floor_nonzero,
    }
    with open(os.path.join(dir_stage, 'config_model.json'), 'w') as f:
        json.dump(config_out, f, indent=2)

    os.makedirs(dir_out, exist_ok=True)
    written = sorted(os.listdir(dir_stage))
    for name in written:
        shutil.copy2(os.path.join(dir_stage, name), os.path.join(dir_out, name))
    shutil.rmtree(dir_staging, ignore_errors=True)
    print(f'wrote {dir_out}: {", ".join(written)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('modelname', help='model directory name under models/')
    parser.add_argument('--dest', default=DEST_DEFAULT,
                        help='engine model directory (default: buzzdetect_dest '
                             f'in paths.local.json, currently {DEST_DEFAULT})')
    parser.add_argument('--force', action='store_true',
                        help='overwrite an existing export')
    parser.add_argument('--from', dest='dir_src', default=None, metavar='DIR',
                        help='read the trained weights and config_model.json '
                             'from DIR instead of models/<modelname>. For a '
                             'model that lives only in buzzdetect, released '
                             'before this repo held it')
    parser.add_argument('--embedder', default=None, metavar='NAME',
                        help="override the embedder named in config_model.json, "
                             "for a config that names one of buzzdetect's "
                             "(yamnet_k2 is this repo's yamnet)")
    parser.add_argument('--verify-audio', default=FIXTURE_AUDIO, metavar='PATH',
                        help='audio to run the parity check on '
                             '(default: the bundled fixture)')
    parser.add_argument('--no-verify-audio', dest='verify_audio',
                        action='store_const', const=None,
                        help='check on synthetic lengths only')
    args = parser.parse_args()

    if args.dest is None:
        raise SystemExit('no destination: pass --dest, or set "buzzdetect_dest" '
                          'in paths.local.json (see paths.local.example.json)')
    if not os.path.isdir(args.dest):
        raise SystemExit(f'destination does not exist: {args.dest}')
    # Checked up front: the export is a slow way to discover a typo.
    if args.verify_audio is not None and not os.path.isfile(args.verify_audio):
        raise SystemExit(f'no such audio: {args.verify_audio}')

    export(args.modelname, args.dest, args.force, args.verify_audio,
           args.dir_src, args.embedder)


if __name__ == '__main__':
    main()
