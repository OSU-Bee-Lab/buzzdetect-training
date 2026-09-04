"""Export a trained model into buzzdetect's engine/models/.

    conda run -n buzzdetect-train python tools/export_onnx.py yamnet_medium_general

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


def build_combined(modelname, dir_src, embeddername):
    """One Keras model, waveform -> predictions.

    The embedder is loaded through its own plugin rather than by reaching for
    a .keras path, so whatever that plugin does at load time -- retuning the
    patch hop, in yamnet's case -- is done here too, and a new embedder needs
    no change in this file.
    """
    import keras

    from embedders.embedding import load_embedder

    # framehop_prop=1: the graph is exported with the patch hop welded to the
    # patch window. An overlapping framehop would have to be a graph parameter,
    # and buzzdetect's ONNX path has never supported one.
    embedder = load_embedder(embeddername, framehop_prop=1, initialize=True)
    trunk = embedder.model
    if not isinstance(trunk, keras.Model):
        raise SystemExit(
            f"embedder '{embeddername}' does not load a Keras model "
            f'({type(trunk).__name__}), so it cannot be exported to ONNX here.')

    head = load_head(dir_src, embedder.n_embeddings)
    predictions = head(trunk.output)
    # A SavedModel hands its outputs back in a dict keyed by layer name.
    if isinstance(predictions, dict):
        (predictions,) = predictions.values()

    combined = keras.Model(trunk.input, predictions, name=modelname)
    combined(np.zeros(int(embedder.framelength_s * embedder.samplerate) * 3,
                      dtype=np.float32))
    return combined, embedder, head


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


def export_graph(combined, path_onnx):
    """Keras -> ONNX -> folded and fused, written to path_onnx."""
    import onnx

    print(f'exporting {path_onnx}')
    combined.export(path_onnx, format='onnx', verbose=False)

    model = onnx.load(path_onnx)
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


def verify(path_onnx, combined, embedder, path_audio):
    """Run the ONNX graph against the Keras model it was built from.

    Real audio first, then synthetic lengths chosen to cover the ragged cases:
    several whole frames, exactly one, one sample under the padding floor, a
    ragged tail, and a clip too short to make a frame at all. Those last three
    exercise the front end's padding, which is inside the graph now.
    """
    import librosa
    import onnxruntime as ort

    session = ort.InferenceSession(path_onnx, providers=['CPUExecutionProvider'])
    name_in = session.get_inputs()[0].name
    print(f'  {os.path.getsize(path_onnx) / 1e6:.2f} MB, '
          f'in {session.get_inputs()[0].shape} out {session.get_outputs()[0].shape}')

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
        expected = np.asarray(combined(samples))
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


def frames_for(n_samples, samples_hop, samples_min):
    """How many frames the graph returns for n_samples of audio.

    Two things here are not what they look like.

    The first frame needs more samples than the hop -- the front end pads up to
    a whole patch plus the STFT window's overhang -- so this is not
    ceil(n_samples / samples_hop).

    And the hop division is a float32 multiply by the reciprocal of the hop,
    not a division. tf2onnx emits it that way, and it matters: 1/15360 is not
    exact in float32, so at some exact multiples of the hop the quotient lands
    just above the integer and the ceil returns one more frame than real
    arithmetic would. Doing it in float64, or as an honest division, is wrong
    by one frame at those lengths -- 61680 samples and 3210480 samples are two
    of them. probe_framing() checks this against the graph before shipping.
    """
    if n_samples <= 0:
        return 0
    after = np.float32(max(0, n_samples - samples_min))
    return 1 + int(np.ceil(after * (np.float32(1.0) / np.float32(samples_hop))))


def probe_framing(path_onnx, embedder):
    """Find the graph's framing rule by asking it, rather than by assuming it.

    samples_hop follows from the frame length, but the floor below which the
    front end pads up to a single frame is a property of that front end -- for
    YAMNet it is the patch window plus the STFT window's overhang, which is not
    something the engine should be expected to know. Binary-searching for it
    costs a handful of runs on inputs under two frames long, and the result is
    checked against the graph at awkward lengths before it is shipped.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(path_onnx, providers=['CPUExecutionProvider'])

    def n_frames(n):
        x = np.zeros(n, dtype=np.float32)
        return session.run(None, {NAME_IN: x})[0].shape[0]

    samples_hop = int(round(embedder.framelength_s * embedder.samplerate))
    # The largest input that still yields exactly one frame.
    lo, hi = 1, 4 * samples_hop
    if n_frames(hi) < 2:
        raise SystemExit(f'{hi} samples still gives one frame; framing is not '
                         f'what this assumes')
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if n_frames(mid) == 1:
            lo = mid
        else:
            hi = mid
    samples_min = lo

    # Every exact multiple of the hop, either side of it, plus a few ragged
    # lengths. The multiples are the ones that matter: that is where the
    # float32 reciprocal in frames_for() disagrees with real arithmetic, and
    # checking a handful of round numbers would miss it.
    checks = [1, samples_min - 1, samples_min, samples_min + 1]
    for m in range(0, 220):
        base = samples_min + samples_hop * m
        checks += [base - 1, base, base + 1]
    checks += [samples_hop * 7 + 137, samples_hop * 40 + 1]
    for n in sorted({n for n in checks if n >= 1}):
        expected = frames_for(n, samples_hop, samples_min)
        got = n_frames(n)
        if expected != got:
            raise SystemExit(f'framing rule wrong at n={n}: predicted {expected} '
                             f'frames, graph returned {got}')
    print(f'framing rule checked at {len(set(checks))} lengths')
    print(f'framing: hop {samples_hop} samples, first frame needs {samples_min}')
    return samples_hop, samples_min


def verify_fixed_length(path_onnx, embedder, samples_hop, samples_min, seconds=200):
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
        got = fixed.run(None, {NAME_IN: padded})[0][:frames_for(n, samples_hop, samples_min)]
        expected = dynamic.run(None, {NAME_IN: samples})[0]
        if expected.shape != got.shape:
            raise SystemExit(f'padded n={n}: {got.shape} against '
                             f'{expected.shape} unpadded')
        worst = max(worst, float(np.abs(expected - got).max()))
    print(f'fixed-length parity OK at {seconds:g}s: {worst:.2e}')
    return n_fixed


def write_fp16(path_onnx, path_fp16, samples):
    """Convert the trunk and head to fp16, leaving the front end alone.

    Two reasons the front end stays in fp32. It is where the dynamic range is
    -- a log of a mel spectrogram, before any normalisation -- and it is cheap,
    so converting it would buy little. It also breaks the converter, which
    mistypes an explicit Cast in the framing code and produces a graph
    onnxruntime refuses to load.

    "Everything from the first convolution on" is the rule for what to convert.
    It is a rule about this shape of model -- a signal front end followed by a
    convolutional trunk -- rather than about YAMNet specifically.
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
    frontend = [n.name for n in nodes[:first_conv]]

    converted = float16.convert_float_to_float16(
        model, keep_io_types=True, node_block_list=frontend)
    onnx.save(converted, path_fp16)

    # It loads, it runs, and it still resembles the model it came from. The
    # fixed length is what the engine will use, so check it at that shape.
    rng = np.random.default_rng(2)
    x = (rng.standard_normal(samples) * 0.1).astype(np.float32)
    so32 = ort.SessionOptions()
    so32.add_free_dimension_override_by_name(DIM_SAMPLES, samples)
    reference = ort.InferenceSession(path_onnx, so32,
                                     providers=['CPUExecutionProvider'])
    so16 = ort.SessionOptions()
    so16.add_free_dimension_override_by_name(DIM_SAMPLES, samples)
    session = ort.InferenceSession(path_fp16, so16,
                                   providers=['CPUExecutionProvider'])
    expected = reference.run(None, {NAME_IN: x})[0]
    got = session.run(None, {NAME_IN: x})[0]
    d = float(np.abs(expected - got).max())
    agree = float((expected.argmax(1) == got.argmax(1)).mean())
    print(f'  {os.path.getsize(path_fp16) / 1e6:.2f} MB, '
          f'{len(frontend)} of {len(nodes)} nodes left in fp32, '
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
    combined, embedder, _ = build_combined(modelname, dir_src, embeddername)

    path_onnx = os.path.join(dir_stage, 'model.onnx')
    export_graph(combined, path_onnx)

    print('checking the graph against the keras model it came from')
    verify(path_onnx, combined, embedder, path_audio)
    samples_hop, samples_min = probe_framing(path_onnx, embedder)
    n_session = verify_fixed_length(path_onnx, embedder, samples_hop, samples_min)

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
