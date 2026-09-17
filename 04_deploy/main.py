"""
Deploy a trained model: refresh its threshold/README card, then export it to
buzzdetect as ONNX.

    conda run -n buzzdetect-train python 04_deploy/main.py cv_baseline

Chains 04_deploy/model_card.py (thresholds + README, from the CV rotations'
predictions) and 04_deploy/export_onnx.py (the fused ONNX graph, carried over
to buzzdetect) the same way root main.py chains stages 2-3. Run either step on
its own -- `python 04_deploy/model_card.py <name>...` or
`python 04_deploy/export_onnx.py <name> --dest ...` -- when only one is needed.
"""
# TensorFlow must be imported before pandas/pyarrow; see root main.py's header
# comment for why. model_card.py's write_model_card() pulls in pandas via
# 03_train/thresholds.py, so it's loaded here only after the import below, not
# at module scope.

import argparse
import importlib.util
import os
import sys


def load_module(path, module_name):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    mod_dir = os.path.dirname(os.path.abspath(path))
    for d in (root_dir, mod_dir):
        if d not in sys.path:
            sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(modelname, dir_dest, force=False, path_audio=None, dir_src=None,
         embeddername=None, skip_card=False):
    import tensorflow  # noqa: F401  -- load-order side effect; see header comment

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    sys.path.insert(0, root)
    import config as cfg

    if not skip_card:
        print('=== model card (thresholds + README) ===')
        model_card = load_module(os.path.join(here, 'model_card.py'), 'deploy_model_card')
        dir_model = dir_src or os.path.join(cfg.DIR_MODELS, modelname)
        if os.path.exists(os.path.join(dir_model, 'config_model.json')):
            model_card.write_model_card(dir_model, modelname)
        else:
            print(f'no config_model.json in {dir_model}; skipping card, exporting as-is')

    print('\n=== export to buzzdetect (ONNX) ===')
    export_onnx = load_module(os.path.join(here, 'export_onnx.py'), 'deploy_export_onnx')
    export_onnx.export(modelname, dir_dest, force, path_audio, dir_src, embeddername)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('modelname', help='model directory name under models/')
    parser.add_argument('--dest', default=None,
                        help='engine model directory (default: buzzdetect_dest '
                             'in paths.local.json)')
    parser.add_argument('--force', action='store_true',
                        help='overwrite an existing export')
    parser.add_argument('--from', dest='dir_src', default=None, metavar='DIR',
                        help='read the trained weights and config_model.json '
                             'from DIR instead of models/<modelname>')
    parser.add_argument('--embedder', default=None, metavar='NAME',
                        help="override the embedder named in config_model.json")
    parser.add_argument('--verify-audio', default=None, metavar='PATH',
                        help='audio to run the export parity check on '
                             '(default: the bundled fixture)')
    parser.add_argument('--no-verify-audio', dest='no_verify_audio', action='store_true',
                        help='check on synthetic lengths only')
    parser.add_argument('--skip-card', action='store_true',
                        help='export as-is; do not refresh thresholds/README first')
    args = parser.parse_args()

    dest = args.dest
    if dest is None:
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.dirname(here))
        import config as cfg
        dest = cfg.local('buzzdetect_dest')
    if dest is None:
        raise SystemExit('no destination: pass --dest, or set "buzzdetect_dest" '
                          'in paths.local.json (see paths.local.example.json)')
    if not os.path.isdir(dest):
        raise SystemExit(f'destination does not exist: {dest}')

    path_audio = args.verify_audio
    if args.no_verify_audio:
        path_audio = None
    elif path_audio is None:
        path_audio = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'fixtures', '230808_1208_s89520.flac')
    if path_audio is not None and not os.path.isfile(path_audio):
        raise SystemExit(f'no such audio: {path_audio}')

    main(
        modelname=args.modelname,
        dir_dest=dest,
        force=args.force,
        path_audio=path_audio,
        dir_src=args.dir_src,
        embeddername=args.embedder,
        skip_card=args.skip_card,
    )
