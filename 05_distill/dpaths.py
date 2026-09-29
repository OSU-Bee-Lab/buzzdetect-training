"""Every path and teacher-dependent constant of 05_distill, in one place.

The teacher comes from `--teacher` on main.py (which exports it), else the
`DISTILL_TEACHER` environment variable, else `distill_teacher` in paths.local.json,
else DEFAULT_TEACHER. Nothing else in 05_distill names a teacher.

Layout (all under paths that live outside the repo):

  <distill_cache>/                       paths.local.json key; the audio drive
      _shared/durations.csv              ffprobe durations, keyed (relpath, size): teacher-independent
      _mel/<spec>/<relpath>/h<hour>.npz  student-input spectrograms per front end: teacher-independent,
                                         shared by every teacher and every student architecture
      <teacher>/                         one directory per teacher
          _manifest/                     plan.csv, teacher.json, teacher_ext.onnx, blacklist.txt
          <relpath>/h<hour>.npz          targets: code, logits, start_s (older caches also embed mel)
          _shards/                       streaming shards (rungs C-D)
  <repo main checkout>/05_distill/data/     (gitignored, like 02_set's data; main checkout even from a worktree)
      _shared/arch/                      speed timings of random-weight candidates, frontend_only.onnx
      <teacher>/                         runs/, models/, eval/, shards/ (packed rungs), ladder.jsonl

`<teacher>` is the name of the model dir the teacher lives in: `models/<name>/` in this project
(config_model.json: classes, activation_centers, set, folds_train) and `<buzzdetect_dest>/<name>/model.onnx`.
Classes without an activation center sit above zero everywhere: they get no loss and no readout.

This module imports only the standard library and config.py, so the engine venv can use it.
"""
import functools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import config  # noqa: E402

MARKER = os.sep + '.claude' + os.sep + 'worktrees' + os.sep
MAIN = ROOT.split(MARKER)[0]

DEFAULT_TEACHER = 'v4-ft-ps-e60-moderate'
DEFAULT_BASELINE = 'cv-baseline-v4-moderate'


def _setting(env, key, default):
    return os.environ.get(env) or config.local(key) or default


TEACHER = _setting('DISTILL_TEACHER', 'distill_teacher', DEFAULT_TEACHER)
# the era baseline the ladder's headline is compared against (a CV model dir under models/)
BASELINE = _setting('DISTILL_BASELINE', 'distill_baseline', DEFAULT_BASELINE)

# ---- other repos / interpreters
ENGINE_MODELS = os.environ.get('DISTILL_ENGINE_MODELS') or config.local('buzzdetect_dest')   # <buzzdetect>/engine/models
ENGINE = os.path.dirname(ENGINE_MODELS) if ENGINE_MODELS else None
ENGINE_PY = os.path.join(ENGINE, '.venv', 'bin', 'python3') if ENGINE else None   # CUDA onnxruntime
ONNX_PY = config.local('distill_onnx_python') or os.path.join(MAIN, '.local', 'venv-onnx', 'bin', 'python')
AUDIO_ROOT = os.environ.get('DISTILL_AUDIO_ROOT') or config.AUDIO_ROOT      # the env override is for tests

# ---- teacher-independent cache level
CACHE_ROOT = os.environ.get('DISTILL_CACHE_ROOT') or config.DISTILL_CACHE     # the env override is for tests
SHARED_CACHE = os.path.join(CACHE_ROOT, '_shared') if CACHE_ROOT else None
MEL_ROOT = os.path.join(CACHE_ROOT, '_mel') if CACHE_ROOT else None

# ---- per-teacher cache
CACHE = os.path.join(CACHE_ROOT, TEACHER) if CACHE_ROOT else None
MANIFEST = os.path.join(CACHE, '_manifest') if CACHE else None
PLAN = os.path.join(MANIFEST, 'plan.csv') if CACHE else None
TEACHER_JSON = os.path.join(MANIFEST, 'teacher.json') if CACHE else None
TEACHER_EXT = os.path.join(MANIFEST, 'teacher_ext.onnx') if CACHE else None
DURATIONS = os.path.join(SHARED_CACHE, 'durations.csv') if CACHE_ROOT else None

# ---- local (repo-side, gitignored) data
LOCAL_ROOT = os.environ.get('DISTILL_LOCAL_ROOT') or os.path.join(MAIN, '05_distill', 'data')
SHARED = os.path.join(LOCAL_ROOT, '_shared')
LOCAL = os.path.join(LOCAL_ROOT, TEACHER)
RUNS, MODELS, EVAL, SHARDS = (os.path.join(LOCAL, d) for d in ('runs', 'models', 'eval', 'shards'))
LADDER = os.path.join(LOCAL, 'ladder.jsonl')
ARCH = os.path.join(SHARED, 'arch')
FRONTEND_ONNX = os.path.join(ARCH, 'frontend_only', 'model.onnx')

# ---- teacher files
TEACHER_MODEL_DIR = os.path.join(os.environ.get('DISTILL_MODELS_DIR') or os.path.join(MAIN, 'models'), TEACHER)
# ^ this project's training output (the env override is for tests)
TEACHER_ENGINE_DIR = os.path.join(ENGINE_MODELS, TEACHER) if ENGINE_MODELS else None
TEACHER_ONNX = os.path.join(TEACHER_ENGINE_DIR, 'model.onnx') if TEACHER_ENGINE_DIR else None
FIXTURE = os.path.join(ROOT, '04_deploy', 'fixtures', '230808_1208_s89520.flac')

FRAME_S = 0.96
HOP = 15360                 # samples per frame at 16 kHz
SLICE_FRAMES = 62
SLICE_SAMPLES = 953600      # 62 frames + 240 samples of STFT lookahead


def need_cache():
    if not CACHE_ROOT:
        sys.exit('set distill_cache in paths.local.json (the directory that holds one subdirectory per teacher)')
    return CACHE


def slice_path(root, relpath, hour):
    """<root>/<relpath without extension>/h<hour:06d>.npz, the layout every cache level shares."""
    return os.path.join(root, os.path.splitext(relpath)[0], f'h{int(hour):06d}.npz')


def mel_dir(spec):
    return os.path.join(MEL_ROOT, spec)


@functools.lru_cache(maxsize=None)
def spec():
    """The teacher's classes and activation centers, read from its training config."""
    p = os.path.join(TEACHER_MODEL_DIR, 'config_model.json')
    if not os.path.isfile(p):
        sys.exit(f'teacher {TEACHER!r}: {p} not found (a teacher is a model dir under models/)')
    cfg = json.load(open(p))
    classes = list(cfg['classes'])
    centers = cfg.get('activation_centers') or {}
    dead = [i for i, c in enumerate(classes) if c not in centers]
    return Spec(name=TEACHER, classes=classes, centers=centers, dead=dead,
                live=[i for i in range(len(classes)) if i not in dead],
                set=cfg.get('set', 'moderate'), translation=cfg.get('translation', 'general'),
                folds_train=cfg.get('folds_train') or [])


class Spec:
    def __init__(self, name, classes, centers, dead, live, set, translation, folds_train):
        self.name, self.classes, self.centers = name, classes, centers
        self.dead, self.live, self.set, self.translation = dead, live, set, translation
        self.folds_train = folds_train
        self.n_classes = len(classes)
        self.buzz = classes.index('ins_buzz')

    @property
    def code_dim(self):
        """Width of the head-input tensor, measured by teacher_onnx.py (teacher.json)."""
        if not TEACHER_JSON or not os.path.isfile(TEACHER_JSON):
            sys.exit(f'{TEACHER_JSON} missing: run teacher_onnx.py (main.py does) before anything reads targets')
        return int(json.load(open(TEACHER_JSON))['code_dim'])

    @property
    def audio_dir(self):
        """The set's framed-audio cache that eval_folds scores (frames as the embedder saw them)."""
        return os.path.join(MAIN, '02_set', 'sets', self.set, 'audio', 'sr16000_fl0.96', 'raw')
