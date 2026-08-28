import os

# All paths are absolute, anchored to the project root (this file's directory).
# This allows stage scripts to be invoked from any working directory.
ROOT = os.path.dirname(os.path.abspath(__file__))

def _p(*parts):
    return os.path.join(ROOT, *parts)

DIR_EMBEDDERS = _p('embedders')
DIR_MODELS = _p('models')
DIR_TRANSLATIONS = _p('translations')

TRAIN_DIR_AUDIO = _p('audio')

# sets
SET_DIR = _p('02_set', 'sets')
SET_SUBDIR_AUDIO = 'audio'
SET_SUBDIR_EMBEDDINGS = 'embeddings'
SET_SUBDIR_RAW = 'raw'


def dir_set(setname):
    return os.path.join(SET_DIR, setname)

def dir_audio(setname):
    return os.path.join(SET_DIR, setname, SET_SUBDIR_AUDIO)

def dir_embeddings_raw(setname, embeddername):
    return os.path.join(SET_DIR, setname, SET_SUBDIR_EMBEDDINGS, embeddername, SET_SUBDIR_RAW)

def dir_embeddings_fold(setname, embeddername, fold):
    return os.path.join(SET_DIR, setname, SET_SUBDIR_EMBEDDINGS, embeddername, SET_SUBDIR_RAW, fold)

def dir_embeddings_augment(setname, embeddername, aug_dirname):
    return os.path.join(SET_DIR, setname, SET_SUBDIR_EMBEDDINGS, embeddername, aug_dirname)


# embedding memory management. Frames per embedder forward pass — a pure
# batching knob, output-invariant. Lower it (via the env var) on GPUs with
# little VRAM; the default suits unified-memory / large-VRAM machines.
CHUNK_FRAMES = int(os.environ.get('BUZZDETECT_CHUNK_FRAMES', '300'))

# snip extraction
SNIP_BUFFER_S = 30.0

def dir_snips(setname):
    return os.path.join(SET_DIR, setname, SET_SUBDIR_AUDIO, 'snips')



