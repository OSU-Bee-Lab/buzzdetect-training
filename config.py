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


# test
SUBDIR_TESTS = 'tests'
FNAME_METRICS = 'metrics.csv'
FNAME_PLOT_TRADEOFF = 'plot_tradeoff.png'
FNAME_PLOT_TRADEOFF_ZOOMED = 'plot_tradeoff_zoomed.png'
FNAME_PLOT_METRICS = 'plot_metrics.png'
FNAME_PLOT_METRICS_ZOOMED = 'plot_metrics_zoomed.png'

PRECISION_ZOOM_MIN = 0.8
TEST_DIR = _p('04_test')
TEST_DIR_AUDIO = _p('04_test', 'audio')
TEST_DIR_EMBEDDINGS = _p('04_test', 'embeddings')

# embedding memory management
CHUNK_FRAMES = 300



