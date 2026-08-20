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


# scoring artifacts
FNAME_METRICS = 'metrics.csv'
FNAME_SX = 'sx.csv'
FNAME_PLOT_TRADEOFF = 'plot_tradeoff.png'
FNAME_PLOT_TRADEOFF_ZOOMED = 'plot_tradeoff_zoomed.png'
FNAME_PLOT_METRICS = 'plot_metrics.png'
FNAME_PLOT_METRICS_ZOOMED = 'plot_metrics_zoomed.png'

PRECISION_ZOOM_MIN = 0.8

# tools/night_positives.py: one all-night recording, the known failure mode
NIGHT_DIR = _p('tools', 'night-positives')
SUBDIR_NIGHT = 'night-positives'

# embedding memory management
CHUNK_FRAMES = 300

# snip extraction
SNIP_BUFFER_S = 30.0

def dir_snips(setname):
    return os.path.join(SET_DIR, setname, SET_SUBDIR_AUDIO, 'snips')



