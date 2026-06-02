import os


DIR_EMBEDDERS = 'embedders'
DIR_MODELS = 'models'
DIR_TRANSLATIONS = 'translations'

TRAIN_DIR_AUDIO = 'audio'

# sets
SET_DIR = '02_set/sets'
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
TEST_DIR = '04_test'
TEST_DIR_AUDIO = os.path.join(TEST_DIR, 'audio')
TEST_DIR_EMBEDDINGS = os.path.join(TEST_DIR, 'embeddings')

# embedding memory management
CHUNK_FRAMES = 300



