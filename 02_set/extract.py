import glob
import json
import os
import pickle
import warnings

from dataclasses import dataclass
import multiprocessing
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view

import config as cfg
from embedders.embedding import load_embedder, BaseEmbedder
from utils import read_pickle_exhaustive


def melt_coverage(cover_df, framelength=None):
    """ where cover_df is a dataframe with start and end columns OR framelength is provided"""
    if 'end' not in cover_df.columns and framelength is None:
        raise ValueError('cover_df has no "end" column and framelength is not provided')

    cover_df = cover_df.copy()  # don't mutate
    if 'end' not in cover_df.columns:
        cover_df['end'] = cover_df['start'] + framelength

    cover_df.sort_values("start", inplace=True)
    cover_df["coverageGroup"] = (cover_df["start"] > cover_df["end"].shift()).cumsum()
    df_coverage = cover_df.groupby("coverageGroup").agg({"start": "min", "end": "max"})

    coverage = list(zip(df_coverage['start'], df_coverage['end']))

    return coverage  # list of tuples


def ranges_overlap(range_frame, range_label, minimum_overlap):
    # Label fully inside frame = True even if the label is very short
    if range_label[0] >= range_frame[0] and range_label[1] <= range_frame[1]:
        return True
    overlap_start = max(range_frame[0], range_label[0])
    overlap_end = min(range_frame[1], range_label[1])
    actual_overlap = max(0, overlap_end - overlap_start)
    return actual_overlap >= minimum_overlap


def frame_audio(audio_data, framelength_s, samplerate, framehop_s):
    """
    Frame audio data using NumPy's sliding_window_view for efficient processing.

    Parameters:
    -----------
    audio_data : array-like
        Input audio data
    framelength : float
        Frame length in seconds
    samplerate : int or float
        Sample rate in Hz
    framehop_s : float
        Hop size between frames in seconds

    Returns:
    --------
    np.ndarray
        2D array where each row is a frame
    """
    audio_data = np.asarray(audio_data)
    framelength_samples = int(framelength_s * samplerate)
    audio_samples = len(audio_data)

    if audio_samples < framelength_samples:
        raise ValueError('sub-frame audio given')

    step = int(framehop_s * samplerate)

    # Use sliding_window_view for efficient framing
    windowed = sliding_window_view(audio_data, window_shape=framelength_samples)

    # Extract frames at specified hop intervals
    frames = windowed[::step]

    return frames


def events_in_frame(range_frame, annotations, event_overlap_s):
    overlap_mask = np.array([ranges_overlap(range_frame, (start, end), event_overlap_s) for (start, end) in zip(annotations['start'], annotations['end'])], dtype=bool)
    events = annotations['label'][overlap_mask].unique().tolist()
    events = sorted(events)

    return events


def expand_chunk(chunk, framelength_s, event_overlap_s, file_duration):
    """ Chunks need to be expanded so that framing starts and ends with the frame overlapping by event_overlap_s"""
    if file_duration < framelength_s:
        raise ValueError('input audio is shorter than one frame')

    start = max(
        # nudge start a bit forward, otherwise events might not quite overlap
        chunk[0] - ((framelength_s - event_overlap_s) * 0.99),
        0  # if start < 0, round up to 0
    )

    end = min(
        chunk[1] + (framelength_s - event_overlap_s),
        file_duration  # if end > file length, round down to file length
    )

    # if the annotation is right at the start of the file and less than a frame,
    # it can't be expanded from the center, so expand right. Vice verca for end of file.
    if (end - start) < framelength_s:
        if end == file_duration:
            start = end - (framelength_s * 0.99)
        elif start == 0:
            end = start + framelength_s
    return start, end


def collapse_labels(labellist):
    separator = '+'
    collapse = separator.join(labellist)
    return collapse


def get_ident_audio_path(ident):
    base_raw = os.path.join(cfg.TRAIN_DIR_AUDIO, ident)

    path_audio = glob.glob(base_raw + '.*')

    if len(path_audio) > 1:
        raise ValueError(f'multiple audio files found for ident {ident}')
    elif not path_audio:
        return []

    path_audio = path_audio[0]
    return path_audio


@dataclass
class ConfigExtract:
    setname: str
    embeddername: str
    overlap_event_prop: float
    framehop_prop: float

    def __post_init__(self):
        self.dir_set = cfg.dir_set(self.setname)
        self.dir_out_embeddings = cfg.dir_embeddings_raw(self.setname, self.embeddername)


class AssignIdent:
    def __init__(self, ident: str, fold: str, config_extract: ConfigExtract, dir_audio_base: str):
        self.ident = ident
        self.fold = fold
        self.config_extract = config_extract

        self.path_audio = get_ident_audio_path(ident)

        self.dir_out_audio = os.path.join(dir_audio_base, fold, ident)
        self.audio_exists =  bool(os.path.exists(self.dir_out_audio) and os.listdir(self.dir_out_audio))

        self.dir_out_embeddings = os.path.join(self.config_extract.dir_out_embeddings, fold, ident)
        self.embeddings_exist =  bool(os.path.exists(self.dir_out_embeddings) and os.listdir(self.dir_out_embeddings))

        self.handle, self.handle_msg = self._init_handle()


    def _init_handle(self):
        if not self.path_audio:
            return 'no_audio', 'audio file not found'
        if not self.audio_exists:
            return 'both', 'audio not extracted'
        if not self.embeddings_exist:
            return 'embeddings', 'audio already extracted'
        return 'skip', 'audio and embeddings already extracted'

class WorkerExtract:
    def __init__(self, config_extract: ConfigExtract, annotations: pd.DataFrame, folds: pd.DataFrame, q_extract: multiprocessing.Queue, name='worker_extract'):
        self.config_extract = config_extract
        self.name = name

        # embedder has framehop of 1 because we're framing manually
        self.embedder: BaseEmbedder = load_embedder(self.config_extract.embeddername, framehop_prop=1, initialize=False)

        self.overlap_event_s = (
            self.embedder.framelength_s * self.config_extract.overlap_event_prop
        )

        audio_key = self.embedder.audio_cache_key(
            self.config_extract.framehop_prop,
            self.config_extract.overlap_event_prop
        )
        self.dir_audio_cache_base = os.path.join(cfg.dir_audio(self.config_extract.setname), audio_key)

        self.folds = folds
        self.annotations = annotations
        self.q_extract = q_extract

        self.framelength_samples = int(self.embedder.framelength_s * self.embedder.samplerate)
        self.chunklength_samples = cfg.CHUNK_FRAMES * self.framelength_samples

    def read_range(self, track: sf.SoundFile, audiorange: tuple[float, float]):
        start_sample = round(track.samplerate * audiorange[0])
        samples_to_read = round(track.samplerate * (audiorange[1] - audiorange[0]))
        track.seek(start_sample)
        audio_data = track.read(samples_to_read, dtype=self.embedder.dtype_in)

        if track.channels > 1:
            audio_data = np.mean(audio_data, axis=1)

        audio_data = librosa.resample(y=audio_data, orig_sr=track.samplerate, target_sr=self.embedder.samplerate)

        return audio_data

    def extract_ident_embeddings(self, a_ident: AssignIdent):
        paths_audio = glob.glob(os.path.join(a_ident.dir_out_audio, '*.pickle'))

        os.makedirs(a_ident.dir_out_embeddings, exist_ok=True)

        def process_extracted_audio(path_audio):
            path_embedding = path_audio.replace(a_ident.dir_out_audio, a_ident.dir_out_embeddings)

            samples = read_pickle_exhaustive(path_audio)

            samples_flat = np.concatenate(samples)

            with open(path_embedding, 'wb') as file:
                for start in range(0, len(samples_flat), self.chunklength_samples):
                    chunk = samples_flat[start: start + self.chunklength_samples]
                    for e in self.embedder.embed(chunk):
                        pickle.dump(e, file)

        for path_audio in paths_audio:
            process_extracted_audio(path_audio)
        return None

    def extract_ident_both(self, a_ident: AssignIdent):
        annotations_sub = self.annotations[self.annotations['ident'] == a_ident.ident].copy()

        track = sf.SoundFile(a_ident.path_audio)
        audio_duration = track.frames / track.samplerate
        chunks_raw = melt_coverage(annotations_sub)
        chunks = [expand_chunk(chunk, self.embedder.framelength_s, self.overlap_event_s, audio_duration) for chunk in chunks_raw]

        frames_by_label = {}

        def process_chunk(chunk: tuple[float, float]):
            print(f'extractor {self.name}: ident {a_ident.ident}, chunk {chunk}')

            audio_data = self.read_range(track, chunk)

            frames = frame_audio(audio_data=audio_data, framelength_s=self.embedder.framelength_s,
                                 samplerate=self.embedder.samplerate, framehop_s=self.config_extract.framehop_prop*self.embedder.framehop_s)

            frametimes = chunk[0] + (np.arange(0, len(frames)) * self.config_extract.framehop_prop * self.embedder.framelength_s)
            frametimes = [(s, s + self.embedder.framelength_s) for s in frametimes]

            for frame, frame_range in zip(frames, frametimes):
                events_frame = events_in_frame(
                    range_frame=frame_range,
                    annotations=annotations_sub,
                    event_overlap_s=self.overlap_event_s
                )

                labels_collapse = collapse_labels(events_frame)
                if labels_collapse == '':
                    annotations_sub.to_csv('no_events_annotations.csv', index=False)
                    msg = (f'extractor {self.name}: no events found in frame {frame_range} for ident {a_ident.ident}'
                           f'Frame times: {frame_range}, events {events_frame}')

                    raise ValueError(msg)

                if labels_collapse not in frames_by_label:
                    frames_by_label[labels_collapse] = [frame]
                else:
                    frames_by_label[labels_collapse].append(frame)

        for chunk in chunks:
            process_chunk(chunk)

        os.makedirs(a_ident.dir_out_audio, exist_ok=True)
        os.makedirs(a_ident.dir_out_embeddings, exist_ok=True)

        for labels_collapse, samples in frames_by_label.items():
            path_out_samples = os.path.join(a_ident.dir_out_audio, labels_collapse + '.pickle')
            path_out_embedding = os.path.join(a_ident.dir_out_embeddings, labels_collapse + '.pickle')
            with open(path_out_samples, 'wb') as file:
                for s in samples:
                    pickle.dump(s, file)

            samples_flat = np.concatenate(samples)

            with open(path_out_embedding, 'wb') as file:
                for start in range(0, len(samples_flat), self.chunklength_samples):
                    chunk = samples_flat[start: start + self.chunklength_samples]
                    for e in self.embedder.embed(chunk):
                        pickle.dump(e, file)
        return None

    def extract_ident(self, ident):
        fold = self.folds[self.folds['ident'] == ident]['fold'].unique()
        if len(fold) > 1:
            warnings.warn(f'extractor {self.name}: ident {ident} has multiple folds; skipping')
            return
        elif len(fold) == 0:
            warnings.warn(f'extractor {self.name}: ident {ident} has no fold; skipping')
            return
        else:
            fold = fold[0]

        a_ident = AssignIdent(ident=ident, fold=fold, config_extract=self.config_extract, dir_audio_base=self.dir_audio_cache_base)
        if a_ident.handle == 'both':
            self.extract_ident_both(a_ident)
        elif a_ident.handle == 'embeddings':
            print(f'extractor {self.name}: extracting embeddings for {ident}')
            self.extract_ident_embeddings(a_ident)
        elif a_ident.handle == 'no_audio':
            warnings.warn(f'extractor {self.name}: skipping {ident}; {a_ident.handle_msg}')
            return
        elif a_ident.handle == 'skip':
            print(f'extractor {self.name}: skipping {ident}; {a_ident.handle_msg}')
        else:
            raise ValueError(f'extractor {self.name}: unknown handle {a_ident.handle} for ident {ident}')

    def run(self):
        self.embedder.initialize()
        ident = self.q_extract.get()
        while ident != 'TERMINATE':
            self.extract_ident(ident)
            ident = self.q_extract.get()
        print(f'extractor {self.name}: terminating')


def run_worker(config_extract: ConfigExtract, annotations: pd.DataFrame, folds: pd.DataFrame, q_extract: multiprocessing.Queue, name):
    worker = WorkerExtract(config_extract, annotations, folds, q_extract, name)
    worker.run()


def extract_set(setname, embeddername, overlap_event_prop, framehop_prop, n_workers):
    dir_set = cfg.dir_set(setname)
    path_config = os.path.join(dir_set, 'config_extract.json')

    if os.path.exists(path_config):
        with open(path_config, 'r') as f:
            saved = json.load(f)
        config_extract = ConfigExtract(**saved)
        print(f'loaded config_extract from {path_config}')
    else:
        config_extract = ConfigExtract(setname=setname, embeddername=embeddername, overlap_event_prop=overlap_event_prop, framehop_prop=framehop_prop)
        os.makedirs(dir_set, exist_ok=True)
        core_fields = {k: getattr(config_extract, k) for k in ConfigExtract.__dataclass_fields__}
        with open(path_config, 'w') as f:
            json.dump(core_fields, f, indent=2)
        print(f'saved config_extract to {path_config}')

    folds = pd.read_csv(os.path.join(dir_set, 'folds.csv'))
    annotations = pd.read_csv(os.path.join(dir_set, 'annotations.csv'))
    idents = annotations['ident'].unique()

    # Pre-filter: determine which idents actually need work before spawning workers
    embedder_tmp = load_embedder(embeddername, framehop_prop=1, initialize=False)
    audio_key = embedder_tmp.audio_cache_key(framehop_prop, overlap_event_prop)
    dir_audio_cache_base = os.path.join(cfg.dir_audio(setname), audio_key)

    idents_todo = []
    for ident in idents:
        fold_rows = folds[folds['ident'] == ident]['fold'].unique()
        if len(fold_rows) != 1:
            continue
        fold = fold_rows[0]
        a_ident = AssignIdent(ident=ident, fold=fold, config_extract=config_extract, dir_audio_base=dir_audio_cache_base)
        if a_ident.handle not in ('skip', 'no_audio'):
            idents_todo.append(ident)

    n_skip = len(idents) - len(idents_todo)
    print(f'  {len(idents_todo)} idents to process, {n_skip} already done or missing audio')

    if not idents_todo:
        print('all idents already extracted; skipping workers')
        return True

    q_extract = multiprocessing.Queue()
    for i in idents_todo:
        q_extract.put(i)

    workers = []
    for w in range(n_workers):
        q_extract.put('TERMINATE')
        workers.append(multiprocessing.Process(target=run_worker, args=(config_extract, annotations, folds, q_extract, w)))

    for w in workers:
        w.start()

    for w in workers:
        w.join()

    print(f'all extractions complete\n:)\n:D\n:O')
    return True
