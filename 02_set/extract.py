import glob
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
import warnings

# Run directly (`python 02_set/extract.py --set <set>`) as well as imported: the project
# root holds config.py and the package dirs this module imports from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    # cummax, not the previous row's end: an annotation that encloses later ones must keep
    # the group open. Without it a segment-spanning label (e.g. a 300 s ins_trill over a
    # gap-filled segment) shatters into one cluster per annotation.
    cover_df["coverageGroup"] = (cover_df["start"] > cover_df["end"].shift().cummax()).cumsum()
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
    """Expand an annotation window so framing starts/ends with frames overlapping by event_overlap_s."""
    if file_duration < framelength_s:
        raise ValueError('input audio is shorter than one frame')

    start = max(
        chunk[0] - ((framelength_s - event_overlap_s) * 0.99),
        0
    )
    end = min(
        chunk[1] + (framelength_s - event_overlap_s),
        file_duration
    )

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


# ---------------------------------------------------------------------------
# Snip extraction (layer 1: HDD → per-annotation-cluster WAV files)
# ---------------------------------------------------------------------------

FNAME_SNIP_MANIFEST = 'manifest.json'
FNAME_FINGERPRINT = 'annotations.fingerprint'


def _fingerprint_annotations(annotations_sub: pd.DataFrame) -> str:
    """Hash one ident's annotations. Everything downstream of the snips — which
    ranges get cut, how frames are labelled — is a pure function of these rows, so
    a matching fingerprint means the extracted product is still current and a
    differing one means it has to be rebuilt."""
    rows = sorted(
        f'{r.start:.6f}\x1f{r.end:.6f}\x1f{r.label}'
        for r in annotations_sub.itertuples()
    )
    return hashlib.sha1('\x1e'.join(rows).encode()).hexdigest()


def _read_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _read_fingerprint(dir_path):
    try:
        with open(os.path.join(dir_path, FNAME_FINGERPRINT), 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def _write_fingerprint(dir_path, fingerprint):
    with open(os.path.join(dir_path, FNAME_FINGERPRINT), 'w') as f:
        f.write(fingerprint + '\n')


def _has_pickles(dir_path):
    """Sidecar files (a fingerprint) don't count as extracted output."""
    return os.path.isdir(dir_path) and any(n.endswith('.pickle') for n in os.listdir(dir_path))


def _find_ident_dirs(root, ident):
    """Directories under root whose trailing path components are the ident.

    A walk rather than a glob: both fold names and idents are path-like and neither
    has a fixed depth, so there is no pattern of '*'s that lands on the right level.
    """
    suffix = os.sep + ident.replace('/', os.sep)
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        if dirpath.endswith(suffix):
            found.append(dirpath)
            dirnames.clear()  # nothing nested under an ident dir
    return found


def _purge_ident_outputs(setname, ident, audio_key=None, embeddername=None):
    """Delete the cached audio and embeddings for one ident, raw and augmented alike.

    Fold is not passed: an ident that moved folds leaves its old copies behind, and
    those are stale for the same reason. Pass audio_key/embeddername to confine the
    purge to one embedder's products; with neither, every embedder's are removed.
    """
    dir_snips_base = cfg.dir_snips(setname)  # snips live under audio/ but are layer 1
    roots = []
    if audio_key is not None:
        roots.append(os.path.join(cfg.dir_audio(setname), audio_key))
    if embeddername is not None:
        roots.append(os.path.join(cfg.dir_set(setname), cfg.SET_SUBDIR_EMBEDDINGS, embeddername))
    if audio_key is None and embeddername is None:
        roots = [cfg.dir_audio(setname),
                 os.path.join(cfg.dir_set(setname), cfg.SET_SUBDIR_EMBEDDINGS)]

    removed = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for path in _find_ident_dirs(root, ident):
            if path == dir_snips_base or path.startswith(dir_snips_base + os.sep):
                continue
            shutil.rmtree(path)
            removed.append(path)
    return removed


def _snip_paths(dir_snips_ident):
    """Snips of one ident, sorted. The directory name comes from an ident (a file
    path), so it is escaped — a '[' in a site name is not a character class."""
    return sorted(glob.glob(os.path.join(glob.escape(dir_snips_ident), 'snip_*.flac')))


def _find_snip_dirs(dir_snips_base):
    """(path, ident) for every directory under dir_snips_base that holds snips.

    An ident is a path into the source audio tree ('Effort/Site/1/file'), so its snip
    directory is nested to match and the top level of dir_snips_base is a list of
    effort names, not idents.
    """
    found = []
    for dirpath, _, filenames in os.walk(dir_snips_base):
        if any(n.startswith('snip_') and n.endswith('.flac') for n in filenames):
            found.append((dirpath, os.path.relpath(dirpath, dir_snips_base)))
    return found


def _snip_filename(start: float, end: float) -> str:
    return f"snip_{start:.3f}_{end:.3f}.flac"


def _parse_snip_bounds(path: str) -> tuple[float, float]:
    name = os.path.basename(path).rsplit('.', 1)[0]
    parts = name.split('_')  # ['snip', '<start>', '<end>']
    return float(parts[1]), float(parts[2])


def _buffer_and_merge(chunks, buffer_s: float, duration: float):
    """Pad each annotation cluster by buffer_s, then merge ranges that now overlap.

    The buffer is a staging decision only — it says which clusters are close enough to
    pull off slow storage in one read. Merging is what keeps snips disjoint; overlapping
    snips would frame (and label, and embed) the shared seconds once per snip. Chunk
    boundaries inside a snip are re-derived from the annotations, so merging here never
    widens a training sample.
    """
    padded = sorted(
        (max(0.0, start - buffer_s), min(duration, end + buffer_s))
        for start, end in chunks
    )

    merged = []
    for start, end in padded:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [(start, end) for start, end in merged]


def _sync_snips_ident(ident: str, annotations_sub: pd.DataFrame, path_audio: str, dir_out: str,
                      verbose=False, t0=None):
    """Bring one ident's snip directory in line with its current annotations.

    Annotation efforts are ongoing, so the snip set an ident wants moves over time:
    two fragmented events become one continuous one, an event is deleted, a cluster
    widens past its neighbour and the two merge. Rather than trusting whatever is on
    disk, this re-derives the wanted snips, deletes the files that no longer appear
    in that list, and reads only the ranges that are missing — the source audio sits
    on slow media, so reading nothing is the point.

    The manifest is a fast path, not the source of truth: when its fingerprint still
    matches and every snip it names is present, the source file is never opened.

    Returns (status, n_written, n_deleted) where status is 'unchanged' or 'updated'.
    """
    fingerprint = _fingerprint_annotations(annotations_sub)
    path_manifest = os.path.join(dir_out, FNAME_SNIP_MANIFEST)
    manifest = _read_json(path_manifest)

    if manifest is not None and manifest.get('fingerprint') == fingerprint:
        wanted_names = [_snip_filename(s, e) for s, e in manifest.get('snips', ())]
        if all(os.path.exists(os.path.join(dir_out, n)) for n in wanted_names):
            return 'unchanged', 0, 0

    os.makedirs(dir_out, exist_ok=True)

    with sf.SoundFile(path_audio) as track:
        duration = track.frames / track.samplerate
        sr = track.samplerate
        chunks_raw = _buffer_and_merge(melt_coverage(annotations_sub), cfg.SNIP_BUFFER_S, duration)
        wanted = {_snip_filename(start, end): (start, end) for start, end in chunks_raw}

        n_deleted = 0
        for path_existing in _snip_paths(dir_out):
            if os.path.basename(path_existing) in wanted:
                continue
            if verbose:
                print(f'{time.time()-t0:.1f}s - extract_snips: {ident} '
                      f'superseded, deleting {os.path.basename(path_existing)}', flush=True)
            os.remove(path_existing)
            n_deleted += 1

        n_written = 0
        for i, (name, (start, end)) in enumerate(wanted.items()):
            path_out = os.path.join(dir_out, name)

            if os.path.exists(path_out):
                continue

            if verbose:
                print(f'{time.time()-t0:.1f}s - extract_snips: {ident} '
                      f'[{i+1}/{len(wanted)}] {start:.1f}-{end:.1f}s', flush=True)

            start_sample = round(sr * start)
            n_samples = round(sr * (end - start))
            track.seek(start_sample)
            audio_data = track.read(n_samples)

            if track.channels > 1:
                audio_data = np.mean(audio_data, axis=1)

            sf.write(path_out, audio_data, sr)
            n_written += 1

    with open(path_manifest, 'w') as f:
        json.dump({'fingerprint': fingerprint,
                   'snips': [[start, end] for start, end in chunks_raw]}, f, indent=2)

    status = 'updated' if (n_written or n_deleted) else 'unchanged'
    return status, n_written, n_deleted


def extract_snips(setname: str, verbose=False):
    """Extract raw audio snips for all idents in a set.

    Run this before extract_set. Safe and cheap to re-run: an ident whose
    annotations are unchanged is skipped without opening its source audio, and one
    whose annotations moved has only the superseded snips deleted and only the
    missing ranges read. Returns {ident: 'unchanged' | 'updated' | 'missing_audio'}
    so callers can invalidate what an 'updated' ident already had extracted.

    Snips are stored as:
        02_set/sets/<setname>/audio/snips/<ident>/snip_<start>_<end>.flac

    Each snip covers one or more annotation clusters padded by SNIP_BUFFER_S on
    each side; clusters whose padded ranges overlap are merged into a single
    snip, so snips for an ident are always disjoint. The buffer is a staging
    decision (what to pull off slow storage in one read), not a training-sample
    boundary — extract_ident_both re-derives chunks from the annotations inside
    each snip. Snips are at the source file's native sample rate (no
    per-embedder resampling at this stage).
    """
    dir_set = cfg.dir_set(setname)
    annotations = pd.read_csv(os.path.join(dir_set, 'annotations.csv'))
    dir_snips_base = cfg.dir_snips(setname)

    idents = annotations['ident'].unique()
    status_by_ident = {}
    n_written = n_deleted = 0
    n_done = 0
    t0 = time.time()

    for ident in idents:
        annotations_sub = annotations[annotations['ident'] == ident]
        dir_out = os.path.join(dir_snips_base, ident)

        path_audio = get_ident_audio_path(ident)
        if not path_audio:
            # Only a problem if this ident actually needs a read; a fully snipped
            # ident doesn't care that the source drive is unplugged.
            manifest = _read_json(os.path.join(dir_out, FNAME_SNIP_MANIFEST))
            if manifest is not None and manifest.get('fingerprint') == _fingerprint_annotations(annotations_sub):
                status_by_ident[ident] = 'unchanged'
                continue
            warnings.warn(f'extract_snips: no audio file for {ident}; skipping')
            status_by_ident[ident] = 'missing_audio'
            continue

        if verbose:
            print(f'{time.time()-t0:.1f}s - extract_snips: [{n_done+1}/{len(idents)}] {ident} — '
                  f'starting ({len(annotations_sub)} annotations)', flush=True)
        status, written, deleted = _sync_snips_ident(
            ident, annotations_sub, path_audio, dir_out, verbose=verbose, t0=t0)
        status_by_ident[ident] = status
        n_written += written
        n_deleted += deleted
        n_done += 1
        if verbose:
            print(f'{time.time()-t0:.1f}s - extract_snips: [{n_done}/{len(idents)}] {ident} — '
                  f'{status} ({written} written, {deleted} deleted)', flush=True)

    # An ident dropped from the annotations entirely: every snip it has is superseded.
    # An ident is a source-audio path, so its snip directory sits at an arbitrary depth
    # under dir_snips_base — walk for directories that actually hold snips rather than
    # reading the top level as a list of idents.
    known = set(idents)
    for path_orphan, ident_orphan in sorted(_find_snip_dirs(dir_snips_base)):
        if ident_orphan in known:
            continue
        warnings.warn(f'extract_snips: {ident_orphan} is no longer in annotations.csv; removing its snips')
        shutil.rmtree(path_orphan)
        for path_removed in _purge_ident_outputs(setname, ident_orphan):
            warnings.warn(f'extract_snips: removed stale output {path_removed}')

    n_updated = sum(1 for s in status_by_ident.values() if s == 'updated')
    n_missing = sum(1 for s in status_by_ident.values() if s == 'missing_audio')
    print(f'{time.time()-t0:.1f}s - extract_snips: {n_updated} ident(s) updated '
          f'({n_written} snip(s) written, {n_deleted} superseded), '
          f'{len(idents) - n_updated - n_missing} unchanged, {n_missing} missing audio')

    return status_by_ident


# ---------------------------------------------------------------------------
# Per-embedder extraction (layers 2–3: snips → framed audio cache → embeddings)
# ---------------------------------------------------------------------------

@dataclass
class ConfigExtract:
    setname: str
    overlap_event_prop: float
    framehop_prop: float
    embeddername: str = None  # runtime-only, not persisted to config_extract.json

    STORED_FIELDS = ('setname', 'overlap_event_prop', 'framehop_prop')

    def __post_init__(self):
        self.dir_set = cfg.dir_set(self.setname)

    def dir_out_embeddings(self, embeddername):
        return cfg.dir_embeddings_raw(self.setname, embeddername)


class AssignIdent:
    """What, if anything, still has to be done for one ident — and what already on
    disk was built from annotations that have since changed.

    Staleness is decided by the annotation fingerprint each output directory carries.
    A directory with no fingerprint predates the check and is left alone: it can't be
    judged, and the snip layer catches the case that matters (annotations that moved
    far enough to change the snips), via `force_stale`.
    """

    def __init__(self, ident: str, fold: str, config_extract: ConfigExtract,
                 dir_audio_base: str, dir_embeddings_base: str, dir_snips_base: str,
                 audio_key: str = None, fingerprint: str = None, force_stale: bool = False):
        self.ident = ident
        self.fold = fold
        self.config_extract = config_extract
        self.audio_key = audio_key
        self.fingerprint = fingerprint

        self.dir_out_audio = os.path.join(dir_audio_base, fold, ident)
        self.audio_exists = _has_pickles(self.dir_out_audio)

        self.dir_out_embeddings = os.path.join(dir_embeddings_base, fold, ident)
        self.embeddings_exist = _has_pickles(self.dir_out_embeddings)

        self.dir_snips_ident = os.path.join(dir_snips_base, ident)
        self.snips_exist = bool(_snip_paths(self.dir_snips_ident))

        self.audio_stale = self.audio_exists and self._is_stale(self.dir_out_audio, force_stale)
        # Embeddings are derived from the cached audio, so stale audio implies stale embeddings.
        self.embeddings_stale = self.embeddings_exist and (
            self.audio_stale or self._is_stale(self.dir_out_embeddings, force_stale))

        self.handle, self.handle_msg = self._init_handle()

    def _is_stale(self, dir_path, force_stale):
        if force_stale:
            return True
        if self.fingerprint is None:
            return False
        found = _read_fingerprint(dir_path)
        return found is not None and found != self.fingerprint

    def purge_stale(self):
        """Delete what the current annotations invalidated, augmented derivatives
        included — an augmentation of a stale frame is just as stale. Idempotent, so
        it is safe to call in the worker after the pre-filter has sized up the run."""
        removed = []
        if self.audio_stale:
            removed += _purge_ident_outputs(
                self.config_extract.setname, self.ident,
                audio_key=self.audio_key, embeddername=self.config_extract.embeddername)
            self.audio_exists = self.audio_stale = False
            self.embeddings_exist = self.embeddings_stale = False
        elif self.embeddings_stale:
            removed += _purge_ident_outputs(
                self.config_extract.setname, self.ident,
                embeddername=self.config_extract.embeddername)
            self.embeddings_exist = self.embeddings_stale = False
        return removed

    def _init_handle(self):
        audio_usable = self.audio_exists and not self.audio_stale
        embeddings_usable = self.embeddings_exist and not self.embeddings_stale

        if audio_usable and embeddings_usable:
            return 'skip', 'audio and embeddings already extracted'
        if audio_usable and not embeddings_usable:
            return 'embeddings', ('embeddings stale; rebuilding from cached audio'
                                  if self.embeddings_stale else 'audio already extracted')
        if not audio_usable and self.snips_exist:
            return 'both', ('annotations changed since extraction; re-extracting'
                            if self.audio_stale else 'snips available, audio not extracted')
        return 'no_snips', 'snips not found; run extract_snips first'


class WorkerExtract:
    def __init__(self, config_extract: ConfigExtract, annotations: pd.DataFrame, folds: pd.DataFrame,
                 q_extract: multiprocessing.Queue, name='worker_extract', verbose=False,
                 idents_forced=()):
        self.config_extract = config_extract
        self.name = name
        self.verbose = verbose
        self.idents_forced = set(idents_forced)  # snips changed → whatever was extracted is stale
        self.n_done = 0  # idents this worker has finished, for verbose progress

        # embedder has framehop of 1 because we're framing manually
        self.embedder: BaseEmbedder = load_embedder(self.config_extract.embeddername, framehop_prop=1, initialize=False)

        self.overlap_event_s = (
            self.embedder.framelength_s * self.config_extract.overlap_event_prop
        )

        self.audio_key = self.embedder.audio_cache_key()
        self.dir_audio_cache_base = os.path.join(cfg.dir_audio(self.config_extract.setname), self.audio_key, 'raw')
        self.dir_embeddings_base = self.config_extract.dir_out_embeddings(self.config_extract.embeddername)
        self.dir_snips_base = cfg.dir_snips(self.config_extract.setname)

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
        return librosa.resample(y=audio_data, orig_sr=track.samplerate, target_sr=self.embedder.samplerate)

    def extract_ident_embeddings(self, a_ident: AssignIdent):
        paths_audio = glob.glob(os.path.join(glob.escape(a_ident.dir_out_audio), '*.pickle'))

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
        if a_ident.fingerprint is not None:
            _write_fingerprint(a_ident.dir_out_embeddings, a_ident.fingerprint)
        return f'{len(paths_audio)} cached label file(s) → embeddings'

    def extract_ident_both(self, a_ident: AssignIdent):
        annotations_sub = self.annotations[self.annotations['ident'] == a_ident.ident].copy()

        snip_paths = _snip_paths(a_ident.dir_snips_ident)
        if not snip_paths:
            raise FileNotFoundError(
                f'no snips for {a_ident.ident} in {a_ident.dir_snips_ident}; run extract_snips first'
            )

        frames_by_label = {}

        for i, snip_path in enumerate(snip_paths):
            snip_start, _ = _parse_snip_bounds(snip_path)

            if self.verbose:
                print(f'{time.time()-self.t0:.1f}s - extractor {self.name}: {a_ident.ident} '
                      f'snip [{i+1}/{len(snip_paths)}] {os.path.basename(snip_path)}', flush=True)

            with sf.SoundFile(snip_path) as track:
                snip_duration = track.frames / track.samplerate

                # Annotations for this snip, converted to snip-relative coords
                ann_rel = annotations_sub[
                    (annotations_sub['end'] > snip_start) &
                    (annotations_sub['start'] < snip_start + snip_duration)
                ].copy()
                ann_rel['start'] = ann_rel['start'] - snip_start
                ann_rel['end'] = ann_rel['end'] - snip_start

                frames_rel = []  # snip-relative ranges of every frame emitted for this snip

                if snip_duration < self.embedder.framelength_s:
                    # Snip is shorter than one frame — skip. Zero-padding to framelength would
                    # corrupt training embeddings (the model would see silence rather than missing
                    # audio), so we discard short clips rather than pad them.
                    warnings.warn(
                        f'extractor {self.name}: snip {snip_start:.3f}s too short for '
                        f'{self.embedder.embeddername} ({snip_duration:.3f}s < '
                        f'{self.embedder.framelength_s}s); skipping'
                    )
                    continue

                chunks_raw = melt_coverage(ann_rel)
                chunks = [
                    expand_chunk(chunk, self.embedder.framelength_s, self.overlap_event_s, snip_duration)
                    for chunk in chunks_raw
                ]

                for chunk in chunks:
                    audio_data = self.read_range(track, chunk)

                    if len(audio_data) < int(self.embedder.framelength_s * self.embedder.samplerate):
                        warnings.warn(
                            f'extractor {self.name}: sub-frame audio in {a_ident.ident} '
                            f'snip {snip_start:.3f}s chunk {chunk}; skipping'
                        )
                        continue

                    frames = frame_audio(
                        audio_data=audio_data,
                        framelength_s=self.embedder.framelength_s,
                        samplerate=self.embedder.samplerate,
                        framehop_s=self.config_extract.framehop_prop * self.embedder.framehop_s,
                    )

                    # Frame times in source coords for events_in_frame
                    chunk_source_start = snip_start + chunk[0]
                    frame_starts = chunk_source_start + np.arange(len(frames)) * self.config_extract.framehop_prop * self.embedder.framelength_s
                    frametimes = [(s, s + self.embedder.framelength_s) for s in frame_starts]

                    frames_rel += [(f[0] - snip_start, f[1] - snip_start) for f in frametimes]

                    for frame, frame_range in zip(frames, frametimes):
                        events_frame = events_in_frame(
                            range_frame=frame_range,
                            annotations=annotations_sub,
                            event_overlap_s=self.overlap_event_s,
                        )
                        labels_collapse = collapse_labels(events_frame)
                        if not labels_collapse:
                            any_overlap = any(
                                max(0, min(frame_range[1], end) - max(frame_range[0], start)) > 0
                                for start, end in zip(annotations_sub['start'], annotations_sub['end'])
                            )
                            if any_overlap:
                                # Boundary frame: annotation touches but falls short of event_overlap_s
                                # threshold due to sample-level rounding in expand_chunk. Safe to skip.
                                warnings.warn(
                                    f'extractor {self.name}: frame {frame_range} has sub-threshold '
                                    f'annotation overlap for ident {a_ident.ident}; '
                                    f'boundary rounding — skipping'
                                )
                            else:
                                raise ValueError(
                                    f'extractor {self.name}: frame {frame_range} has NO annotation '
                                    f'overlap for ident {a_ident.ident} — extraction integrity violation'
                                )
                            continue
                        frames_by_label.setdefault(labels_collapse, []).append(frame)

                # Rescue annotations the frame grid missed. Frames are cut on a grid anchored
                # at each chunk's start, and with framehop_prop=1 they do not overlap each
                # other, so a short event straddling a grid boundary — or sitting in the
                # sub-frame remainder after a chunk's last whole hop — can hold less than
                # overlap_event_s in every frame and be labelled nowhere. A labelled event
                # must always reach the training data, so cut one extra frame centred on it.
                for _, row in ann_rel.iterrows():
                    event = (row['start'], row['end'])
                    if any(ranges_overlap(f, event, self.overlap_event_s) for f in frames_rel):
                        continue

                    frame_start = min(
                        max((event[0] + event[1]) / 2 - self.embedder.framelength_s / 2, 0.0),
                        snip_duration - self.embedder.framelength_s,
                    )
                    frame_range_rel = (frame_start, frame_start + self.embedder.framelength_s)
                    # read a hair wide, then truncate: resampling the exact frame length can
                    # land a sample short depending on the source rate
                    audio_data = self.read_range(
                        track,
                        (frame_range_rel[0], min(frame_range_rel[1] + 0.01, snip_duration)),
                    )[:self.framelength_samples]

                    if len(audio_data) < self.framelength_samples:
                        warnings.warn(
                            f'extractor {self.name}: could not rescue {row["label"]!r} at '
                            f'{snip_start + event[0]:.3f}s in {a_ident.ident}; sub-frame audio'
                        )
                        continue

                    frame_range = (snip_start + frame_range_rel[0], snip_start + frame_range_rel[1])
                    labels_collapse = collapse_labels(events_in_frame(
                        range_frame=frame_range,
                        annotations=annotations_sub,
                        event_overlap_s=self.overlap_event_s,
                    ))
                    if not labels_collapse:
                        raise ValueError(
                            f'extractor {self.name}: rescue frame {frame_range} centred on '
                            f'{row["label"]!r} at {snip_start + event[0]:.3f}s in '
                            f'{a_ident.ident} matched no annotation — extraction integrity violation'
                        )

                    frames_by_label.setdefault(labels_collapse, []).append(audio_data)
                    frames_rel.append(frame_range_rel)

                    if self.verbose:
                        print(f'{time.time()-self.t0:.1f}s - extractor {self.name}: rescued '
                              f'{row["label"]!r} at {snip_start + event[0]:.3f}s '
                              f'({event[1]-event[0]:.3f}s) as {labels_collapse}', flush=True)

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

        # Stamped last: a fingerprint means the directory's whole contents were built
        # from these annotations, so a crash mid-write must not leave one behind.
        if a_ident.fingerprint is not None:
            _write_fingerprint(a_ident.dir_out_audio, a_ident.fingerprint)
            _write_fingerprint(a_ident.dir_out_embeddings, a_ident.fingerprint)

        n_frames = sum(len(s) for s in frames_by_label.values())
        return (f'{len(snip_paths)} snip(s) → {n_frames} frames, '
                f'{len(frames_by_label)} label(s)')

    def _log_ident(self, ident, msg, t_ident):
        """One line per ident, verbose only — the granularity a human watching a
        long extraction wants. The default run stays silent here; its progress
        signal is the ident counts printed by extract_set."""
        if not self.verbose:
            return
        self.n_done += 1
        print(f'{time.time()-self.t0:.1f}s - extractor {self.name}: [{self.n_done}] {ident} — {msg} '
              f'(ident took {time.time()-t_ident:.1f}s)', flush=True)

    def extract_ident(self, ident):
        fold = self.folds[self.folds['ident'] == ident]['fold'].unique()
        if len(fold) > 1:
            warnings.warn(f'extractor {self.name}: ident {ident} has multiple folds; skipping')
            return
        elif len(fold) == 0:
            warnings.warn(f'extractor {self.name}: ident {ident} has no fold; skipping')
            return
        else:
            fold = str(fold[0])

        a_ident = AssignIdent(
            ident=ident, fold=fold, config_extract=self.config_extract,
            dir_audio_base=self.dir_audio_cache_base,
            dir_embeddings_base=self.dir_embeddings_base,
            dir_snips_base=self.dir_snips_base,
            audio_key=self.audio_key,
            fingerprint=_fingerprint_annotations(self.annotations[self.annotations['ident'] == ident]),
            force_stale=ident in self.idents_forced,
        )
        for path_removed in a_ident.purge_stale():
            warnings.warn(f'extractor {self.name}: {ident} superseded by new annotations; '
                          f'removed {path_removed}')
        if self.verbose:
            print(f'{time.time()-self.t0:.1f}s - extractor {self.name}: [{self.n_done+1}] {ident} — '
                  f'starting ({a_ident.handle})', flush=True)
        t_ident = time.time()
        try:
            if a_ident.handle == 'both':
                msg = self.extract_ident_both(a_ident)
                self._log_ident(ident, msg, t_ident)
            elif a_ident.handle == 'embeddings':
                msg = self.extract_ident_embeddings(a_ident)
                self._log_ident(ident, msg, t_ident)
            elif a_ident.handle == 'no_snips':
                warnings.warn(f'extractor {self.name}: skipping {ident}; {a_ident.handle_msg}')
                return
            elif a_ident.handle == 'skip':
                self._log_ident(ident, f'skipped — {a_ident.handle_msg}', t_ident)
            else:
                raise ValueError(f'extractor {self.name}: unknown handle {a_ident.handle} for ident {ident}')
        except ValueError:
            raise  # integrity violations must not be swallowed
        except Exception as e:
            warnings.warn(f'extractor {self.name}: error on {ident}: {e}')

    def run(self):
        self.t0 = time.time()
        self.embedder.initialize()
        ident = self.q_extract.get()
        while ident != 'TERMINATE':
            self.extract_ident(ident)
            ident = self.q_extract.get()
        if self.verbose:
            print(f'{time.time()-self.t0:.1f}s - extractor {self.name}: terminating')


def run_worker(config_extract: ConfigExtract, annotations: pd.DataFrame, folds: pd.DataFrame,
               q_extract: multiprocessing.Queue, name, verbose=False, idents_forced=()):
    try:
        worker = WorkerExtract(config_extract, annotations, folds, q_extract, name, verbose=verbose,
                               idents_forced=idents_forced)
        worker.run()
    except Exception as e:
        import traceback
        print(f'WORKER {name} FATAL: {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()
        raise


def extract_set(setname, embeddername, overlap_event_prop=None, framehop_prop=None, n_workers=4, verbose=False):
    t0 = time.time()
    dir_set = cfg.dir_set(setname)
    path_config = os.path.join(dir_set, 'config_extract.json')

    if os.path.exists(path_config):
        with open(path_config, 'r') as f:
            saved = json.load(f)
        # The saved config always wins — embeddings already on disk were built
        # under it, and honoring a new value here would silently mix framings
        # within one set. Changing extraction params means deleting
        # config_extract.json (and the embeddings) and re-extracting.
        conflicts = {}
        if overlap_event_prop is not None and overlap_event_prop != saved.get('overlap_event_prop'):
            conflicts['overlap_event_prop'] = (saved['overlap_event_prop'], overlap_event_prop)
        if framehop_prop is not None and framehop_prop != saved.get('framehop_prop'):
            conflicts['framehop_prop'] = (saved['framehop_prop'], framehop_prop)
        if conflicts:
            msgs = [f"  {k}: keeping saved={v[0]}, ignoring requested={v[1]}" for k, v in conflicts.items()]
            warnings.warn(
                f"Set '{setname}' already has a config_extract.json; ignoring the "
                f"extraction params passed for it:\n" + "\n".join(msgs) +
                f"\nDelete {path_config} and re-extract to change them."
            )
        config_extract = ConfigExtract(**saved)
        print(f"{time.time()-t0:.1f}s - [{setname}/{embeddername}] existing config: "
              f"overlap_event_prop={saved['overlap_event_prop']}, "
              f"framehop_prop={saved['framehop_prop']}")
        if verbose:
            print(f'{time.time()-t0:.1f}s -   loaded from {path_config}')
    else:
        if overlap_event_prop is None or framehop_prop is None:
            raise ValueError(f"No config_extract.json found for set '{setname}'; overlap_event_prop and framehop_prop must be provided")
        config_extract = ConfigExtract(setname=setname, overlap_event_prop=overlap_event_prop, framehop_prop=framehop_prop)
        os.makedirs(dir_set, exist_ok=True)
        stored = {k: getattr(config_extract, k) for k in ConfigExtract.STORED_FIELDS}
        with open(path_config, 'w') as f:
            json.dump(stored, f, indent=2)
        print(f'{time.time()-t0:.1f}s - [{setname}/{embeddername}] new config: '
              f'overlap_event_prop={overlap_event_prop}, framehop_prop={framehop_prop}')
        if verbose:
            print(f'{time.time()-t0:.1f}s -   saved to {path_config}')

    config_extract.embeddername = embeddername

    folds = pd.read_csv(os.path.join(dir_set, 'folds.csv'))
    annotations = pd.read_csv(os.path.join(dir_set, 'annotations.csv'))
    idents = annotations['ident'].unique()

    dir_snips_base = cfg.dir_snips(setname)
    # Always sync: annotation efforts are ongoing, and an ident whose annotations moved
    # needs its superseded snips replaced before anything is extracted from them. Idents
    # that didn't move cost a manifest read each.
    snip_status = extract_snips(setname, verbose=verbose)
    idents_forced = {i for i, s in snip_status.items() if s == 'updated'}

    # Pre-filter: determine which idents actually need work before spawning workers
    embedder_tmp = load_embedder(embeddername, framehop_prop=1, initialize=False)
    audio_key = embedder_tmp.audio_cache_key()
    dir_audio_cache_base = os.path.join(cfg.dir_audio(setname), audio_key, 'raw')
    dir_embeddings_base = config_extract.dir_out_embeddings(embeddername)

    idents_todo = []
    n_stale = 0
    for ident in idents:
        fold_rows = folds[folds['ident'] == ident]['fold'].unique()
        if len(fold_rows) != 1:
            continue
        fold = str(fold_rows[0])
        fingerprint = _fingerprint_annotations(annotations[annotations['ident'] == ident])
        a_ident = AssignIdent(
            ident=ident, fold=fold, config_extract=config_extract,
            dir_audio_base=dir_audio_cache_base,
            dir_embeddings_base=dir_embeddings_base,
            dir_snips_base=dir_snips_base,
            audio_key=audio_key,
            fingerprint=fingerprint,
            force_stale=ident in idents_forced,
        )
        if a_ident.audio_stale or a_ident.embeddings_stale:
            n_stale += 1
        if a_ident.handle not in ('skip', 'no_snips'):
            idents_todo.append(ident)
        elif a_ident.handle == 'skip':
            # Adopt: nothing has changed under this ident, so record the fingerprint it
            # was built from. Directories extracted before fingerprinting existed can't
            # be judged retroactively; stamping them now makes the *next* edit detectable.
            for dir_done in (a_ident.dir_out_audio, a_ident.dir_out_embeddings):
                if _read_fingerprint(dir_done) is None:
                    _write_fingerprint(dir_done, fingerprint)

    n_skip = len(idents) - len(idents_todo)
    print(f'{time.time()-t0:.1f}s - [{setname}/{embeddername}] {len(idents_todo)} of {len(idents)} idents to extract '
          f'({n_skip} already done or missing snips; {n_stale} superseded by new annotations)')

    if not idents_todo:
        print(f'{time.time()-t0:.1f}s - [{setname}/{embeddername}] nothing to do')
        return True

    if verbose:
        print(f'{time.time()-t0:.1f}s -   {n_workers} worker(s); folds: '
              f"{folds[folds['ident'].isin(idents_todo)]['fold'].nunique()}")

    if n_workers <= 0:
        # In-process extraction: avoids multiprocessing fork, required for embedders
        # that use TF SavedModel (fork corrupts GCD thread pools on macOS).
        worker = WorkerExtract(config_extract, annotations, folds,
                               multiprocessing.Queue(), 'main', verbose=verbose,
                               idents_forced=idents_forced)
        worker.t0 = time.time()
        worker.embedder.initialize()
        for ident in idents_todo:
            worker.extract_ident(ident)
        print(f'{time.time()-t0:.1f}s - [{setname}/{embeddername}] extracted {len(idents_todo)} ident(s)\n:)\n:D\n:O')
        return True

    q_extract = multiprocessing.Queue()
    for i in idents_todo:
        q_extract.put(i)

    workers = []
    for w in range(n_workers):
        q_extract.put('TERMINATE')
        workers.append(multiprocessing.Process(
            target=run_worker,
            args=(config_extract, annotations, folds, q_extract, w, verbose, idents_forced)))

    for w in workers:
        w.start()

    for w in workers:
        w.join()

    failed = [w for w in workers if w.exitcode != 0]
    if failed:
        raise RuntimeError(
            f'{len(failed)}/{len(workers)} extraction worker(s) crashed '
            f'(exit codes: {[w.exitcode for w in failed]})'
        )

    print(f'{time.time()-t0:.1f}s - all extractions complete\n:)\n:D\n:O')
    return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Sync raw audio snips for a set to its annotations (no embedder required).')
    parser.add_argument('--set', required=True, dest='setname')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    extract_snips(args.setname, verbose=args.verbose)
