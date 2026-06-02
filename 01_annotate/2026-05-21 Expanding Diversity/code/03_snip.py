"""Extract snip regions defined by 02_selections_*.csv from source mp3 audio.

Each source mp3 is opened once with soundfile.SoundFile and walked with a
single forward read head: snips are sorted by start time, and the gap between
snips is consumed by reading-and-discarding samples rather than seeking from
the file origin. This mirrors the sequential-readhead approach in
/Users/luke/Tools/night-positives/extract_night_positives.py.

Output is FLAC at the source file's native samplerate:

    seenote/<project>/<ident parent dirs>/<ident leaf>_s<start>.flac

<project> is taken from the CSV filename (e.g. `02_selections_even.csv`
-> `even`). Idents are processed in parallel across `threads` worker threads;
each thread owns one ident (and one open SoundFile) at a time.
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_ROOT = PROJECT_ROOT / '..' / '..' / 'audio'
SEENOTE_ROOT = PROJECT_ROOT / 'seenote'

DEFAULT_THREADS = 6


def project_from_csv(path: Path) -> str:
    return re.sub(r'^02_selections_', '', path.stem)


def _process_ident(
    ident: str,
    group: pd.DataFrame,
    audio_root: Path,
    output_root: Path,
    project: str,
) -> None:
    src = audio_root / f'{ident}.mp3'
    if not src.exists():
        print(f'  MISSING source: {src}  ({len(group)} row(s) skipped)')
        return

    ident_path = Path(ident)
    out_dir = output_root / project / 'audio' / ident_path.parent

    try:
        track = sf.SoundFile(str(src))
    except Exception as e:
        print(f'  cannot open {src}: {e}')
        return

    try:
        sr = track.samplerate
        n_frames = track.frames
        # Process snips in start order so the readhead only moves forward.
        ordered = group.sort_values('start')
        print(f'  {ident}  ({len(ordered)} snip(s), {sr} Hz)')

        head = 0  # current readhead position in frames
        skip_chunk = sr * 30  # discard up to 30s per read while skipping gaps

        for row in ordered.itertuples(index=False):
            start = int(row.start)
            end = int(row.end)
            duration = end - start
            if duration <= 0:
                print(f'    skip (bad duration): [{start},{end}]')
                continue

            out_path = out_dir / f'{ident_path.name}_s{start}.flac'
            target_frame = start * sr
            n = duration * sr

            if target_frame < head:
                print(f'    skip (overlaps previous snip): [{start},{end}]')
                continue
            if target_frame >= n_frames:
                print(f'    skip (start past EOF): [{start},{end}]')
                continue

            # Walk forward to the snip start by reading-and-discarding.
            gap = target_frame - head
            discarded = None
            while gap > 0:
                chunk = min(gap, skip_chunk)
                try:
                    discarded = track.read(chunk)
                except Exception as e:
                    print(f'    error skipping to {start}s: {e}')
                    discarded = None
                    break
                if len(discarded) == 0:
                    break  # EOF
                head += len(discarded)
                gap -= len(discarded)
            if gap > 0 and discarded is None:
                continue

            if out_path.exists():
                # Still must advance the readhead past this region so subsequent snips align.
                try:
                    passed = track.read(n)
                    head += len(passed)
                except Exception as e:
                    print(f'    error advancing past existing {out_path.name}: {e}')
                print(f'    skip (exists): {out_path.name}')
                continue

            try:
                samples = track.read(n)
            except Exception as e:
                print(f'    error reading {start}s: {e}')
                continue
            head += len(samples)

            if len(samples) < n:
                pad_shape = list(samples.shape)
                pad_shape[0] = n - len(samples)
                samples = np.concatenate([samples, np.zeros(pad_shape, dtype=samples.dtype)])

            out_dir.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), samples, sr)
            print(f'    [{start}..{end}] -> {out_path.name}')
    finally:
        track.close()


def snip_selections(
    selections_csv: Path,
    audio_root: Path = AUDIO_ROOT,
    output_root: Path = SEENOTE_ROOT,
    threads: int = DEFAULT_THREADS,
) -> None:
    selections_csv = Path(selections_csv)
    project = project_from_csv(selections_csv)
    df = pd.read_csv(selections_csv)

    missing_required = {'ident', 'start', 'end'} - set(df.columns)
    if missing_required:
        raise ValueError(f'{selections_csv} missing columns: {missing_required}')

    groups = [(str(ident), g) for ident, g in df.groupby('ident', sort=False)]
    print(f'\n{selections_csv}  ->  project "{project}"  '
          f'({len(df)} rows, {len(groups)} idents, {threads} threads)')

    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [
            pool.submit(_process_ident, ident, g, audio_root, output_root, project)
            for ident, g in groups
        ]
        for fut in as_completed(futures):
            fut.result()


if __name__ == '__main__':
    for name in ('02_selections_even.csv', '02_selections_night.csv'):
        snip_selections(PROJECT_ROOT / 'data' / name)
