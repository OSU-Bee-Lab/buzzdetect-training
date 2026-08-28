"""
Extract 5-minute clips from source MP3s using pre-computed offsets.

Input:  data/01_snips.csv  (columns: path, ident, start_datetime, start_filetime)
Output: data/audio/<ident-parent>/<ident-name>_s<start_filetime>.flac

Clips are written as 16-bit FLAC. The sources are low-bitrate MP3s, so 16-bit is
transparent (its quantization noise sits ~96 dB down, far below MP3 artifacts).

This is where clip duration is actually verified. 01_build_snips.R estimates run
time from file size (a coarse bitrate guess) only to pick hour offsets; here we
seek into the real file and read. If a clip falls more than SHORT_TOLERANCE_S
short of the target, we WARN unconditionally (even without --verbose) and still
write whatever audio we got, leaving it to the user to decide what to do.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIR_OUTPUT = PROJECT_ROOT / 'data' / 'audio'
SNIPS_CSV = PROJECT_ROOT / 'data' / '01_snips.csv'

CLIP_DURATION_S = 5 * 60
SHORT_TOLERANCE_S = 10  # warn if a clip falls more than this short of the target


def _extract(path: str, offset_s: float, output_path: Path, verbose: bool) -> None:
    try:
        with sf.SoundFile(str(path)) as f:
            sr = f.samplerate
            expected = int(CLIP_DURATION_S * sr)
            f.seek(int(offset_s * sr))
            data = f.read(expected, dtype='float32', always_2d=False)
    except Exception as e:
        print(f'  ERROR: {output_path.name}: {e}')
        return

    if len(data) == 0:
        print(f'  ERROR: {output_path.name}: no audio at offset {offset_s}s (past end of file?)')
        return

    got_s = len(data) / sr
    if got_s < CLIP_DURATION_S - SHORT_TOLERANCE_S:
        # Unconditional warning — surfaces even without --verbose.
        print(f'  WARNING: {output_path.name}: only {got_s:.1f}s '
              f'(target {CLIP_DURATION_S}s); writing partial clip')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), data, sr)  # FLAC -> 16-bit PCM by default
    if verbose:
        print(f'  -> {output_path.relative_to(DIR_OUTPUT)} ({got_s:.1f}s)')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    if not SNIPS_CSV.exists():
        sys.exit(f'ERROR: {SNIPS_CSV} not found')

    df = pd.read_csv(SNIPS_CSV)

    current_ident = None
    for _, row in df.iterrows():
        ident = Path(row['ident'])
        if str(ident) != current_ident:
            current_ident = str(ident)
            print(f'starting {ident}')
        out_path = DIR_OUTPUT / ident.parent / f"{ident.name}_s{int(row['start_filetime'])}.flac"
        if out_path.exists() and not args.overwrite:
            if args.verbose:
                print(f'  skip (exists): {out_path.relative_to(DIR_OUTPUT)}')
            continue
        _extract(row['path'], row['start_filetime'], out_path, args.verbose)

    print('\nDone.')


if __name__ == '__main__':
    main()
