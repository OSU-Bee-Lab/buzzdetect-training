"""Annotation-audit triage from existing surprisal.csv (IDEAS.md item 7, reduced
scope -- see README note below). No training, no extraction, no new inference.

Reads <model dir>/surprisal/**/*_surprisal.csv (written by 03_train/surprisal.py
during training -- held-out cross-fold activations against the annotation as
written) and ranks two lists across every ident at once:

  A. HARD NEGATIVES / missing annotations -- frames whose label does NOT
     assert ins_buzz, ranked by activation_ins_buzz descending. A high score
     here is either a genuine hard negative or a buzz that never got
     annotated.
  B. HARD POSITIVES / mislabels -- frames whose label DOES assert ins_buzz,
     ranked by activation_ins_buzz ascending. A low score here is either a
     genuine faint/hard positive (expected, esp. at 1_150) or an annotation
     span drawn wider than the sound it covers.

Both lists are restricted to *already-annotated* held-out audio -- see the
scope note. Wall-clock time is parsed from the ident's file-start stem
(YYMMDD_HHMM) plus `start` seconds into that file.

Usage: python diagnostics/2026-09-13_annotation-triage/annotation_triage.py <model dir> [--top 50] [--out csv]
"""
import argparse
import glob
import os
import re
from datetime import datetime, timedelta

import pandas as pd

STEM_RE = re.compile(r'(\d{6})_(\d{4})')


def _wall_clock(path, start_s):
    stem = os.path.basename(path).replace('_surprisal.csv', '')
    m = STEM_RE.match(stem)
    if not m:
        return None
    file_start = datetime.strptime(m.group(1) + m.group(2), '%y%m%d%H%M')
    return file_start + timedelta(seconds=float(start_s))


def load_surprisal(model_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(model_dir, 'surprisal', '**', '*_surprisal.csv'), recursive=True)):
        df = pd.read_csv(path)
        df['ident'] = os.path.relpath(os.path.dirname(path), os.path.join(model_dir, 'surprisal'))
        df['wall_clock'] = [_wall_clock(path, s) for s in df['start']]
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model_dir')
    ap.add_argument('--top', type=int, default=50)
    args = ap.parse_args()

    df = load_surprisal(args.model_dir)
    if df is None:
        raise SystemExit(f'no surprisal csvs under {args.model_dir}/surprisal')

    labels = df['label'].fillna('').astype(str)
    is_buzz = labels.str.contains('ins_buzz')

    cols = ['ident', 'start', 'wall_clock', 'label', 'activation_ins_buzz']

    hard_neg = df[~is_buzz].sort_values('activation_ins_buzz', ascending=False)[cols].head(args.top)
    hard_pos = df[is_buzz].sort_values('activation_ins_buzz', ascending=True)[cols].head(args.top)

    print(f'\n=== List A: hard negatives / missing annotations (top {args.top} by activation_ins_buzz, label != buzz) ===')
    print(hard_neg.to_string(index=False))
    print(f'\nident counts in list A:\n{hard_neg["ident"].value_counts().to_string()}')

    print(f'\n=== List B: hard positives / mislabels (bottom {args.top} by activation_ins_buzz, label == buzz) ===')
    print(hard_pos.to_string(index=False))
    print(f'\nident counts in list B:\n{hard_pos["ident"].value_counts().to_string()}')


if __name__ == '__main__':
    main()
