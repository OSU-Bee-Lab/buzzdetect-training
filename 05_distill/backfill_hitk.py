"""Add `buzz_hitk` (teacher-hit at K, distill_train.hit_at_k) to finished runs that predate it.

    backfill_hitk.py [--names a,b,...] [--redo] [--dry-run]

Re-scores each run's saved student on the full V pool and writes the value into curve.json's final val
entry; nothing else in curve.json changes. The fresh pass also re-counts the teacher's buzz detections and
the student's misses: a mismatch with the recorded counts is printed (the saved model is the run's last
weights, so they should agree up to GPU nondeterminism near logit 0). The frame count and the teacher's
buzz count do not depend on the student: if either differs, the run was scored on a different V pool and
nothing is written (2026-10-04: the 2026-10-02 re-plan swapped 22 V slices, the old fingerprint-less packs
kept the old pool, packs rebuilt since have the new one; the frame count matched by coincidence).
Runs named test_* are left alone.
Default order: WSD branches, then the rung-A repeats (the noise spread), then the rest. Train env, GPU.
"""
import argparse
import json
import os
import sys

import tensorflow as tf  # noqa: F401  (must come first, see CLAUDE.md)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import distill_train as dt  # noqa: E402
import dpaths as D  # noqa: E402
import student  # noqa: E402,F401  (registers the student's layers for load_model)


def final(cur):
    return [x for x in cur['val'] if x.get('final')][-1]


def todo(names, redo=False):
    out = []
    for n in names or sorted(os.listdir(D.RUNS)):
        d = os.path.join(D.RUNS, n)
        if n.startswith('test_') or not os.path.exists(os.path.join(d, 'TRAIN_DONE')):
            continue
        if redo or 'buzz_hitk' not in final(json.load(open(os.path.join(d, 'curve.json')))):
            out.append(n)
    return sorted(out, key=lambda n: (0 if '_wsd' in n else 1 if n.startswith('lad_A') else 2, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', default='')
    ap.add_argument('--redo', action='store_true', help='re-score runs that already have buzz_hitk')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    names = todo([n for n in a.names.split(',') if n], a.redo)
    print(f'[hitk] {len(names)} runs to score: {" ".join(names)}', flush=True)
    if a.dry_run:
        return
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    cache, vals, bad = dt.cache_root(), {}, 0
    for n in names:
        d = os.path.join(D.RUNS, n)
        path = os.path.join(d, 'curve.json')
        cur = json.load(open(path))
        args = cur.get('args', {})
        fe = args.get('frontend', 'yamnet')
        dt.set_active(args['classes'].split(',') if args.get('classes') else None)
        if fe not in vals:
            vals[fe] = dt.Shards(dt.pack(cache, 'V', frontend=fe))
        model = tf.keras.models.load_model(os.path.join(d, 'student_mel.keras'), compile=False)
        v = dt.val_flips(model, vals[fe])
        f = final(cur)
        diff = {k: (f[k], v[k]) for k in ('frames', 'buzz_teacher', 'buzz_lost') if f.get(k) != v[k]}
        if 'frames' in diff or 'buzz_teacher' in diff:
            print(f'[hitk] {n}: NOT WRITTEN, V pool differs from the recorded one {diff}', flush=True)
            bad += 1
            tf.keras.backend.clear_session()
            continue
        f['buzz_hitk'] = v['buzz_hitk']
        tmp = path + '.tmp'
        json.dump(cur, open(tmp, 'w'))
        os.replace(tmp, path)
        print(f'[hitk] {n}: hit@K {v["buzz_hitk"]}/{v["buzz_teacher"]} = '
              f'{100 * v["buzz_hitk"] / max(1, v["buzz_teacher"]):.2f}%  (hit at 0: '
              f'{100 - 100 * f["buzz_lost"] / max(1, f["buzz_teacher"]):.2f}%)'
              + (f'  RECOUNT DIFFERS {diff}' if diff else ''), flush=True)
        tf.keras.backend.clear_session()
    print(f'[hitk] done; {bad} not written (V pool differs)', flush=True)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
