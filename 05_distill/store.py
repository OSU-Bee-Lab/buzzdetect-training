"""Slice I/O and cache hygiene shared by the distill scripts.

Three ideas, the same ones 02_set uses for its embeddings:

* **Atomic writes, presence means done.** Every npz is written to a temp name and renamed, so a
  killed job leaves no half-file and a rerun skips exactly what exists.
* **A fingerprint beside every derived product.** A cache directory (a front end's mels, a packed
  shard set) carries `fingerprint.json` naming what it was built from. A matching fingerprint means
  the product is current and is reused; a differing one is a stale product, and the caller either
  rebuilds it (cheap, derived data: shards) or stops and says why (expensive: mels, teacher targets).
* **Share at the level of what the product depends on.** A spectrogram depends on the audio and the
  front end, not on the teacher or the student, so it lives in `_mel/<spec>/` and every teacher and
  architecture reads it; teacher targets depend on the teacher and live under it; ffprobe durations
  depend on the file alone.
"""
import csv
import hashlib
import json
import os
import shutil
import sys

import numpy as np

import dpaths as D

MEL_VERSION = 1        # bump when frontends.mel_patches' math changes: invalidates every `_mel/<spec>`


# ---------------------------------------------------------------- hashing / fingerprints

def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_stamp(d):
    p = os.path.join(d, 'fingerprint.json')
    return json.load(open(p)) if os.path.isfile(p) else None


def write_stamp(d, fp, **detail):
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, 'fingerprint.json.tmp')
    json.dump({'fingerprint': fp, **detail}, open(tmp, 'w'), indent=1)
    os.replace(tmp, os.path.join(d, 'fingerprint.json'))


def stamp_or_refuse(d, fp, label, **detail):
    """Adopt `d` if it has no stamp yet, keep it if the stamp matches, stop if it differs.
    For products too costly to rebuild by accident."""
    cur = read_stamp(d)
    if cur is None:
        write_stamp(d, fp, **detail)
    elif cur['fingerprint'] != fp:
        sys.exit(f'{label}: {d} was built from something else ({cur.get("what", cur["fingerprint"])}) than '
                 f'now ({detail.get("what", fp)}). Delete it to rebuild, or rename what changed.')


# ---------------------------------------------------------------- slices

def write_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def write_targets(cache, rel, hour, code, logits, start_s):
    write_npz(D.slice_path(cache, rel, hour), code=code.astype(np.float16),
              logits=logits.astype(np.float32), start_s=np.float64(start_s))


def write_mel(spec, rel, hour, mel):
    write_npz(D.slice_path(D.mel_dir(spec), rel, hour), mel=mel.astype(np.float16))


def has_targets(cache, rel, hour):
    return os.path.exists(D.slice_path(cache, rel, hour))


def mel_path(spec, rel, hour, cache=None):
    """The file holding this slice's mel for `spec`, or None. Search order: the shared `_mel/<spec>`;
    then, for caches built before the shared level existed, `<cache>/_fe/<spec>` and (spec 'yamnet')
    the `mel` embedded in the targets npz."""
    p = D.slice_path(D.mel_dir(spec), rel, hour)
    if os.path.exists(p):
        return p
    cache = cache or D.CACHE
    if spec != 'yamnet':
        legacy = D.slice_path(os.path.join(cache, '_fe', spec), rel, hour)
        return legacy if os.path.exists(legacy) else None
    t = D.slice_path(cache, rel, hour)
    if os.path.exists(t):
        with np.load(t) as z:
            return t if 'mel' in z.files else None
    return None


def load_mel(path):
    with np.load(path) as z:
        return z['mel']


def mel_fingerprint(spec):
    """What a `_mel/<spec>` directory was computed by."""
    if spec == 'yamnet':       # the deployed YAMNet front-end graph, exactly as the first cache used it
        return {'what': 'yamnet frontend_only.onnx', 'sha256': file_sha(D.FRONTEND_ONNX)}
    import dataclasses
    import frontends as fes
    return {'what': f'frontends.{spec} v{MEL_VERSION}', 'spec': dataclasses.asdict(fes.get(spec)),
            'version': MEL_VERSION}


def stamp_mel(spec):
    fp = mel_fingerprint(spec)
    stamp_or_refuse(D.mel_dir(spec), fingerprint(fp), f'front end {spec}', **fp)


# ---------------------------------------------------------------- durations (shared across teachers)

def durations_csv():
    """<distill_cache>/_shared/durations.csv, seeded once from a teacher's own copy so the ffprobe
    sweep of the whole audio tree is never repeated for a second teacher."""
    if not os.path.exists(D.DURATIONS):
        os.makedirs(os.path.dirname(D.DURATIONS), exist_ok=True)
        for t in sorted(os.listdir(D.CACHE_ROOT)):
            legacy = os.path.join(D.CACHE_ROOT, t, '_manifest', 'durations.csv')
            if os.path.exists(legacy):
                shutil.copy2(legacy, D.DURATIONS)
                print(f'[durations] seeded {D.DURATIONS} from {legacy}', flush=True)
                break
    return D.DURATIONS


def load_durations():
    """{relpath: (size, duration_s)} for every probeable file."""
    p = durations_csv()
    if not os.path.exists(p):
        return {}
    return {r['relpath']: (int(r['size']), float(r['dur'])) for r in csv.DictReader(open(p)) if r['dur']}


# ---------------------------------------------------------------- teacher identity

def teacher_sha():
    return file_sha(D.TEACHER_ONNX)


def require_current_teacher():
    """Stop if teacher.json was written for a different teacher ONNX than the one on disk (a retrained
    model re-shipped under the same name would otherwise silently mix old and new targets)."""
    if not os.path.isfile(D.TEACHER_JSON):
        sys.exit(f'{D.TEACHER_JSON} missing: run teacher_onnx.py first')
    built = json.load(open(D.TEACHER_JSON))['onnx_sha256']
    if built != teacher_sha():
        sys.exit(f'teacher {D.TEACHER}: model.onnx changed since this cache was built ({built[:12]} -> '
                 f'{teacher_sha()[:12]}). The cached targets are stale: move {D.CACHE} aside and rebuild, '
                 f'or name the new teacher differently.')
