"""Tests for the teacher-generic layer of 05_distill: layout, fingerprints, shared-cache lookups, the
migration, main.py's stage bookkeeping and (with --train) training's checkpoint/resume. Everything runs
against a scratch tree via the DISTILL_* overrides in dpaths.py; the real data and the GPU are never touched.

    conda run -n buzzdetect-train python 05_distill/test_distill.py            # seconds, no TensorFlow
    conda run -n buzzdetect-train python 05_distill/test_distill.py --train    # + resume test, ~2 min on CPU
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp(prefix='distill_test_')
CLASSES = ['aambient_scraping', 'ambient_background', 'ambient_music', 'ambient_noise', 'ambient_rain',
           'ambient_thunder', 'animal', 'human', 'ins_buzz', 'ins_trill', 'mech_auto', 'mech_hum',
           'mech_machinery', 'mech_plane', 'mech_quadcopter']
ENV = {'DISTILL_TEACHER': 'test_teacher', 'DISTILL_CACHE_ROOT': f'{TMP}/cache', 'DISTILL_LOCAL_ROOT': f'{TMP}/local',
       'DISTILL_MODELS_DIR': f'{TMP}/models', 'DISTILL_ENGINE_MODELS': f'{TMP}/engine', 'CUDA_VISIBLE_DEVICES': ''}
os.environ.update(ENV)
sys.path.insert(0, HERE)

os.makedirs(f'{TMP}/models/test_teacher')
json.dump({'classes': CLASSES, 'set': 'moderate', 'translation': 'general', 'folds_train': [],
           'activation_centers': {c: 0.5 for c in CLASSES if c not in ('aambient_scraping', 'mech_quadcopter')}},
          open(f'{TMP}/models/test_teacher/config_model.json', 'w'))
os.makedirs(f'{TMP}/cache/test_teacher/_manifest')
onnx = f'{TMP}/teacher.onnx'
open(onnx, 'wb').write(b'teacher v1')

import dpaths as D  # noqa: E402
import store  # noqa: E402

D.TEACHER_ONNX = onnx
json.dump({'onnx_sha256': store.file_sha(onnx), 'code_dim': 2048}, open(D.TEACHER_JSON, 'w'))
FAILED = []


def check(name, cond):
    print(('ok   ' if cond else 'FAIL ') + name)
    if not cond:
        FAILED.append(name)


def raises_exit(fn):
    try:
        fn()
    except SystemExit:
        return True
    return False


def test_layout_and_spec():
    check('cache is <root>/<teacher>', D.CACHE == f'{TMP}/cache/test_teacher')
    check('mels are shared above the teacher', D.mel_dir('fast32') == f'{TMP}/cache/_mel/fast32')
    check('local data is per teacher, arch timings shared',
          D.RUNS == f'{TMP}/local/test_teacher/runs' and D.ARCH == f'{TMP}/local/_shared/arch')
    sp = D.spec()
    check('classes without a center are dead', sp.dead == [0, 14] and sp.live == list(range(1, 14)))
    check('buzz index found by name', sp.buzz == 8 and sp.n_classes == 15 and sp.code_dim == 2048)
    check('slice path mirrors the audio tree',
          D.slice_path('/c', 'proj/dep/rec/x.mp3', 3) == '/c/proj/dep/rec/x/h000003.npz')


def test_fingerprints():
    d = f'{TMP}/fp'
    store.stamp_or_refuse(d, 'aaa', 'thing', what='v1')
    store.stamp_or_refuse(d, 'aaa', 'thing', what='v1')                      # same: kept
    check('matching stamp is kept', store.read_stamp(d)['fingerprint'] == 'aaa')
    check('a different stamp stops the run', raises_exit(lambda: store.stamp_or_refuse(d, 'bbb', 'thing', what='v2')))
    check('fingerprint ignores key order', store.fingerprint({'a': 1, 'b': 2}) == store.fingerprint({'b': 2, 'a': 1}))
    store.require_current_teacher()
    open(onnx, 'wb').write(b'teacher v2 retrained')
    check('a re-shipped teacher ONNX is caught', raises_exit(store.require_current_teacher))
    meta = json.load(open(D.TEACHER_JSON))
    json.dump(dict(meta, onnx_sha256_equivalent=[store.file_sha(onnx)]), open(D.TEACHER_JSON, 'w'))
    check('an adopted re-export keeps the original identity', store.teacher_sha() == meta['onnx_sha256'])
    store.require_current_teacher()
    json.dump(meta, open(D.TEACHER_JSON, 'w'))
    open(onnx, 'wb').write(b'teacher v1')


def test_mel_lookup():
    rel, h = 'proj/dep/rec/x.mp3', 7
    cache = D.CACHE
    check('no mel anywhere', store.mel_path('yamnet', rel, h) is None and store.mel_path('fast32', rel, h) is None)
    mel = np.ones((2, 96, 64), np.float16)
    store.write_targets(cache, rel, h, np.zeros((2, 4)), np.zeros((2, 15)), 0.0)
    check('targets without mel still have none', store.mel_path('yamnet', rel, h) is None)
    np.savez(D.slice_path(cache, rel, h), mel=mel, code=np.zeros((2, 4), np.float16),
             logits=np.zeros((2, 15), np.float32))                      # a pre-split cache embeds mel
    check('embedded mel is found for yamnet', store.mel_path('yamnet', rel, h) == D.slice_path(cache, rel, h))
    legacy = D.slice_path(os.path.join(cache, '_fe', 'fast32'), rel, h)
    store.write_npz(legacy, mel=mel)
    check('pre-split _fe is found', store.mel_path('fast32', rel, h) == legacy)
    store.write_mel('fast32', rel, h, mel)
    check('the shared level wins', store.mel_path('fast32', rel, h) == D.slice_path(D.mel_dir('fast32'), rel, h))
    check('mel round-trips', np.array_equal(store.load_mel(store.mel_path('fast32', rel, h)), mel))


def test_durations_seed():
    os.makedirs(f'{D.CACHE_ROOT}/other_teacher/_manifest', exist_ok=True)
    open(f'{D.CACHE_ROOT}/other_teacher/_manifest/durations.csv', 'w').write('relpath,size,dur\na/b.mp3,100,3600.5\nc.mp3,5,\n')
    check('durations seed from another teacher once', store.load_durations() == {'a/b.mp3': (100, 3600.5)})
    open(f'{D.CACHE_ROOT}/other_teacher/_manifest/durations.csv', 'w').write('relpath,size,dur\n')
    check('the shared copy is then authoritative', 'a/b.mp3' in store.load_durations())


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **ENV}, **kw)


def test_migrate():
    L = D.LOCAL_ROOT
    for d in ('runs/r1', 'models/m1', 'eval/e1', 'shards/B', 'arch/frontend_only'):
        os.makedirs(f'{L}/{d}')
    open(f'{L}/ladder.jsonl', 'w').write('{"name": "r1"}\n')
    os.makedirs(f'{D.CACHE}/_fe/lo32/p')
    r = run([sys.executable, f'{HERE}/migrate_layout.py', '--apply'])
    check('apply refuses while a real chain is alive (skipped when none is)', r.returncode == 0 or 'wait for them' in r.stdout + r.stderr)
    r = run([sys.executable, f'{HERE}/migrate_layout.py'])
    check('dry run moves nothing', r.returncode == 0 and os.path.isdir(f'{L}/runs/r1') and 'planned' in r.stdout)
    r = run([sys.executable, f'{HERE}/migrate_layout.py', '--apply', '--force'])
    check('apply moves the local data under the teacher', r.returncode == 0 and os.path.isdir(f'{L}/test_teacher/runs/r1')
          and os.path.isfile(f'{L}/test_teacher/ladder.jsonl') and not os.path.exists(f'{L}/runs'))
    check('arch timings go to _shared', os.path.isdir(f'{L}/_shared/arch/frontend_only'))
    # test_mel_lookup left a fast32 in both places: the collision is kept, not clobbered
    check('_fe specs move to the shared _mel, a collision is left alone',
          os.path.isdir(f'{D.CACHE_ROOT}/_mel/lo32/p') and not os.path.exists(f'{D.CACHE}/_fe/lo32')
          and os.path.isdir(f'{D.CACHE}/_fe/fast32'))
    r = run([sys.executable, f'{HERE}/migrate_layout.py', '--apply', '--force'])
    check('a second apply is a no-op', r.returncode == 0 and '0 move(s)' in r.stdout)


def test_main_stages():
    os.makedirs(os.path.dirname(D.FRONTEND_ONNX), exist_ok=True)
    open(D.FRONTEND_ONNX, 'wb').write(b'x')
    open(D.PLAN, 'w').write('relpath,hour\n')
    r = run([sys.executable, f'{HERE}/main.py', '--runs', 'yamnet:a0.50:select fast32:a0.375', '--dry-run'])
    out = r.stdout
    check('dry run lists every stage', r.returncode == 0 and 'fe_B_yamnet_a0.50_s1_select train' in out
          and 'fe_B_fast32_a0.375_s1 speed' in out and 'cache_fe B fast32' in out)
    check('finished teacher stages are reported done', '[done     ] plan' in out and '[done     ] frontend_only.onnx' in out)
    run_dir = f'{D.RUNS}/fe_B_yamnet_a0.50_s1_select'
    os.makedirs(run_dir, exist_ok=True)
    open(f'{run_dir}/TRAIN_DONE', 'w').close()
    out = run([sys.executable, f'{HERE}/main.py', '--runs', 'yamnet:a0.50:select', '--dry-run']).stdout
    check('a finished stage is skipped on rerun', '[done     ] fe_B_yamnet_a0.50_s1_select train' in out
          and '[pending  ] fe_B_yamnet_a0.50_s1_select export' in out)


def test_main_wsd():
    """WSD stage list: trunk segments interleaved with decayed branches, budgets in steps or passes,
    --wsd-eval limiting the judged branches, and --wsd-stop leaving out what follows a plateau."""
    plan = open(D.PLAN).read()
    open(D.PLAN, 'w').write('relpath,hour,first_rung,n_frames\na,0,A,5120\nb,0,B,5120\nc,0,C,99\nv,0,V,7\n')
    t = 'fe_B_yamnet_a0.25_s1'
    base = [sys.executable, f'{HERE}/main.py', '--runs', 'yamnet:a0.25', '--dry-run']
    out = run(base + ['--wsd', '100,200,400', '--wsd-eval', '400']).stdout
    order = [f'{t}_wsd trunk to 85', f'{t}_wsd100 train (decay 85-100)', f'{t}_wsd trunk to 170',
             f'{t}_wsd200 train (decay 170-200)', f'{t}_wsd trunk to 340', f'{t}_wsd400 train (decay 340-400)',
             f'{t}_wsd400 export', f'{t}_wsd curve']
    pos = [out.find(s) for s in order]
    check('wsd stages interleave trunk and branches', -1 not in pos and pos == sorted(pos))
    check('--wsd-eval judges only the listed budgets', f'{t}_wsd100 export' not in out and f'{t}_wsd400 record' in out)
    out = run(base + ['--wsd-max', '400', '--wsd-halvings', '2', '--wsd-eval', '400']).stdout
    check('--wsd-max generates the ceiling and its halvings', [out.find(s) for s in order] == pos)
    out = run(base + ['--wsd', '1p,2p']).stdout
    check('budgets in passes resolve on the plan (rung B = 10240 frames, batch 512)',
          f'{t}_wsd20 train (decay 17-20)' in out and f'{t}_wsd40 train (decay 34-40)' in out)
    r = run(base + ['--passes', '1'])
    check('--passes sets the cosine steps', '= 20 steps' in r.stdout)
    # hit@K 90.0 then 90.1 (below a 1.3 tolerance) while hit% at logit 0 jumps 70 -> 90: the rule reads hit@K
    for b, lost, hitk in ((100, 300, 900), (200, 100, 901)):
        d = f'{D.RUNS}/{t}_wsd{b}'
        os.makedirs(d, exist_ok=True)
        json.dump({'val': [{'final': True, 'buzz_lost': lost, 'buzz_teacher': 1000, 'buzz_hitk': hitk}]},
                  open(f'{d}/curve.json', 'w'))
        open(f'{d}/TRAIN_DONE', 'w').close()
    out = run(base + ['--wsd', '100,200,400', '--wsd-stop', '1.3']).stdout
    check('--wsd-stop: after a hit@K plateau the trunk goes no further',
          f'[plateau  ] {t}_wsd trunk to 340' in out and f'[plateau  ] {t}_wsd400 train' in out
          and f'[pending  ] {t}_wsd trunk to 170' in out)
    out = run(base + ['--wsd', '100,200,400', '--wsd-stop', '0.05']).stdout
    check('--wsd-stop: a gain above the tolerance keeps going', f'[pending  ] {t}_wsd trunk to 340' in out)
    json.dump({'val': [{'final': True, 'buzz_lost': 100, 'buzz_teacher': 1000}]},
              open(f'{D.RUNS}/{t}_wsd200/curve.json', 'w'))
    r = run(base + ['--wsd', '100,200,400', '--wsd-stop', '1.3'])
    check('--wsd-stop: a branch without hit@K is refused, not read as no plateau',
          r.returncode != 0 and 'backfill_hitk.py' in r.stdout + r.stderr)
    open(D.PLAN, 'w').write(plan)


def test_cache_fe():
    """cache_fe.py end to end on a synthetic wav in the onnx venv: writes the shared mel, skips on rerun,
    stops when the stored fingerprint no longer matches the spec."""
    import shutil
    import wave
    if not (os.path.exists(D.ONNX_PY) and shutil.which('ffmpeg')):
        print('skip cache_fe (no onnx venv or ffmpeg)')
        return
    audio = f'{TMP}/audio'
    rel = 'proj/dep/rec/a.wav'
    os.makedirs(f'{audio}/proj/dep/rec')
    x = (np.random.default_rng(0).normal(0, 0.05, 16000 * 61) * 32767).astype(np.int16)
    with wave.open(f'{audio}/{rel}', 'wb') as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000), w.writeframes(x.tobytes())
    open(D.PLAN, 'w').write('relpath,hour,start_s,deployment,rank,first_rung,n_frames\n'
                            f'{rel},0,0.0,proj/dep,0.1,A,62\n')
    store.write_targets(D.CACHE, rel, 0, np.zeros((62, 4)), np.zeros((62, 15)), 0.0)
    env = {**ENV, 'DISTILL_AUDIO_ROOT': audio}
    cmd = [D.ONNX_PY, f'{HERE}/cache_fe.py', '--rung', 'A', '--frontends', 'fast32', '--workers', '1']
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env})
    out = D.slice_path(D.mel_dir('fast32'), rel, 0)
    check('cache_fe writes the shared mel', r.returncode == 0 and os.path.exists(out)
          and store.load_mel(out).shape == (62, 96, 32, 1))
    check('cache_fe stamps the spec dir', store.read_stamp(D.mel_dir('fast32')) is not None)
    r2 = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env})
    check('a rerun has nothing to do', r2.returncode == 0 and '0 to do' in r2.stdout)
    store.write_stamp(D.mel_dir('fast32'), 'stale', what='an older definition')
    r3 = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env})
    check('a stale stamp stops cache_fe', r3.returncode != 0 and 'built from something else' in r3.stdout + r3.stderr)
    if r.returncode:
        print(r.stdout[-1200:], r.stderr[-1200:])


def test_resume():
    """Kill training after its first checkpoint, rerun: it continues, finishes, and a third run is a no-op."""
    import tensorflow as tf  # noqa: F401
    import student as st
    import student_init as si
    import test_synth as ts
    n_tr, n_va = 640, 320
    mel = si.mel_of(si.frames_from_folds(n_tr + n_va, seed=3))
    full = st.build_student(st.widths_for(1.0), input_type='mel', expose_code=True)
    si.init_from_yamnet(full, si.load_yamnet())
    rng = np.random.default_rng(0)
    head = (rng.normal(0, 0.15, (1024, 15)).astype(np.float32), rng.normal(0, 1, 15).astype(np.float32))
    ts.write(f'{TMP}/synth/train', mel[:n_tr], full, head)
    ts.write(f'{TMP}/synth/val', mel[n_tr:], full, head)
    base = [sys.executable, f'{HERE}/distill_train.py', '--rung', 'A', '--steps', '40', '--batch', '64', '--eval-every', '20',
            '--arch', 'a0.25', '--init', 'select', '--init-frames', '256', '--name', 'resume_test', '--val-limit', '256',
            '--shards', f'{TMP}/synth/train', '--val-shards', f'{TMP}/synth/val']
    run_dir = f'{D.RUNS}/resume_test'
    r1 = run(base + ['--halt-at', '20'])
    check('first run stops after its checkpoint', r1.returncode == 3 and os.path.exists(f'{run_dir}/ckpt.npz')
          and not os.path.exists(f'{run_dir}/TRAIN_DONE'))
    r2 = run(base)
    check('second run resumes at step 20 and finishes', r2.returncode == 0 and 'continuing from step 20 of 40' in r2.stdout
          and os.path.exists(f'{run_dir}/TRAIN_DONE') and not os.path.exists(f'{run_dir}/ckpt.npz'))
    curve = json.load(open(f'{run_dir}/curve.json'))
    check('curve spans the whole run', [v['step'] for v in curve['val']] == [0, 20, 40]
          and curve['train'][-1]['step'] == 40)
    r3 = run(base)
    check('third run is a no-op', r3.returncode == 0 and 'already trained' in r3.stdout)
    r4 = run([c if c != '64' else '32' for c in base])
    check('same name, other settings stops', r4.returncode != 0 and 'other settings' in (r4.stdout + r4.stderr))
    if r2.returncode:
        print(r2.stdout[-1500:], r2.stderr[-1500:])

    # warmup-stable-decay: trunk in segments (killed mid-stable once), a decayed branch, then extension
    wsd = [c if c != 'resume_test' else 'wsd_test' for c in base]
    wsd = wsd[:wsd.index('--steps') + 1] + ['40'] + wsd[wsd.index('--steps') + 2:] + \
        ['--schedule', 'wsd', '--warmup', '5', '--eval-every', '10', '--branch-at', '20']
    tdir = f'{D.RUNS}/wsd_test'
    r = run(wsd + ['--stop-at', '20', '--halt-at', '10'])
    check('wsd trunk killed mid-stable leaves a checkpoint', r.returncode == 3 and os.path.exists(f'{tdir}/ckpt.npz'))
    r = run(wsd + ['--stop-at', '20'])
    check('wsd trunk resumes and pauses at its branch point', r.returncode == 0 and 'continuing from step 10' in r.stdout
          and os.path.exists(f'{tdir}/branches/s20.npz') and not os.path.exists(f'{tdir}/TRAIN_DONE'))
    br = [c if c != 'wsd_test' else 'wsd_test24' for c in wsd]
    del br[br.index('--branch-at'):br.index('--branch-at') + 2]
    br[br.index('--steps') + 1] = '24'
    r = run(br + ['--decay-from', 'wsd_test:20'])
    cb = json.load(open(f'{D.RUNS}/wsd_test24/curve.json')) if r.returncode == 0 else {}
    lr_end = cb.get('train', [{}])[-1].get('lr', -1)        # step 24 used lr at k = 3/4: 1e-3 (1 - sqrt(.75))
    check('decay branch starts from the trunk and anneals to its budget',
          r.returncode == 0 and '[branch]' in r.stdout and cb['branched_at'] == 20
          and [v['step'] for v in cb['val']] == [0, 10, 20, 24] and abs(lr_end - 1e-3 * (1 - 0.75 ** 0.5)) < 1e-7)
    if r.returncode:
        print(r.stdout[-1500:], r.stderr[-1500:])
    r = run([c if c != '1e-3' else '2e-3' for c in br] + ['--decay-from', 'wsd_test:20', '--lr', '2e-3'])
    check('a branch with other settings than its trunk stops', r.returncode != 0 and 'other settings' in r.stdout + r.stderr)
    r = run(wsd)
    ct = json.load(open(f'{tdir}/curve.json'))
    check('wsd trunk finishes at a constant rate and keeps its last point', r.returncode == 0
          and os.path.exists(f'{tdir}/TRAIN_DONE') and os.path.exists(f'{tdir}/branches/s40.npz')
          and abs(ct['train'][-1]['lr'] - 1e-3) < 1e-9)
    ext = list(wsd)
    ext[ext.index('--steps') + 1] = '50'
    r = run(ext)
    check('a finished trunk extends to a larger --steps', r.returncode == 0 and '[extend]' in r.stdout
          and os.path.exists(f'{tdir}/branches/s50.npz'))

    # class subset: a 3-output head, trained, exported with the config listing only those classes
    import shutil
    real_cfg = os.path.join(D.config.local('buzzdetect_dest'), D.DEFAULT_TEACHER, 'config_model.json')
    os.makedirs(D.TEACHER_ENGINE_DIR, exist_ok=True)
    shutil.copy(real_cfg, os.path.join(D.TEACHER_ENGINE_DIR, 'config_model.json'))
    keep = ['ins_buzz', 'ambient_rain', 'human']
    cls = [c if c != 'resume_test' else 'cls_test' for c in base]
    r5 = run(cls + ['--classes', ','.join(keep), '--lam', '0'])
    curve = json.load(open(f'{D.RUNS}/cls_test/curve.json'))
    m = tf.keras.models.load_model(f'{D.RUNS}/cls_test/student_mel.keras', compile=False)
    check('class-subset training builds a 3-output head', r5.returncode == 0 and m.get_layer('logits').units == 3
          and curve['args']['keep_classes'] == keep and 'classes [' in r5.stdout)
    r6 = run([sys.executable, f'{HERE}/export_student.py', 'export', '--run', 'cls_test', '--name', 'cls_test'])
    cfg_path = f'{D.MODELS}/cls_test/config_model.json'
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    check('export writes a config with only those classes', r6.returncode == 0 and cfg.get('classes') == keep
          and 'parity' in r6.stdout and 'PASS' in r6.stdout)
    import onnx
    from onnx import numpy_helper
    kb = m.get_layer('logits').get_weights()[1]
    inits = [numpy_helper.to_array(i) for i in onnx.load(f'{D.MODELS}/cls_test/model.onnx').graph.initializer] \
        if r6.returncode == 0 else []
    check('export keeps the trained head bias (targets are already on the deployed scale, no center shift)',
          any(i.shape == kb.shape and np.allclose(i, kb, atol=1e-5) for i in inits))
    if r5.returncode or r6.returncode:
        print(r5.stdout[-800:], r5.stderr[-800:], r6.stdout[-800:], r6.stderr[-1200:])


if __name__ == '__main__':
    test_layout_and_spec()
    test_fingerprints()
    test_mel_lookup()
    test_durations_seed()
    test_migrate()
    test_main_stages()
    test_main_wsd()
    test_cache_fe()
    if '--train' in sys.argv:
        test_resume()
    print(f'\n{len(FAILED)} failure(s)' if FAILED else '\nall passed', f'(scratch: {TMP})')
    sys.exit(1 if FAILED else 0)
