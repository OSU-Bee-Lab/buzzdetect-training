"""Distil the teacher into the width-alpha student, on the cached teacher targets.

    tools/launch_job.sh <log> -- conda run -n buzzdetect-train python \
        05_distill/distill_train.py --rung A --steps 6000 --name a05_A     # teacher: DISTILL_TEACHER / main.py --teacher

Resumable: a checkpoint (weights, BN statistics, optimizer state, curve) is written at every
--eval-every steps to runs/<name>/ckpt.npz, a rerun continues from it, and TRAIN_DONE marks a
finished run (a rerun is then a no-op). Same name with other settings stops.

Data: `<cache>/_manifest/plan.csv` says which slices belong to a rung (first_rung
<= rung in A<B<C<D; `V` is the held-out validation deployments). Per slice, the targets npz
(code f16 (n,D), logits f32 (n,C)) and the front end's mel (shared `_mel/<frontend>`, or the
`mel` a pre-split cache embeds) are packed once into contiguous local shards
`.local/distill/<teacher>/shards/<rung>[__<frontend>]/{mel,code,logits}.npy`
(float16 memmaps, written in plan order so the HDD is read sequentially; reused while
meta.json's fingerprint of teacher, plan, front end and slices present still matches) and
training reads those. Batches are frame-level random draws from a full
permutation, so a shuffle buffer is not needed; on-disk shards stay page-cached.

Model: student.build_student(widths_for(alpha), input_type='mel'), initialised
from YAMNet by channel selection (student_init), BN moments re-measured on
training mel, head and aux map ridge-fit to the targets so step 0 already tracks
the teacher. Loss = Huber(logits; classes 0 and 14 weight 0) + lam * MSE(aux(code),
standardised teacher code). Adam, cosine decay, fixed --steps, batch 512, fp32.

Every --eval-every steps (and at the end) the model is scored on the validation
pool vs the teacher's cached logits: detections (logit > 0) gained/lost on
ins_buzz, the other 12 live classes, mean |logit error|. Outputs in
`.local/distill/runs/<name>/`: student_mel.keras, aux.npz, curve.json.
"""
import argparse
import json
import os
import sys
import threading
import queue
import time

import tensorflow as tf  # noqa: F401  (must come first, see CLAUDE.md)
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import student as st  # noqa: E402
import student_init as si  # noqa: E402
import frontends as fes  # noqa: E402
import dpaths as D  # noqa: E402
import store  # noqa: E402

LOCAL = D.LOCAL
SP = D.spec()                       # the teacher's classes; classes without an activation center are dead
CLASSES = SP.classes
BUZZ = SP.buzz
DEAD = tuple(SP.dead)               # no center: above 0 everywhere, zero weight
LIVE = SP.live
N_CLASSES = SP.n_classes


def set_active(names=None):
    """Which teacher classes the student is trained on and judged against. Default: all of them (dead ones
    carry zero loss, as always). `--classes a,b,c`: a student whose head has only those outputs, in that
    order; the rest of the teacher's logits are neither fitted nor read. Positions below index the student's
    own output, `KEEP` maps them back to the teacher's columns."""
    global KEEP, ACT_CLASSES, ACT_DEAD, ACT_LIVE, ACT_BUZZ
    if names:
        bad = [n for n in names if n not in CLASSES or CLASSES.index(n) in DEAD]
        if bad:
            sys.exit(f'--classes: {bad} not live classes of teacher {D.TEACHER} ({[CLASSES[i] for i in LIVE]})')
        if 'ins_buzz' not in names:
            sys.exit('--classes must include ins_buzz (the headline is buzz sensitivity)')
    KEEP = [CLASSES.index(n) for n in names] if names else list(range(N_CLASSES))
    ACT_CLASSES = [CLASSES[i] for i in KEEP]
    ACT_DEAD = [j for j, i in enumerate(KEEP) if i in DEAD]
    ACT_LIVE = [j for j in range(len(KEEP)) if j not in ACT_DEAD]
    ACT_BUZZ = ACT_CLASSES.index('ins_buzz')


set_active()
RUNG_ORDER = 'ABCDE'
ARCHS = {'a0.50': (0.5, 14), 'a0.50_d12': (0.5, 12), 'a0.375': (0.375, 14), 'a0.25': (0.25, 14)}   # name -> (alpha, depth)


def cache_root(arg=None):
    return arg or D.need_cache()


# ---------------------------------------------------------------- packing

slice_path = D.slice_path


def pack_fingerprint(cache, rung, frontend, n_avail):
    """Everything a packed shard set is derived from: a different teacher build, plan, front end
    definition, or a rung that has gained slices since (n_avail) means the pack is stale and is
    rebuilt (it is derived data)."""
    return store.fingerprint({'rung': rung, 'frontend': frontend, 'code_dim': SP.code_dim, 'n_avail': n_avail,
                              'teacher_sha': json.load(open(D.TEACHER_JSON))['onnx_sha256'],
                              'plan_sha': store.file_sha(os.path.join(cache, '_manifest', 'plan.csv')),
                              'mel': store.mel_fingerprint(frontend) if frontend != 'yamnet' else 'yamnet'})


def pack(cache, rung, out_dir=None, frontend='yamnet'):
    """npz -> contiguous shards. Returns the shard dir. Reused when meta.json's fingerprint matches what
    it would be built from now (a pack from before fingerprints is trusted); a stale pack is deleted and
    rebuilt. `mel` comes from the shared `_mel/<frontend>` level (or a pre-split cache's embedded /
    `_fe` copy, store.mel_path), targets from the teacher's cache; slices missing either are dropped
    together."""
    fe = None if frontend == 'yamnet' else fes.get(frontend)
    out = out_dir or os.path.join(D.SHARDS, rung if fe is None else f'{rung}__{frontend}')
    plan = pd.read_csv(os.path.join(cache, '_manifest', 'plan.csv'))
    if rung == 'V':
        plan = plan[plan['first_rung'] == 'V']
    else:
        ok = [r for r in RUNG_ORDER[:RUNG_ORDER.index(rung) + 1]]
        plan = plan[plan['first_rung'].isin(ok)]
    plan = plan.sort_values(['relpath', 'hour'])
    avail = []                                   # (plan row, targets npz, mel npz) for slices with both
    for r in plan.itertuples():
        p = slice_path(cache, r.relpath, r.hour)
        pf = store.mel_path(frontend, r.relpath, r.hour, cache) if os.path.exists(p) else None
        if pf is not None:
            avail.append((r, p, pf))
    fp = pack_fingerprint(cache, rung, frontend, len(avail))
    if os.path.exists(os.path.join(out, 'meta.json')):
        have = json.load(open(os.path.join(out, 'meta.json'))).get('fingerprint')
        if have in (None, fp):
            return out
        print(f'[pack {rung}] {out} is stale (fingerprint {have} != {fp}): rebuilding', flush=True)
        import shutil
        shutil.rmtree(out)
    cap = int(plan['n_frames'].sum())
    os.makedirs(out, exist_ok=True)
    mel_shape = (96, 64) if fe is None else (fe.frames, fe.bands, fe.n_channels)
    fm = {k: np.lib.format.open_memmap(os.path.join(out, k + '.npy'), mode='w+', dtype=dt, shape=(cap,) + sh)
          for k, dt, sh in (('mel', np.float16, mel_shape), ('code', np.float16, (SP.code_dim,)),
                            ('logits', np.float32, (N_CLASSES,)))}
    n, sl = 0, []
    missing = len(plan) - len(avail)
    t0 = time.time()
    for i, (r, p, pf) in enumerate(avail):
        with np.load(p) as z:
            m = len(z['logits'])
            mel = store.load_mel(pf)
            assert len(mel) == m, (pf, len(mel), m)
            fm['mel'][n:n + m] = mel
            fm['code'][n:n + m] = z['code']
            fm['logits'][n:n + m] = z['logits']
        sl.append((r.deployment, n, m))
        n += m
        if i % 500 == 0:
            print(f'[pack {rung}] {i}/{len(plan)} slices, {n} frames, {time.time() - t0:.0f} s', flush=True)
    for v in fm.values():
        v.flush()
    # frames beyond n are unused; meta records the real count (written last: its presence is "done")
    dep = [d for d, _, _ in sl]
    json.dump({'frames': n, 'slices': len(sl), 'missing_slices': missing, 'capacity': cap,
               'slice_index': sl, 'deployments': len(set(dep)), 'fingerprint': fp},
              open(os.path.join(out, 'meta.json'), 'w'))
    print(f'[pack {rung}] {n} frames from {len(sl)} slices ({missing} missing) -> {out}', flush=True)
    return out


class Shards:
    def __init__(self, d):
        self.meta = json.load(open(os.path.join(d, 'meta.json')))
        self.n = self.meta['frames']
        self.mel = np.load(os.path.join(d, 'mel.npy'), mmap_mode='r')
        self.code = np.load(os.path.join(d, 'code.npy'), mmap_mode='r')
        self.logits = np.load(os.path.join(d, 'logits.npy'), mmap_mode='r')

    def take(self, idx):
        idx = np.sort(idx)
        return (np.asarray(self.mel[idx], np.float32), np.asarray(self.code[idx], np.float32),
                np.asarray(self.logits[idx], np.float32))

    def code_stats(self, n=20000, seed=0):
        idx = np.sort(np.random.default_rng(seed).choice(self.n, min(n, self.n), replace=False))
        c = np.asarray(self.code[idx], np.float32)
        return c.mean(0), c.std(0) + 1e-3


def batches(sh, batch, steps, seed=0, depth=6):
    """Yield `steps` batches, frame-level permutation, prefetched on a thread."""
    q = queue.Queue(depth)

    def work():
        rng = np.random.default_rng(seed)
        perm, pos = rng.permutation(sh.n), 0
        for _ in range(steps):
            if pos + batch > sh.n:
                perm, pos = rng.permutation(sh.n), 0
            q.put(sh.take(perm[pos:pos + batch]))
            pos += batch
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while True:
        b = q.get()
        if b is None:
            return
        yield b


# ---------------------------------------------------------------- eval

def val_flips(model, val, limit=None, batch=512):
    """Detections vs the teacher's cached logits on the validation pool."""
    n = val.n if not limit else min(limit, val.n)
    idx = np.linspace(0, val.n - 1, n).astype(int) if limit else np.arange(val.n)
    S, T = [], []
    for i in range(0, n, batch):
        mel, _, lg = val.take(idx[i:i + batch])
        S.append(model(mel, training=False)[0].numpy())
        T.append(lg[:, KEEP])
    S, T = np.concatenate(S), np.concatenate(T)
    sp, tp = S > 0, T > 0
    gained = (sp & ~tp).sum(0)
    lost = (~sp & tp).sum(0)
    B = ACT_BUZZ
    others = [c for c in ACT_LIVE if c != B]
    out = {'frames': int(n), 'buzz_teacher': int(tp[:, B].sum()), 'buzz_student': int(sp[:, B].sum()),
           'buzz_gained': int(gained[B]), 'buzz_lost': int(lost[B]),
           'other_gained': int(gained[others].sum()), 'other_lost': int(lost[others].sum()),
           'mae_live': float(np.abs(S - T)[:, ACT_LIVE].mean()), 'mae_buzz': float(np.abs(S - T)[:, B].mean()),
           'per_class_flips': {ACT_CLASSES[c]: [int(gained[c]), int(lost[c])] for c in ACT_LIVE}}
    return out


def fmt_val(step, v):
    return (f'[val {step}] ins_buzz teacher {v["buzz_teacher"]} student {v["buzz_student"]} '
            f'gained {v["buzz_gained"]} lost {v["buzz_lost"]} | other classes gained '
            f'{v["other_gained"]} lost {v["other_lost"]} | mae live {v["mae_live"]:.3f} buzz '
            f'{v["mae_buzz"]:.3f} ({v["frames"]} frames)')


# ---------------------------------------------------------------- init

def ridge(x, y, lam):
    x1 = np.c_[x, np.ones(len(x), np.float32)].astype(np.float64)
    w = np.linalg.solve(x1.T @ x1 + lam * np.eye(x1.shape[1]), x1.T @ y.astype(np.float64))
    return w[:-1].astype(np.float32), w[-1].astype(np.float32)


RESUME_KEYS = ('rung', 'steps', 'name', 'alpha', 'depth', 'frontend', 'arch', 'loader', 'batch', 'lr', 'lam',
               'huber', 'init', 'init_frames', 'ridge', 'bn_momentum', 'seed', 'classes')


def ckpt_path(out):
    return os.path.join(out, 'ckpt.npz')


def save_ckpt(out, step, model, aux, opt, curve):
    """Everything a resume needs: model + aux weights (BN moving stats included), the optimizer's slots and
    iteration count (the cosine schedule is a function of it), and the curve so far. Atomic."""
    arrays = {'step': np.int64(step), 'curve': np.frombuffer(json.dumps(curve, default=float).encode(), np.uint8)}
    for tag, ws in (('m', model.get_weights()), ('a', aux.get_weights()),
                    ('o', [np.array(v.numpy() if hasattr(v, 'numpy') else v) for v in opt.variables])):
        arrays.update({f'{tag}{i}': w for i, w in enumerate(ws)})
    tmp = ckpt_path(out) + '.tmp'
    with open(tmp, 'wb') as f:
        np.savez(f, **arrays)
    os.replace(tmp, ckpt_path(out))


def load_resume_state(out, cli):
    """'done' if this run already finished, the checkpoint's arrays if it was interrupted, else None.
    A run dir left by other settings under the same name stops the run (a mix-up, not a resume)."""
    def same(prev):
        diff = {k: (prev[k], cli[k]) for k in cli if k in prev and prev[k] != cli[k]}
        if diff:
            sys.exit(f'{out} was trained with other settings {diff}: pick another --name or delete it')
    if os.path.exists(os.path.join(out, 'TRAIN_DONE')):
        cv = os.path.join(out, 'curve.json')
        if os.path.exists(cv):
            same(json.load(open(cv)).get('cli', {}))
        return 'done'
    if os.path.exists(ckpt_path(out)):
        z = dict(np.load(ckpt_path(out)))
        same(json.loads(bytes(z['curve']).decode()).get('cli', {}))
        return z
    return None


def restore(z, model, aux, opt):
    def group(tag):
        return [z[f'{tag}{i}'] for i in range(sum(1 for k in z if k[0] == tag and k[1:].isdigit()))]
    model.set_weights(group('m'))
    aux.set_weights(group('a'))
    for v, w in zip(opt.variables, group('o')):
        v.assign(w)
    return int(z['step']), json.loads(bytes(z['curve']).decode())


def build_and_init(a, tr, mu, sd, skip_fit=False):
    filters = st.widths_for(a.alpha, a.depth)
    model = st.build_student(filters, n_out=len(KEEP), input_type='mel', expose_code=True, name='student_mel',
                             frontend=a.frontend)
    for l in model.layers:                      # moving stats must track fast: the init's BN is
        if isinstance(l, tf.keras.layers.BatchNormalization):   # identity+bias (refit), not YAMNet's
            l.momentum = a.bn_momentum
    aux = tf.keras.layers.Dense(SP.code_dim, name='aux')
    aux.build((None, filters[-1]))
    if a.frontend != 'yamnet' and a.init == 'yamnet':
        # the layer-wise refit pairs student and YAMNet activations position by position; a different
        # front end puts different frequencies at those positions, so only selection + BN moments apply
        print('[init] front end != yamnet: init yamnet -> select (channel selection + BN recalibration)', flush=True)
        a.init = 'select'
    if skip_fit:                                # resuming: every weight comes from the checkpoint
        return model, aux
    sels = si.init_from_yamnet(model) if a.init in ('yamnet', 'select') else None
    rng = np.random.default_rng(1)
    idx = np.sort(rng.choice(tr.n, min(a.init_frames, tr.n), replace=False))
    mel, code, lg = tr.take(idx)
    lg = lg[:, KEEP]
    if a.init == 'select':
        si.recalibrate_bn(model, mel)
    elif a.init == 'yamnet':
        si.refit_layerwise(model, mel, sels)
    gap = np.concatenate([model(mel[i:i + 256], training=False)[1].numpy() for i in range(0, len(mel), 256)])
    # head + aux: ridge onto teacher logits / standardised code
    w, b = ridge(gap, lg, a.ridge)
    model.get_layer('logits').set_weights([w, b])
    w2, b2 = ridge(gap, (code - mu) / sd, a.ridge)
    aux.set_weights([w2, b2])
    return model, aux


# ---------------------------------------------------------------- train

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rung', default='A')
    ap.add_argument('--steps', type=int, required=True, help='fixed budget, stated up front')
    ap.add_argument('--name', required=True)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--depth', type=int, default=14)
    ap.add_argument('--frontend', default='yamnet', choices=list(fes.FRONTENDS),
                    help='spectrogram the student sees (frontends.py); needs cache_fe.py output for the rung and V')
    ap.add_argument('--arch', choices=list(ARCHS), help='named variant: sets alpha and depth')
    ap.add_argument('--loader', choices=['mem', 'stream'], default='mem',
                    help='mem: pack the rung locally (memmap); stream: shard-level shuffle buffer from <cache>/_shards')
    ap.add_argument('--buffer-gb', type=float, default=0, help='stream: shuffle buffer size (0 = auto from free RAM, max 8)')
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lam', type=float, default=0.1)
    ap.add_argument('--huber', type=float, default=1.0)
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--val-limit', type=int, default=40000, help='frames per periodic val pass (final is full)')
    ap.add_argument('--init', choices=['yamnet', 'select', 'random'], default='yamnet',
                    help='yamnet: channel selection + layer-wise refit; select: selection + BN recalibration only')
    ap.add_argument('--init-frames', type=int, default=12000)
    ap.add_argument('--ridge', type=float, default=1.0)
    ap.add_argument('--bn-momentum', type=float, default=0.9)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--shards', help='override shard dir for the train rung (selftest)')
    ap.add_argument('--val-shards', help='override shard dir for validation (selftest)')
    ap.add_argument('--halt-at', type=int, default=0, help=argparse.SUPPRESS)   # test hook, see test_distill.py
    ap.add_argument('--classes', default='',
                    help='comma-separated teacher classes to distil (must include ins_buzz); the student\'s head has '
                         'only these outputs. Default: all (dead classes at zero loss)')
    a = ap.parse_args()
    set_active(a.classes.split(',') if a.classes else None)
    a.keep_classes = ACT_CLASSES                         # recorded in curve.json for export_student

    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    tf.keras.utils.set_random_seed(a.seed)
    if a.arch:
        a.alpha, a.depth = ARCHS[a.arch]
    cache = None if a.shards else cache_root()
    out = os.path.join(D.RUNS, a.name)
    os.makedirs(out, exist_ok=True)
    cli = {k: v for k, v in vars(a).items() if k in RESUME_KEYS}
    resume = load_resume_state(out, cli)                 # None, or the checkpoint's (step, curve, arrays)
    if resume == 'done':
        print(f'[done] {out} already trained (TRAIN_DONE); delete it to retrain', flush=True)
        return
    if a.frontend != 'yamnet':
        assert a.loader == 'mem', 'front-end experiments train from the in-memory pack (rung B)'
    if a.loader == 'stream':
        import shards as sh
        tr = sh.StreamPool(cache, a.rung, a.buffer_gb, a.seed)
    else:
        tr = Shards(a.shards or pack(cache, a.rung, frontend=a.frontend))
    val = Shards(a.val_shards or pack(cache, 'V', frontend=a.frontend))
    total = getattr(tr, 'total', tr.n)
    print(f'[data] train rung {a.rung} ({a.loader}): {total} frames ({tr.meta["slices"]} slices, '
          f'{tr.meta["deployments"]} deployments); val {val.n} frames '
          f'({val.meta["deployments"]} deployments); steps {a.steps} x {a.batch} '
          f'= {a.steps * a.batch / total:.2f} passes; arch alpha {a.alpha} depth {a.depth} '
          f'frontend {a.frontend}; classes {ACT_CLASSES if a.classes else "all"}; lam {a.lam}', flush=True)
    mu, sd = tr.code_stats()
    model, aux = build_and_init(a, tr, mu, sd, skip_fit=resume is not None)
    sched = tf.keras.optimizers.schedules.CosineDecay(a.lr, a.steps, alpha=0.01)
    opt = tf.keras.optimizers.Adam(sched)
    tvars = model.trainable_variables + aux.trainable_variables
    opt.build(tvars)
    if resume is None:
        curve = {'args': vars(a), 'cli': cli, 'train': [], 'val': []}
        v0 = val_flips(model, val, a.val_limit)
        print(fmt_val(0, v0), flush=True)
        curve['val'].append({'step': 0, **v0})
        start = 0
    else:
        start, curve = restore(resume, model, aux, opt)
        print(f'[resume] {out}: continuing from step {start} of {a.steps}', flush=True)
    live = tf.constant([0. if j in ACT_DEAD else 1. for j in range(len(KEEP))])
    mu_t, sd_t = tf.constant(mu), tf.constant(sd)

    @tf.function
    def step(mel, code, lg):
        with tf.GradientTape() as tape:
            logits, gap = model(mel, training=True)
            hub = tf.keras.losses.huber(lg[..., None], logits[..., None], delta=a.huber)  # (B,15)
            hub = tf.reduce_sum(hub * live) / (tf.reduce_sum(live) * tf.cast(tf.shape(mel)[0], tf.float32))
            cmse = tf.reduce_mean(tf.square(aux(gap) - (code - mu_t) / sd_t))
            loss = hub + a.lam * cmse
        opt.apply_gradients(zip(tape.gradient(loss, tvars), tvars))
        return loss, hub, cmse

    def save():
        model.save(os.path.join(out, 'student_mel.keras'))
        np.savez(os.path.join(out, 'aux.npz'), w=aux.get_weights()[0], b=aux.get_weights()[1], mu=mu, sd=sd)
        json.dump(curve, open(os.path.join(out, 'curve.json'), 'w'))

    # a resumed run draws a fresh batch sequence (seed + start): statistically the same, not bit-identical
    left, seed = a.steps - start, a.seed + start
    gen = tr.batches(a.batch, left, seed) if hasattr(tr, 'batches') else batches(tr, a.batch, left, seed)
    t0, acc = time.time(), []
    for i, (mel, code, lg) in enumerate(gen, start + 1):
        loss, hub, cmse = step(tf.constant(mel), tf.constant(code), tf.constant(lg[:, KEEP]))
        acc.append([float(loss), float(hub), float(cmse)])
        if i % 100 == 0 or i == a.steps:
            m = np.mean(acc, 0)
            acc = []
            curve['train'].append({'step': i, 'loss': m[0], 'huber': m[1], 'code_mse': m[2]})
            if i % 500 == 0 or i == a.steps:
                print(f'[train {i}/{a.steps}] loss {m[0]:.4f} huber {m[1]:.4f} code_mse {m[2]:.4f} '
                      f'{(i - start) / (time.time() - t0):.1f} step/s', flush=True)
        if i % a.eval_every == 0 and i != a.steps:
            v = val_flips(model, val, a.val_limit)
            print(fmt_val(i, v), flush=True)
            curve['val'].append({'step': i, **v})
            save()
            save_ckpt(out, i, model, aux, opt, curve)
            if a.halt_at == i:                            # test hook: simulate a crash right after a checkpoint
                print(f'[halt] --halt-at {i}', flush=True)
                sys.exit(3)
    v = val_flips(model, val)
    print(fmt_val(a.steps, v) + ' FINAL', flush=True)
    curve['val'].append({'step': a.steps, 'final': True, **v})
    save()
    open(os.path.join(out, 'TRAIN_DONE'), 'w').close()
    if os.path.exists(ckpt_path(out)):
        os.remove(ckpt_path(out))
    print(f'[done] {out}', flush=True)


if __name__ == '__main__':
    main()
