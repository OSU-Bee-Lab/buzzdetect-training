"""Distil the teacher into the width-alpha student, on the cached teacher targets.

    tools/launch_job.sh <log> -- conda run -n buzzdetect-train python \
        05_distill/distill_train.py --rung A --steps 6000 --name a05_A

Data: `<cache>/_manifest/plan.csv` says which slices belong to a rung (first_rung
<= rung in A<B<C<D; `V` is the held-out validation deployments). The npz files
(mel f16 (62,96,64), code f16 (62,2048), logits f32 (62,15)) are packed once
into contiguous local shards `.local/distill/shards/<rung>/{mel,code,logits}.npy`
(float16 memmaps, written in plan order so the HDD is read sequentially) and
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

MAIN = st.main_checkout()
LOCAL = os.path.join(MAIN, '.local', 'distill')
CLASSES = ['aambient_scraping', 'ambient_background', 'ambient_music', 'ambient_noise',
           'ambient_rain', 'ambient_thunder', 'animal', 'human', 'ins_buzz', 'ins_trill',
           'mech_auto', 'mech_hum', 'mech_machinery', 'mech_plane', 'mech_quadcopter']
BUZZ = CLASSES.index('ins_buzz')
DEAD = (0, 14)                      # no center: above 0 everywhere, zero weight
LIVE = [i for i in range(15) if i not in DEAD]
RUNG_ORDER = 'ABCDE'
ARCHS = {'a0.50': (0.5, 14), 'a0.50_d12': (0.5, 12), 'a0.375': (0.375, 14), 'a0.25': (0.25, 14)}   # name -> (alpha, depth)


def cache_root(arg=None):
    if arg:
        return arg
    import config
    if not config.DISTILL_CACHE:
        sys.exit('set distill_cache in paths.local.json')
    return config.DISTILL_CACHE


# ---------------------------------------------------------------- packing

def slice_path(cache, relpath, hour):
    return os.path.join(cache, os.path.splitext(relpath)[0], f'h{int(hour):06d}.npz')


def fe_path(cache, frontend, relpath, hour):
    return slice_path(os.path.join(cache, '_fe', frontend), relpath, hour)


def pack(cache, rung, out_dir=None, frontend='yamnet'):
    """npz -> contiguous shards. Returns the shard dir. Skips if meta.json exists.
    frontend != 'yamnet': `mel` comes from the `_fe/<frontend>` cache (cache_fe.py), targets from the
    main cache; slices missing from either are dropped together."""
    fe = None if frontend == 'yamnet' else fes.get(frontend)
    out = out_dir or os.path.join(LOCAL, 'shards', rung if fe is None else f'{rung}__{frontend}')
    if os.path.exists(os.path.join(out, 'meta.json')):
        return out
    plan = pd.read_csv(os.path.join(cache, '_manifest', 'plan.csv'))
    if rung == 'V':
        plan = plan[plan['first_rung'] == 'V']
    else:
        ok = [r for r in RUNG_ORDER[:RUNG_ORDER.index(rung) + 1]]
        plan = plan[plan['first_rung'].isin(ok)]
    plan = plan.sort_values(['relpath', 'hour'])
    cap = int(plan['n_frames'].sum())
    os.makedirs(out, exist_ok=True)
    mel_shape = (96, 64) if fe is None else (fe.frames, fe.bands, fe.n_channels)
    fm = {k: np.lib.format.open_memmap(os.path.join(out, k + '.npy'), mode='w+', dtype=dt, shape=(cap,) + sh)
          for k, dt, sh in (('mel', np.float16, mel_shape), ('code', np.float16, (2048,)),
                            ('logits', np.float32, (15,)))}
    n, missing, sl = 0, 0, []
    t0 = time.time()
    for i, r in enumerate(plan.itertuples()):
        p = slice_path(cache, r.relpath, r.hour)
        pf = p if fe is None else fe_path(cache, frontend, r.relpath, r.hour)
        if not (os.path.exists(p) and os.path.exists(pf)):
            missing += 1
            continue
        with np.load(p) as z:
            m = len(z['logits'])
            if fe is None:
                fm['mel'][n:n + m] = z['mel']
            else:
                with np.load(pf) as zf:
                    assert len(zf['mel']) == m, (pf, len(zf['mel']), m)
                    fm['mel'][n:n + m] = zf['mel']
            fm['code'][n:n + m] = z['code']
            fm['logits'][n:n + m] = z['logits']
        sl.append((r.deployment, n, m))
        n += m
        if i % 500 == 0:
            print(f'[pack {rung}] {i}/{len(plan)} slices, {n} frames, {time.time() - t0:.0f} s', flush=True)
    for v in fm.values():
        v.flush()
    # frames beyond n are unused; meta records the real count
    dep = [d for d, _, _ in sl]
    json.dump({'frames': n, 'slices': len(sl), 'missing_slices': missing, 'capacity': cap,
               'slice_index': sl, 'deployments': len(set(dep))},
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
        T.append(lg)
    S, T = np.concatenate(S), np.concatenate(T)
    sp, tp = S > 0, T > 0
    gained = (sp & ~tp).sum(0)
    lost = (~sp & tp).sum(0)
    out = {'frames': int(n), 'buzz_teacher': int(tp[:, BUZZ].sum()), 'buzz_student': int(sp[:, BUZZ].sum()),
           'buzz_gained': int(gained[BUZZ]), 'buzz_lost': int(lost[BUZZ]),
           'other_gained': int(gained[[c for c in LIVE if c != BUZZ]].sum()),
           'other_lost': int(lost[[c for c in LIVE if c != BUZZ]].sum()),
           'mae_live': float(np.abs(S - T)[:, LIVE].mean()), 'mae_buzz': float(np.abs(S - T)[:, BUZZ].mean()),
           'per_class_flips': {CLASSES[c]: [int(gained[c]), int(lost[c])] for c in LIVE}}
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


def build_and_init(a, tr, mu, sd):
    filters = st.widths_for(a.alpha, a.depth)
    model = st.build_student(filters, input_type='mel', expose_code=True, name='student_mel',
                             frontend=a.frontend)
    for l in model.layers:                      # moving stats must track fast: the init's BN is
        if isinstance(l, tf.keras.layers.BatchNormalization):   # identity+bias (refit), not YAMNet's
            l.momentum = a.bn_momentum
    aux = tf.keras.layers.Dense(2048, name='aux')
    aux.build((None, filters[-1]))
    if a.frontend != 'yamnet' and a.init == 'yamnet':
        # the layer-wise refit pairs student and YAMNet activations position by position; a different
        # front end puts different frequencies at those positions, so only selection + BN moments apply
        print('[init] front end != yamnet: init yamnet -> select (channel selection + BN recalibration)', flush=True)
        a.init = 'select'
    sels = si.init_from_yamnet(model) if a.init in ('yamnet', 'select') else None
    rng = np.random.default_rng(1)
    idx = np.sort(rng.choice(tr.n, min(a.init_frames, tr.n), replace=False))
    mel, code, lg = tr.take(idx)
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
    ap.add_argument('--cache')
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
    a = ap.parse_args()

    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    tf.keras.utils.set_random_seed(a.seed)
    if a.arch:
        a.alpha, a.depth = ARCHS[a.arch]
    cache = None if a.shards else cache_root(a.cache)
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
          f'frontend {a.frontend}', flush=True)
    mu, sd = tr.code_stats()
    model, aux = build_and_init(a, tr, mu, sd)
    out = os.path.join(LOCAL, 'runs', a.name)
    os.makedirs(out, exist_ok=True)
    curve = {'args': vars(a), 'train': [], 'val': []}
    v0 = val_flips(model, val, a.val_limit)
    print(fmt_val(0, v0), flush=True)
    curve['val'].append({'step': 0, **v0})

    sched = tf.keras.optimizers.schedules.CosineDecay(a.lr, a.steps, alpha=0.01)
    opt = tf.keras.optimizers.Adam(sched)
    live = tf.constant([0. if i in DEAD else 1. for i in range(15)])
    tvars = model.trainable_variables + aux.trainable_variables
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

    t0, acc = time.time(), []
    for i, (mel, code, lg) in enumerate((tr.batches(a.batch, a.steps, a.seed) if hasattr(tr, 'batches')
                                    else batches(tr, a.batch, a.steps, a.seed)), 1):
        loss, hub, cmse = step(tf.constant(mel), tf.constant(code), tf.constant(lg))
        acc.append([float(loss), float(hub), float(cmse)])
        if i % 100 == 0 or i == a.steps:
            m = np.mean(acc, 0)
            acc = []
            curve['train'].append({'step': i, 'loss': m[0], 'huber': m[1], 'code_mse': m[2]})
            if i % 500 == 0 or i == a.steps:
                print(f'[train {i}/{a.steps}] loss {m[0]:.4f} huber {m[1]:.4f} code_mse {m[2]:.4f} '
                      f'{i / (time.time() - t0):.1f} step/s', flush=True)
        if i % a.eval_every == 0 and i != a.steps:
            v = val_flips(model, val, a.val_limit)
            print(fmt_val(i, v), flush=True)
            curve['val'].append({'step': i, **v})
            save()
    v = val_flips(model, val)
    print(fmt_val(a.steps, v) + ' FINAL', flush=True)
    curve['val'].append({'step': a.steps, 'final': True, **v})
    save()
    print(f'[done] {out}', flush=True)


if __name__ == '__main__':
    main()
