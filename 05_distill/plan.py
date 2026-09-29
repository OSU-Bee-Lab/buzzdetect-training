"""Plan the distillation cache: which 1-minute slices exist and which rung each joins.

Enumerates deployment audio (blacklist.py's rules), gets each file's duration
(ffprobe, parallel, cached in <cache>/_manifest/durations.csv), builds every
slice (62 frames = 59.52 s at h*3600 s; a file shorter than that gives one
short slice at 0 with floor(dur/0.96) frames), ranks them deterministically
(u = sha1(relpath:h) in [0,1)), and assigns nested rungs A<B<C<D with a
per-deployment floor. 10% of deployments (sha1 of the deployment) are a
validation pool: never in A-D, sampled separately as rung V.

Writes <cache>/_manifest/plan.csv (+ blacklist.txt) and prints the rung table. The ffprobe
durations are shared across teachers (<distill_cache>/_shared/durations.csv), so a second
teacher's plan never probes the audio tree again. Run (no GPU):
    python 05_distill/plan.py            # any python with numpy-free stdlib
Rung targets (hours) and floors are the RUNGS constant below.

An existing plan.csv is kept (like a set's config_extract.json): the cached slices belong
to *this* plan, so it is only rewritten with --replan (new audio on the drive, changed
RUNGS). The cache is keyed by (relpath, hour) and the rank is a hash of that key, so a
replan reuses every cached slice that stays in the plan.
"""
import argparse, bisect, collections, csv, hashlib, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blacklist as bl  # noqa: E402
import dpaths as D  # noqa: E402
import store  # noqa: E402

FRAME_S = 0.96
SLICE_FRAMES = 62
SLICE_S = SLICE_FRAMES * FRAME_S          # 59.52
MIN_FRAMES = 10                            # shorter files are skipped
VAL_FRAC = 0.10
# rung -> (target hours, per-deployment floor). D takes everything (target None).
RUNGS = [('A', 50, 2), ('B', 200, 4), ('C', 800, 8), ('D', None, 8)]
V_TARGET_H, V_FLOOR = 100, 4


def u_of(s):
    return int(hashlib.sha1(s.encode()).hexdigest()[:13], 16) / float(16 ** 13)


def is_val(dep):
    return u_of('deployment:' + dep) < VAL_FRAC


def probe(path):
    try:
        r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of',
                            'csv=p=0', path], capture_output=True, text=True, timeout=120)
        return float(r.stdout.strip().split()[0])
    except Exception:
        return None


def durations(files, root, cache_csv, workers):
    """files: [(rel, size)] -> {rel: dur}; cached by (rel, size)."""
    have = {}
    if os.path.isfile(cache_csv):
        for r in csv.DictReader(open(cache_csv)):
            have[r['relpath']] = (int(r['size']), float(r['dur']) if r['dur'] else None)
    todo = [(r, s) for r, s in files if r not in have or have[r][0] != s]
    print(f'durations: {len(files) - len(todo)} cached, probing {len(todo)}', flush=True)
    if todo:
        with ThreadPoolExecutor(workers) as ex:
            for i, ((r, s), d) in enumerate(zip(todo, ex.map(lambda t: probe(os.path.join(root, t[0])), todo))):
                have[r] = (s, d)
                if (i + 1) % 5000 == 0:
                    print(f'  probed {i + 1}/{len(todo)}', flush=True)
        os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
        with open(cache_csv, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['relpath', 'size', 'dur'])
            for r in sorted(have):
                w.writerow([r, have[r][0], '' if have[r][1] is None else repr(have[r][1])])
    return {r: have[r][1] for r, _ in files}


def slices_of(rel, dur):
    """[(hour, start_s, n_frames)]"""
    out = []
    if dur is None or dur < MIN_FRAMES * FRAME_S:
        return out
    if dur < SLICE_S + 0.5:
        return [(0, 0.0, min(SLICE_FRAMES, int((dur - 0.05) / FRAME_S)))]
    h = 0
    while h * 3600 + SLICE_S + 0.5 <= dur:
        out.append((h, float(h * 3600), SLICE_FRAMES)); h += 1
    return out


def choose(items, target_h, floor):
    """items: [(rank, dep, hours)] -> set of indices in the rung: rank<f or
    among the deployment's `floor` lowest ranks; f bisected to hit target_h
    (None = everything)."""
    order = sorted(range(len(items)), key=lambda i: items[i][0])
    floored = set()
    cnt = collections.Counter()
    for i in order:
        d = items[i][1]
        if cnt[d] < floor:
            floored.add(i); cnt[d] += 1
    if target_h is None:
        return set(order), 1.0
    def sel(f):
        return floored | {i for i in order if items[i][0] < f}
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if sum(items[i][2] for i in sel(mid)) / 3600 < target_h: lo = mid
        else: hi = mid
    return sel(hi), hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio-root', default=D.AUDIO_ROOT)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--replan', action='store_true', help='rewrite an existing plan.csv')
    a = ap.parse_args()
    D.need_cache()
    man = D.MANIFEST
    os.makedirs(man, exist_ok=True)
    if os.path.exists(D.PLAN) and not a.replan:
        print(f'{D.PLAN} exists: keeping it (--replan to rewrite)')
        return

    black = bl.blacklist_dirs()
    open(os.path.join(man, 'blacklist.txt'), 'w').write('\n'.join(black) + '\n')
    bset = set(black)
    files = [(r, s) for r, s in bl.walk_audio(a.audio_root) if not bl.excluded(r, bset)]
    print(f'files kept {len(files)}', flush=True)
    dur = durations(files, a.audio_root, store.durations_csv(), a.workers)
    bad = [r for r, d in dur.items() if d is None]
    print(f'unprobeable files {len(bad)}; too short (<{MIN_FRAMES} frames) '
          f'{sum(1 for d in dur.values() if d is not None and d < MIN_FRAMES * FRAME_S)}')

    rows = []   # relpath, hour, start_s, n_frames, deployment, rank
    for r, _ in files:
        dep = bl.deployment(r)
        for h, st, nf in slices_of(r, dur[r]):
            rows.append((r, h, st, nf, dep, u_of(f'{r}:{h}')))
    print(f'slices {len(rows)}, deployments {len({x[4] for x in rows})}', flush=True)

    val = [i for i, x in enumerate(rows) if is_val(x[4])]
    trn = [i for i, x in enumerate(rows) if not is_val(x[4])]
    first = {}
    hrs = lambda i: rows[i][3] * FRAME_S
    prev = set()
    table = []
    for name, tgt, floor in RUNGS:
        items = [(rows[i][5], rows[i][4], hrs(i)) for i in trn]
        idx, f = choose(items, tgt, floor)
        sel = {trn[j] for j in idx}
        assert prev <= sel, 'rungs not nested'
        for i in sel - prev:
            first[i] = name
        prev = sel
        table.append((name, len(sel), sum(hrs(i) for i in sel) / 3600, len({rows[i][4] for i in sel}), f, floor))
    items = [(rows[i][5], rows[i][4], hrs(i)) for i in val]
    idx, f = choose(items, V_TARGET_H, V_FLOOR)
    vsel = {val[j] for j in idx}
    for i in vsel:
        first[i] = 'V'
    table.append(('V', len(vsel), sum(hrs(i) for i in vsel) / 3600, len({rows[i][4] for i in vsel}), f, V_FLOOR))

    with open(os.path.join(man, 'plan.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['relpath', 'hour', 'start_s', 'deployment', 'rank', 'first_rung', 'n_frames'])
        for i in sorted(first, key=lambda i: (rows[i][0], rows[i][1])):
            r, h, st, nf, dep, u = rows[i]
            w.writerow([r, h, st, dep, f'{u:.8f}', first[i], nf])
    print(f'\nrung  slices    hours  deployments  f_rung  min_per_deploy  (cumulative; V = validation pool, separate)')
    for name, n, h, nd, f, fl in table:
        print(f'{name:>4} {n:8d} {h:8.1f} {nd:12d} {f:8.5f} {fl:8d}')
    ntrn = len({rows[i][4] for i in trn}); nval = len({rows[i][4] for i in val})
    print(f'deployments: train pool {ntrn}, validation pool {nval}; '
          f'all-slice hours: train {sum(hrs(i) for i in trn)/3600:.0f}, val {sum(hrs(i) for i in val)/3600:.0f}')


if __name__ == '__main__':
    main()
