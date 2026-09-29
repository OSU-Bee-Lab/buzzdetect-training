"""Move data from the older layouts to the current one (dpaths.py). One-time, renames only.

    python 05_distill/migrate_layout.py                    # dry run: print every move
    python 05_distill/migrate_layout.py --apply            # do it (twice, if the first step applies: see below)

    .local/distill/                                        -> 05_distill/data/     (the data root moved out of .local)
    <data>/{runs,models,eval,shards,ladder.jsonl}          -> <data>/<teacher>/    (single-teacher -> per-teacher)
    <data>/{arch,arch_fe,arch_fe2}                         -> <data>/_shared/
    <distill_cache>/<teacher>/_fe/<spec>                   -> <distill_cache>/_mel/<spec>     (shared by all teachers)

The first line renames the whole tree in one step; the layout moves under it are then planned against the
new root, so a checkout still on the oldest layout runs `--apply` twice (the second run does the rest).

`<teacher>` is the default (or --teacher): the data predates teachers, so it all belongs to the one teacher
it was made for. Everything is an os.rename on one filesystem (instant, nothing copied) and idempotent: a
destination that already exists is left alone and reported. `<distill_cache>/_shared/durations.csv` needs no
migration (store.durations_csv seeds it from the teacher's own copy on first use), and a `_mel/<spec>` that
arrives without a fingerprint.json is adopted on first use.

Refuses to run while a distillation job is alive (a chain's scripts would lose the paths under them);
--force overrides that. Logs and CACHE_*_DONE markers travel with the tree and are otherwise left alone.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dpaths as D  # noqa: E402

LEGACY_LOCAL = os.path.join(D.MAIN, '.local', 'distill')      # where the data lived before 05_distill/data
LOCAL_TO_TEACHER = ['runs', 'models', 'eval', 'shards', 'ladder.jsonl']
LOCAL_TO_SHARED = ['arch', 'arch_fe', 'arch_fe2']
# the bracket keeps pgrep from matching its own command line
JOBS = r'[d]istill_train|[c]ache_fe\.py|[c]ache\.py|[c]hain_frontends|[c]hain_ladder|[e]val_folds|[e]xport_student|[s]hards\.py'


def running_jobs():
    r = subprocess.run(['pgrep', '-af', JOBS], capture_output=True, text=True)
    return [l for l in r.stdout.splitlines() if 'migrate_layout' not in l]


def plan_moves():
    moves = []
    if os.path.lexists(LEGACY_LOCAL) and not os.path.lexists(D.LOCAL_ROOT):
        return [(LEGACY_LOCAL, D.LOCAL_ROOT)]       # everything else is planned against the new root afterwards
    for name in LOCAL_TO_TEACHER:
        moves.append((os.path.join(D.LOCAL_ROOT, name), os.path.join(D.LOCAL, name)))
    for name in LOCAL_TO_SHARED:
        moves.append((os.path.join(D.LOCAL_ROOT, name), os.path.join(D.SHARED, name)))
    fe = os.path.join(D.CACHE, '_fe')
    if os.path.isdir(fe):
        for spec in sorted(os.listdir(fe)):
            moves.append((os.path.join(fe, spec), os.path.join(D.MEL_ROOT, spec)))
    return moves


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--teacher', help='the teacher the existing data belongs to (default: the current default)')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--force', action='store_true', help='run even though a distillation job is alive')
    a = ap.parse_args()
    if a.teacher and a.teacher != D.TEACHER:
        sys.exit(f'set DISTILL_TEACHER={a.teacher} (dpaths reads it at import)')
    D.need_cache()
    jobs = running_jobs()
    if jobs and a.apply and not a.force:
        print('distillation jobs are running; their scripts read the old paths:\n  ' + '\n  '.join(j[:140] for j in jobs))
        sys.exit('wait for them, or --force')
    n = 0
    for src, dst in plan_moves():
        if not os.path.lexists(src):
            continue
        if os.path.lexists(dst):
            print(f'  keep      {dst} exists; {src} left where it is')
            continue
        print(f'  {"move" if a.apply else "would move"}  {src}\n         -> {dst}')
        n += 1
        if a.apply:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.rename(src, dst)
    fe = os.path.join(D.CACHE, '_fe')
    if a.apply and os.path.isdir(fe) and not os.listdir(fe):
        os.rmdir(fe)
    print(f'{n} move(s) {"done" if a.apply else "planned (--apply to do them)"}')


if __name__ == '__main__':
    main()
