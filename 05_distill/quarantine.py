"""Move runs out of the live data root into a quarantine dir, with their ladder rows; nothing is deleted.

    python 05_distill/quarantine.py --tag contaminated_2026-10-02 --keep fe_C_fast32h16_ [--dry-run]

Every run under runs/ is moved unless its name starts with a --keep prefix (repeatable) or `test_`:
runs/<n>, models/<n> and eval/<n> go to <teacher data>/_<tag>/{runs,models,eval}/<n>, and its ladder.jsonl
rows to _<tag>/ladder.jsonl. _<tag>/manifest.json lists each moved run with its curve.json args, which is
what chain_clean.sh retrains from. Idempotent: a rerun moves what is still live and appends to the manifest.

Written for 2026-10-04: packs without a fingerprint predated the 2026-10-02 SeeNote blacklist, so every
student except the fast32h16 rung-C ones (C__fast32h16 was packed after it) trained on test-set audio
(`Luke - Pollinator Habitat/2025-07-11/gru`); FRONTENDS.md, "Update 2026-10-04". No GPU, no TensorFlow.
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpaths as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--keep', action='append', default=[], help='run-name prefix to leave live (repeatable)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    q = os.path.join(D.LOCAL, f'_{a.tag}')
    keep = tuple(a.keep) + ('test_',)
    names = sorted(n for n in os.listdir(D.RUNS) if not n.startswith(keep))
    print(f'[quarantine] {len(names)} runs -> {q}; kept: {sorted(n for n in os.listdir(D.RUNS) if n.startswith(keep))}')
    if a.dry_run or not names:
        print('\n'.join(names))
        return
    man_path = os.path.join(q, 'manifest.json')
    man = json.load(open(man_path)) if os.path.exists(man_path) else {}
    for n in names:
        cur = os.path.join(D.RUNS, n, 'curve.json')
        man[n] = {'args': json.load(open(cur)).get('args', {}) if os.path.exists(cur) else {},
                  'done': os.path.exists(os.path.join(D.RUNS, n, 'TRAIN_DONE'))}
    os.makedirs(q, exist_ok=True)
    json.dump(man, open(man_path, 'w'), indent=1)          # manifest first: a crash below leaves it complete
    rows = [json.loads(line) for line in open(D.LADDER)]
    moved = [r for r in rows if r['name'] in names]
    with open(os.path.join(q, 'ladder.jsonl'), 'a') as f:
        f.writelines(json.dumps(r) + '\n' for r in moved)
    tmp = D.LADDER + '.tmp'
    with open(tmp, 'w') as f:
        f.writelines(json.dumps(r) + '\n' for r in rows if r['name'] not in names)
    os.replace(tmp, D.LADDER)
    for n in names:
        for kind, root in (('runs', D.RUNS), ('models', D.MODELS), ('eval', D.EVAL)):
            src = os.path.join(root, n)
            if os.path.exists(src):
                dst = os.path.join(q, kind, n)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.exists(dst):
                    sys.exit(f'{dst} exists already; not overwriting')
                shutil.move(src, dst)
    print(f'[quarantine] moved {len(names)} runs and {len(moved)} ladder rows; {len(rows) - len(moved)} rows stay live')


if __name__ == '__main__':
    main()
